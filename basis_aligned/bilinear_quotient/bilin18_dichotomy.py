"""The depth-functional dichotomy (section 139): early components compute
content (deletion hurts hard tokens most), late components sharpen (deletion
hurts easy, relieves hard). Six components: attention of L2/L13/L14 (full
mean-ablation) and top-8 MLP output spans of L2/L14/L16. Per-token deltas,
quartiles by base loss. REGISTERED: (a) all late four (L13a, L14a, L14m, L16m)
show easy-mean > 0 AND hard-mean < easy-mean; (b) both early (L2a, L2m) show
hard-mean > easy-mean; (c) hard-minus-easy decreases with depth across the six.

Prior context -- signature of the late-attention harm (section 138). Is L14's negative the
section-98 pattern -- sharpening easy tokens at the cost of overshooting hard
ones? Per-token CE deltas for deleting L14's attention (and L13's as a
harmful-vs-helpful control), quartiles by the base model's per-token loss.
REGISTERED: (a) L14's deletion benefit >= 60% concentrated in the hardest
quartile; (b) easy quartile hurt or flat (mean delta >= -0.001); (c) L13's
deletion (agenuinely  harmful one) HURTS the hard quartile (opposite sign there).

Prior context -- replication of the L14 negative (section 137): deleting L14's entire
attention output improved held-out CE by 0.036 on rows 300-364. Replicate on
disjoint rows 384-448, with L10/L15/L16 (the other non-positives) and L13 (a
positive control) alongside. REGISTERED: (a) L14 keeps its sign; (b) magnitude
>= 40% of original; (c) L13 stays positive (the instrument discriminates).

Prior context -- the per-layer ATTENTION ablation profile -- the one component/operator cell the
program never measured (MLP writes are profiled exhaustively; attention writes
never). Per layer 0-17: replace the attention output with its training-rows mean
(mean-ablation, the standard operator), held-out CE damage.

REGISTERED PREDICTIONS: (a) dilution extends to attention: damage rank-correlates
with the attention write's share of the stream (Spearman >= 0.6 across layers
2-15, excluding the special ends); (b) L6 is an outlier above its share
(the section-134 cargo edge makes its attention unusually load-bearing);
(c) the front (L0-L2) carries the largest attention damages (context assembly
happens early -- patterns are contextual from the start, section 126)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_dichotomy_results.json')

@torch.no_grad()
def run_ce(ablate_li=None, att_mean=None, collect_li=None, per_token=False):
    tot,n=0.0,0; means={}; ces=[]
    rows=range(384,448,4) if collect_li is None else range(0,24,6)
    for i in rows:
        step=4 if collect_li is None else 6
        bb=FW[i:i+step,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B,T=idx.shape
        cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for li in range(18):
            blk=m.transformer.h[li]; a=blk.attn
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            hcur=F.rms_norm(x,(D,))
            def qk(l):
                z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,))
                return apply_rot(z,cosb,sinb)
            v=a.c_v(hcur).view(B,T,NH,HD)
            if v1 is None: v1=v
            v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
            q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
            s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/HD
            s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
            pat=(s1*s2).masked_fill(~mask,0.0)
            att=a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
            if collect_li is not None and li==collect_li:
                means.setdefault('a',[]).append(att.detach().reshape(-1,D).float())
            if ablate_li is not None and li==ablate_li:
                att=att_mean[None,None,:].to(att.dtype).expand_as(att)
            x=x+att
            xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
            x=x+mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
        if collect_li is None:
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            if per_token:
                ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                           reduction='none'))
            else:
                c=F.cross_entropy(lg.view(-1,lg.size(-1)),tg)
                tot+=float(c)*tg.numel(); n+=tg.numel()
    if collect_li is not None:
        A=torch.cat(means['a'])
        return A.mean(0), float((A-A.mean(0)).pow(2).sum(1).mean())
    if per_token: return torch.cat(ces)
    return tot/n

@torch.no_grad()
def main():
    t0=time.time()
    base=run_ce()
    print(f'base {base:.4f}\n',flush=True)
    ce0=run_ce(per_token=True)
    q=torch.quantile(ce0.float(),torch.tensor([0.25,0.75],device=DEV))
    easy=ce0<=q[0]; hard=ce0>=q[1]
    # MLP top-8 span ablation support
    def mlp_span(li):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        return Q,Ybar@Q
    def mlp_ablate_ce(li):
        Q,cbar=mlp_span(li)
        def hook(mod,i_,o_):
            c=o_.float()@Q
            return (o_-((c-cbar)@Q.T).to(o_.dtype))
        h=m.transformer.h[li].mlp.register_forward_hook(hook)
        import torch.nn.functional as F2
        ces=[]
        for i in range(384,448,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F2.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F2.rms_norm(x,(D,)))/30)).float()
            ces.append(F2.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                        reduction='none'))
        h.remove()
        return torch.cat(ces)
    comps=[]
    for li in (2,13,14):
        mu,_=run_ce(collect_li=li)
        ce1=run_ce(ablate_li=li,att_mean=mu,per_token=True)
        comps.append((f'L{li}attn',li,ce1))
    for li in (2,14,16):
        comps.append((f'L{li}mlp8',li,mlp_ablate_ce(li)))
    out={}
    for name,li,ce1 in comps:
        d=ce1-ce0
        emean=float(d[easy].mean()); hmean=float(d[hard].mean())
        out[name]={'depth':li,'easy_mean':emean,'hard_mean':hmean,
                   'h_minus_e':hmean-emean}
        print(f'{name:8s}: easy {emean:+.4f} | hard {hmean:+.4f} | '
              f'h-e {hmean-emean:+.4f}',flush=True)
    late=('L13attn','L14attn','L14mlp8','L16mlp8')
    early=('L2attn','L2mlp8')
    pa=all(out[n]['easy_mean']>0 and out[n]['hard_mean']<out[n]['easy_mean']
           for n in late)
    pb=all(out[n]['hard_mean']>out[n]['easy_mean'] for n in early)
    seq=sorted(out.values(),key=lambda r:r['depth'])
    he=[r['h_minus_e'] for r in seq]
    inv=sum(1 for i in range(len(he)-1) if he[i+1]>he[i])
    pc=inv<=1
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"\n(a) late four sharpen (easy+ hard<easy): {'HELD' if pa else 'FAILED'}")
    print(f"(b) early two content (hard>easy): {'HELD' if pb else 'FAILED'}")
    print(f"(c) h-e decreases with depth (<=1 inversion): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
