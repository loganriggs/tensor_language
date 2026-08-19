"""Fresh-text replication of the kinship directionality (the relay's
statistical leg, section-190-era, published: attention components partner
with upstream MLPs 16/18). The atlas fingerprints were computed on text rows
384-448; recompute all 36 component fingerprints on FRESH rows 320-384
(never used for fingerprints), spans/means identical to the atlas
construction, and rerun the directionality stats.

REGISTERED PREDICTIONS: (a) upstream-or-same fraction >= 60% on the fresh
window (original 89%); (b) the certified cargo edge attn6~mlp5 ranks top-5
of all 324 cross-type pairs (original held); (c) token-shuffle null gives
40-60% upstream (no directionality without token identity)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_fingerprints import attn_mean, spearman, NH, HD
from tier2_model import rope_tables, apply_rot
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_kinship_fresh_results.json')
R0,R1=320,384

@torch.no_grad()
def per_token_fresh(mlp_span=None, attn_layer=None):
    hs=[]
    if mlp_span is not None:
        li,(Q,cbar)=mlp_span
        def hook(mod,i_,o_):
            c=o_.float()@Q
            return (o_-((c-cbar)@Q.T).to(o_.dtype))
        hs.append(m.transformer.h[li].mlp.register_forward_hook(hook))
    ces=[]
    for i in range(R0,R1,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B,T=idx.shape
        if attn_layer is None:
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        else:
            ali,amu=attn_layer
            cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
            cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
            mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for li2 in range(18):
                blk=m.transformer.h[li2]; a=blk.attn
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
                att=a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v)
                             .reshape(B,T,-1))
                if li2==ali:
                    att=amu[None,None,:].to(att.dtype).expand_as(att)
                x=x+att
                xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
                x=x+mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none'))
    for h in hs: h.remove()
    return torch.cat(ces)

@torch.no_grad()
def main():
    t0=time.time()
    ce0=per_token_fresh()
    fps={}
    for li in range(18):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
            accs.append(acc[0])
        Y=torch.cat(accs); Yb=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        fps[f'mlp{li}']=(per_token_fresh(mlp_span=(li,(Q,Yb@Q)))-ce0).cpu()
        print(f'mlp{li} done',flush=True)
    for li in range(18):
        mu=attn_mean(li)
        fps[f'attn{li}']=(per_token_fresh(attn_layer=(li,mu))-ce0).cpu()
        print(f'attn{li} done',flush=True)
    def top_partner(ak, shuffle=False, g=None):
        a=fps[ak].float()
        if shuffle:
            a=a[torch.randperm(len(a),generator=g)]
        best=None; bv=-1
        for li in range(18):
            s=abs(spearman(a,fps[f'mlp{li}'].float()))
            if s>bv: bv=s; best=li
        return best,bv
    ups=0; pairs=[]
    for ali in range(18):
        b,v=top_partner(f'attn{ali}')
        ups+=(b<=ali)
        print(f'attn{ali}: top mlp{b} |rho| {v:.3f} '
              f'{"UP" if b<=ali else "down"}',flush=True)
    for ali in range(18):
        for mli in range(18):
            s=abs(spearman(fps[f'attn{ali}'].float(),
                           fps[f'mlp{mli}'].float()))
            pairs.append((s,ali,mli))
    pairs.sort(reverse=True)
    rank=[i for i,(s,a,mm) in enumerate(pairs) if a==6 and mm==5][0]+1
    g=torch.Generator().manual_seed(0)
    ups_n=sum(1 for ali in range(18)
              if top_partner(f'attn{ali}',shuffle=True,g=g)[0]<=ali)
    pa=ups/18>=0.60; pb=rank<=5; pc=0.40<=ups_n/18<=0.60
    out={'upstream_frac':ups/18,'cargo_rank':rank,
         'null_frac':ups_n/18,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc)}
    print(f"\nupstream {ups}/18 ({ups/18:.0%}) | cargo edge rank {rank} | "
          f"null {ups_n}/18")
    print(f"(a) >=60%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) cargo top-5: {'HELD' if pb else 'FAILED'}")
    print(f"(c) null 40-60%: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
