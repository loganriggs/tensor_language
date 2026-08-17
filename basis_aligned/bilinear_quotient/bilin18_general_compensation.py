"""Generality of compound-damage compensation. Section 119 found the network
destructively interferes with joint perturbations at the 16-17 pair (curvature
would give 2.8x the observed excess). Is that pair special, or does the model
cancel compound damage everywhere? Six span pairs among tail layers
((5,9),(7,13),(9,16),(11,15),(13,17),(6,12)), PCA-8 spans, free final norm (the
full model as it runs): true joint excess vs synthetic excess with superposed
logit deltas. REGISTERED: (a) synthetic > true for >= 5/6 pairs; (b) median
cancellation 1 - true/synthetic >= 40%.

Prior context -- content-level reinterpretation of the tail profiles. Section 117: L17's span
damage is 82% norm-mediated, L16's 35%. Section 96 profiled every tail layer with
raw span-ablation CE -- how much of each was the norm channel? Recompute the
PCA-8 span damages for layers 5-17 with the final gain frozen to the same
model's no-damage per-token rms.

REGISTERED PREDICTIONS: (a) norm-mediated share grows with depth (Spearman >=
0.6 across 13 layers -- late spans hold more of the final vector's energy);
(b) the CONTENT-level damage ranking differs from the raw ranking (Kendall tau
<= 0.7 -- the norm channel reordered section 96's profile); (c) control: frozen
gain exact at zero damage (<= 0.002)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_general_compensation_results.json')

@torch.no_grad()
def fwd_arm(idx, span, freeze):
    B,T=idx.shape
    cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    def emb(): x=F.rms_norm(m.transformer.wte(idx),(D,)); return x,x,None
    xc,x0c,v1c=emb(); xh,x0h,v1h=emb()
    for li in range(18):
        blk=m.transformer.h[li]; a=blk.attn; mlp=blk.mlp
        def step(x,x0,v1,dmg):
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            hcur=F.rms_norm(x,(D,))
            def qk(l):
                z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,))
                return apply_rot(z,cosb,sinb)
            v=a.c_v(hcur).view(B,T,NH,HD)
            v1n=v if v1 is None else v1
            v=(1-a.lamb)*v+a.lamb*v1n.view_as(v)
            q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
            s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/HD
            s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
            pat=(s1*s2).masked_fill(~mask,0.0)
            x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
            xhat=F.rms_norm(x,(D,))
            mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            if dmg is not None and li==dmg[0]:
                Q,cbar=dmg[1]
                c=mo.float().reshape(-1,D)@Q
                mo=mo-((c-cbar)@Q.T).to(mo.dtype).view_as(mo)
            return x+mo,v1n
        xc,v1c=step(xc,x0c,v1c,None)
        xh,v1h=step(xh,x0h,v1h,span)
    if freeze:
        rms_c=xc.float().pow(2).mean(-1,keepdim=True).sqrt()
        xn=(xh.float()/rms_c.clamp_min(1e-8)).to(xh.dtype)
    else:
        xn=F.rms_norm(xh,(D,))
    return (30*torch.tanh(m.lm_head(xn)/30)).float()



@torch.no_grad()
def main():
    t0=time.time()
    spans={}
    for li in (5,6,7,9,11,12,13,15,16,17):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        spans[li]=(Q,Ybar@Q)
    PAIRS=[(5,9),(7,13),(9,16),(11,15),(13,17),(6,12)]
    def logits_arm(idx, dmglist):
        # fwd_arm handles one span; extend inline for two
        B,T=idx.shape
        cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for li in range(18):
            blk=m.transformer.h[li]; a=blk.attn; mlp=blk.mlp
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
            x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
            xhat=F.rms_norm(x,(D,))
            mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            if li in dmglist:
                Q,cbar=spans[li]
                c=mo.float().reshape(-1,D)@Q
                mo=mo-((c-cbar)@Q.T).to(mo.dtype).view_as(mo)
            x=x+mo
        return (30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
    res={}
    for A,Bl in PAIRS:
        num_t=num_s=0.; ce_acc={'b':0,'a':0,'c':0,'j':0,'s':0}; n=0
        for i in range(300,348,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            lb=logits_arm(idx,[]); la=logits_arm(idx,[A])
            lc=logits_arm(idx,[Bl]); lj=logits_arm(idx,[A,Bl])
            lsyn=lb+(la-lb)+(lc-lb)
            ntok=tg.numel(); n+=ntok
            for k,lg in (('b',lb),('a',la),('c',lc),('j',lj),('s',lsyn)):
                ce_acc[k]+=float(F.cross_entropy(lg.view(-1,lg.size(-1)),tg))*ntok
        for k in ce_acc: ce_acc[k]/=n
        exc_t=ce_acc['j']-ce_acc['a']-ce_acc['c']+ce_acc['b']
        exc_s=ce_acc['s']-ce_acc['a']-ce_acc['c']+ce_acc['b']
        canc=1-exc_t/exc_s if exc_s>1e-5 else float('nan')
        res[f'{A}-{Bl}']={'true':exc_t,'synthetic':exc_s,'cancel':canc}
        print(f'pair L{A}-L{Bl}: true {exc_t:+.4f} | synthetic {exc_s:+.4f} | '
              f'cancel {canc if canc==canc else 0:.0%}',flush=True)
    ok=sum(1 for r in res.values() if r['synthetic']>r['true'])
    cs=sorted(r['cancel'] for r in res.values() if r['cancel']==r['cancel'])
    medc=cs[len(cs)//2] if cs else float('nan')
    pa=ok>=5; pb=medc>=0.4
    out={'pairs':res,'n_synth_gt_true':ok,'median_cancellation':medc,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"\n(a) compensation general (>=5/6): {'HELD' if pa else 'FAILED'} ({ok}/6)")
    print(f"(b) median cancellation >=40%: {'HELD' if pb else 'FAILED'} "
          f"({medc if medc==medc else 0:.0%})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
