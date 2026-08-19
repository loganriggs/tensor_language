"""MOVER PROFILE -- what does the IOI name-mover a14.h4 do on
NATURAL text? (Owner-graph family analysis: tie the task circuit
into the census graph.) Within-row mean-ablate head (14,4) over the
census grid, per-position dCE.
REGISTERED PREDICTIONS:
  (a) damage is concentrated: top-5% positions carry >=50% of total
      |dCE|;
  (b) among the 1% most-damaged positions, the modal class is a
      name/copy class (name, rep, or ind) -- the mover serves
      name/copy prediction in the wild, not just in IOI prompts;
  (c) >=1 census leaf implicates it: some leaf's member-mean |dCE|
      >= 0.3 (the task circuit joins the census dependence graph)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mover_profile_results.json'
LI,HD=14,4

@torch.no_grad()
def main():
    t0=time.time()
    mod2=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod2.apply_rotary_emb
    at=m.transformer.h[LI].attn
    def fh(mo_,args,out):
        y,v1r=out
        X=args[0]; v1=args[1] if args[1] is not None else v1r
        Bb,T=X.shape[0],X.shape[1]
        v=at.c_v(X).view(Bb,T,9,128)
        vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
        cos,sin=at.rotary(at.c_q(X).view(Bb,T,9,128))
        qf=F.rms_norm(at.c_q(X).view(Bb,T,9,128),(128,))
        kf=F.rms_norm(at.c_k(X).view(Bb,T,9,128),(128,))
        qf,kf=are(qf,cos,sin),are(kf,cos,sin)
        q2=F.rms_norm(at.c_q2(X).view(Bb,T,9,128),(128,))
        k2=F.rms_norm(at.c_k2(X).view(Bb,T,9,128),(128,))
        q2,k2=are(q2,cos,sin),are(k2,cos,sin)
        sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),kf.float())/128
        s2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),k2.float())/128
        pat=(sc*s2)*torch.tril(torch.ones(T,T,device=DEV))
        z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
        z[:,HD]=z[:,HD].mean(1,keepdim=True)
        yn=at.c_proj(z.transpose(1,2).contiguous()
                     .view(Bb,T,-1).to(X.dtype))
        return (yn,v1r)
    h=at.register_forward_hook(fh)
    d=cl.ce_sweep([h])-cl.base_ce()
    ad=d.abs()
    tot=float(ad.sum())
    k5=int(0.05*len(ad))
    top5=float(ad.topk(k5).values.sum())
    pa=top5>=0.5*tot
    print(f'total |dCE| {tot:.0f} | top-5% share {top5/tot:.2%}',
          flush=True)
    f=cl.surface_features()
    k1=int(0.01*len(ad))
    top1=ad.topk(k1).indices
    from collections import Counter
    cnt=Counter()
    for nm in f:
        if nm.startswith('class_'):
            cnt[nm]=int(f[nm][top1].sum())
    modal=cnt.most_common(5)
    print('class profile of top-1%:',modal,flush=True)
    pb=modal[0][0] in ('class_name','class_rep','class_ind')
    hits=[]
    for lf in cl.state()['leaves']:
        md=float(d[lf['member']].mean())
        if abs(md)>=0.3: hits.append((lf['tag'],round(md,3)))
    hits.sort(key=lambda kv:-abs(kv[1]))
    print('census leaves implicated:',hits[:8],flush=True)
    pc=len(hits)>=1
    exs=[]
    for gi in top1[:5].tolist():
        pre,tgt,_=cl.context(gi)
        exs.append({'context':pre[-60:],'target':tgt,
                    'dce':round(float(d[gi]),2)})
    out={'total_abs':round(tot,1),'top5_share':round(top5/tot,3),
         'class_top1pct':modal,'census_hits':hits[:20],
         'examples_top':exs,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"(a) top-5% >=50%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) modal class name/rep/ind: {'HELD' if pb else 'FAILED'}")
    print(f"(c) >=1 leaf |mean dCE|>=0.3: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
