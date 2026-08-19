"""TOPK4 FRONTIER -- find the largest head set that runs on 4-read
code within +0.10 (377: whole stack +0.22, cheapest-80-at-k1 ~0).
Reuse head_code_frontier's per-head ordering (k=1 costs as ranking
proxy); evaluate joint 4-read code at N = 100, 120, 140, 162.
REGISTERED PREDICTIONS:
  (a) >=120 heads fit within +0.10 joint at k=4;
  (b) fresh at the largest passing N <=2x its grid cost;
  (c) curve reported."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'topk4_frontier_results.json'

byl_ref=[{}]

def code_hooks():
    import sys as s_
    hs=[]
    for li,hds in byl_ref[0].items():
        at=m.transformer.h[li].attn
        def fh(mo_,args,out,li=li,hds=hds,at=at):
            y,v1r=out
            X=args[0]; v1=args[1] if args[1] is not None else v1r
            are=s_.modules[type(at).__module__].apply_rotary_emb
            Bb,Tq=X.shape[0],X.shape[1]
            v=at.c_v(X).view(Bb,Tq,9,128)
            vm=(1-at.lamb)*v+at.lamb*(v1.view_as(v)
                                      if v1 is not None else v)
            cos,sin=at.rotary(at.c_q(X).view(Bb,Tq,9,128))
            qf=F.rms_norm(at.c_q(X).view(Bb,Tq,9,128),(128,))
            kf=F.rms_norm(at.c_k(X).view(Bb,Tq,9,128),(128,))
            qf,kf=are(qf,cos,sin),are(kf,cos,sin)
            q2=F.rms_norm(at.c_q2(X).view(Bb,Tq,9,128),(128,))
            k2=F.rms_norm(at.c_k2(X).view(Bb,Tq,9,128),(128,))
            q2,k2=are(q2,cos,sin),are(k2,cos,sin)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),kf.float())/128
            s2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),k2.float())/128
            pat=(sc*s2)*torch.tril(torch.ones(Tq,Tq,device=DEV))
            z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
            for hd in hds:
                p1=pat[:,hd]
                _,idx=p1.abs().topk(4,dim=-1)
                msk=torch.zeros_like(p1).scatter(-1,idx,1.0)
                z[:,hd]=torch.einsum('bqk,bkd->bqd',p1*msk,
                                     vm[:,:,hd].float())
            yn=at.c_proj(z.transpose(1,2).contiguous()
                         .view(Bb,Tq,-1).to(X.dtype))
            return (yn,v1r)
        hs.append(at.register_forward_hook(fh))
    return hs

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()
    import json as j9
    order=[tuple(map(int,k.split('.'))) for k in
           j9.load(open(PT+'head_code_frontier_results.json'))
           ['cheapest_order']]
    allh=[(li,hd) for li in range(18) for hd in range(9)]
    rest=[h for h in allh if h not in order]
    order=order+rest
    base=cl.ce_sweep([])
    pts={}
    bestN=0
    for N in (100,120,140,162):
        sel=order[:N]
        byl={}
        for li,hd in sel: byl.setdefault(li,[]).append(hd)
        byl_ref[0]=byl
        d=cl.ce_sweep(code_hooks())
        pts[N]=round(float((d-base).mean()),4)
        print(f'N={N}: {pts[N]:+.4f}',flush=True)
        if pts[N]<=0.10: bestN=N
    pa=bestN>=120
    fr=None
    if bestN:
        sel=order[:bestN]
        byl={}
        for li,hd in sel: byl.setdefault(li,[]).append(hd)
        byl_ref[0]=byl
        FRESH=cl.fresh_rows(120)
        bF=cl.ce_sweep([],tok=FRESH)
        cF=cl.ce_sweep(code_hooks(),tok=FRESH)
        fr=round(float((cF-bF).mean()),4)
    pb=fr is not None and fr<=2*max(pts.get(bestN,1e-3),1e-3)
    out={'points':pts,'bestN':bestN,'fresh_bestN':fr,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True}
    print(f'bestN {bestN} | fresh {fr}')
    print(f"(a) >=120 within +0.10: {'HELD' if pa else 'FAILED'}")
    print(f"(b) fresh <=2x grid: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
