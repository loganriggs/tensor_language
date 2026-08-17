"""Replacement-complexity ladder (user question, 2026-08-17 evening): when a
linear stand-in preserves a layer's function, how much of the map's structure is
needed? Ladder per layer (L1 front, L9 mid, L16 late): (r0) constant mean
output; rank-r ridge maps r in (1,8,64); full linear (1152). Params(r) ~
2*1152*r. This is also the core instrument of the fidelity-vs-complexity
benchmark: each rung is a (params, delta-CE) point on the Pareto curve.

REGISTERED PREDICTIONS: (a) mid/late layers (9,16) reach within 0.02 nats of
their full-linear cost by rank 64 (the needed structure is low-rank); (b) L1
does NOT (its linear part is high-rank: gap > 0.05 at rank 64); (c) rank-0
(constant mean = 'just the scale/mean') is far worse than rank-8 at every layer
(>= 3x the full-linear cost) -- the stand-ins are not merely preserving scale."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_effective_linearity import fwd_all
import bilin18_pipe_refit as PR
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_replacement_ladder_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    tri=[]
    for i in range(0,48,6): tri.append(fwd_all(FW[i:i+6,:257].to(DEV)))
    def fit_rank(li,r):
        X=torch.cat([a[li] for a,b in tri]); Y=torch.cat([b[li] for a,b in tri])
        bx=X.mean(0); by=Y.mean(0)
        if r==0:
            return {'W':torch.zeros(D,D,device=DEV),'bx':bx,'by':by}
        Xc=X-bx; Yc=Y-by
        lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
        W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                             Xc.T@Yc/Xc.shape[0])
        if r>=D: return {'W':W,'bx':bx,'by':by}
        U,S,Vh=torch.linalg.svd(W)
        Wr=U[:,:r]@torch.diag(S[:r])@Vh[:r]
        return {'W':Wr,'bx':bx,'by':by}
    def ce():
        tot,n=0.0,0
        for i in range(384,448,4):
            b=FW[i:i+4,:257].to(DEV)
            lg,_=PR.fwd_lin(b[:,:-1].contiguous())
            c=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
            tot+=float(c)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
        return tot/n
    PR.LINS={}
    base=ce()
    out={'base':base,'ladder':{}}
    for li in (1,9,16):
        row={}
        for r in (0,1,8,64,1152):
            PR.LINS={li:fit_rank(li,r)}
            row[r]=ce()-base
            PR.LINS={}
        out['ladder'][str(li)]=row
        print(f'L{li:2d}: r0 +{row[0]:.3f} | r1 +{row[1]:.3f} | r8 +{row[8]:.3f} '
              f'| r64 +{row[64]:.3f} | full +{row[1152]:.3f}',flush=True)
    la=out['ladder']
    pa=all(la[str(li)][64]-la[str(li)][1152]<=0.02 for li in (9,16))
    pb=(la['1'][64]-la['1'][1152])>0.05
    pc=all(la[str(li)][0]>=3*max(la[str(li)][1152],1e-3) for li in (1,9,16))
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"\n(a) mid/late low-rank sufficient (r64 within 0.02): {'HELD' if pa else 'FAILED'}")
    print(f"(b) L1 high-rank (gap >0.05 at r64): {'HELD' if pb else 'FAILED'}")
    print(f"(c) rank-0 >= 3x full (not just scale): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
