"""First reference submission for the Track-2 benchmark (BENCHMARK.md): a
replacement program assigned from the program's own knowledge, scored JOINTLY
(never per-layer sums -- section 104's drift).

Step 1: individual rank-0 (constant-mean) cost for each layer 5-15; layers at
<= 0.05 get assigned CONSTANT (0 params); the rest plus L16 get RANK-8 linear
(2*1152*8 params each). L0-L4 and L17 stay real (the front is the model; L17 is
the interaction skin). Step 2: joint install of the assignment vs joint install
of all-full-linear on the same layers (the naive baseline).

REGISTERED PREDICTIONS: (a) the knowledge assignment's joint cost <= 1.2x the
all-full-linear joint cost while using <= 5% of its stand-in parameters;
(b) >= 6 of layers 5-15 qualify for constant."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_effective_linearity import fwd_all
import bilin18_pipe_refit as PR
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_reference_submission_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    tri=[]
    for i in range(0,48,6): tri.append(fwd_all(FW[i:i+6,:257].to(DEV)))
    def fit_rank(li,r):
        X=torch.cat([a[li] for a,b in tri]); Y=torch.cat([b[li] for a,b in tri])
        bx=X.mean(0); by=Y.mean(0)
        if r==0: return {'W':torch.zeros(D,D,device=DEV),'bx':bx,'by':by}
        Xc=X-bx; Yc=Y-by
        lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
        W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                             Xc.T@Yc/Xc.shape[0])
        if r>=D: return {'W':W,'bx':bx,'by':by}
        U,S,Vh=torch.linalg.svd(W)
        return {'W':U[:,:r]@torch.diag(S[:r])@Vh[:r],'bx':bx,'by':by}
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
    r0cost={}
    for li in range(5,16):
        PR.LINS={li:fit_rank(li,0)}
        r0cost[li]=ce()-base
        PR.LINS={}
        print(f'L{li:2d} rank-0 alone: +{r0cost[li]:.4f}',flush=True)
    const_layers=[li for li in range(5,16) if r0cost[li]<=0.05]
    rank8_layers=[li for li in range(5,17) if li not in const_layers]
    print(f'\nassignment: constant {const_layers} | rank-8 {rank8_layers}',flush=True)
    assign={}
    for li in const_layers: assign[li]=fit_rank(li,0)
    for li in rank8_layers: assign[li]=fit_rank(li,8)
    PR.LINS=assign
    joint_assign=ce()-base
    PR.LINS={li:fit_rank(li,1152) for li in range(5,17)}
    joint_full=ce()-base
    PR.LINS={}
    params_assign=len(rank8_layers)*2*D*8
    params_full=12*D*D
    pa=(joint_assign<=1.2*joint_full) and (params_assign<=0.05*params_full)
    pb=len(const_layers)>=6
    out={'r0cost':{str(k):v for k,v in r0cost.items()},
         'const_layers':const_layers,'rank8_layers':rank8_layers,
         'joint_assign':joint_assign,'joint_full_linear':joint_full,
         'params_assign':params_assign,'params_full':params_full,
         'pred_a':bool(pa),'pred_b':bool(pb),'base':base}
    print(f'joint: assignment +{joint_assign:.3f} ({params_assign/1e6:.2f}M params) '
          f'| all-full-linear +{joint_full:.3f} ({params_full/1e6:.1f}M params)')
    print(f"(a) <=1.2x cost at <=5% params: {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=6 constants: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
