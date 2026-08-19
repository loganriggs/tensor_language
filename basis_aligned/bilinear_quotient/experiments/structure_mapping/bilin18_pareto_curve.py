"""Pareto frontier trace: refit assignments at ranks 4, 16, 64 (same 4
constants; 8 refit low-rank maps at each rank), plus naive at the same ranks.
REGISTERED: (a) refit-r64 joint <= +1.40; (b) refit dominates naive at every
rank; (c) monotone: cost decreases with rank in both protocols.

Prior context -- reference submission v2: push toward the Pareto frontier. Improvements over
v1: (i) SEQUENTIAL REFIT -- each stand-in is fit on the model with upstream
stand-ins already installed (section 105's rescue); (ii) budgeted ranks -- the
four cheap layers get constants, the rest get rank-16 (still ~0.3M params).
REGISTERED: (a) refit assignment joint <= 1.8 nats (vs naive v1's 2.68) at
<= 0.4M params; (b) refitting buys >= 25% cost reduction vs the naive v1
assignment protocol at the same architecture.

Original v1 docstring: first reference submission for the Track-2 benchmark (BENCHMARK.md): a
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
     'bilin18_pareto_curve_results.json')

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
    def fit_rank_live_def(): pass
    def fit_rank_live(li,r):
        # capture (post-mix input, real mlp out) on the current hybrid
        xs=[];ys=[]
        for i in range(0,48,6):
            _,cap=PR.fwd_lin(FW[i:i+6,:256].to(DEV), want=li)
            xs.append(cap[0]); ys.append(cap[1])
        X=torch.cat(xs); Y=torch.cat(ys)
        bx=X.mean(0); by=Y.mean(0)
        if r==0: return {'W':torch.zeros(D,D,device=DEV),'bx':bx,'by':by}
        Xc=X-bx; Yc=Y-by
        lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
        W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                             Xc.T@Yc/Xc.shape[0])
        U,S,Vh=torch.linalg.svd(W)
        return {'W':U[:,:r]@torch.diag(S[:r])@Vh[:r],'bx':bx,'by':by}
    curve={}
    for r in (4,16,64):
        assign={li:fit_rank(li,0) for li in const_layers}
        for li in rank8_layers: assign[li]=fit_rank(li,r)
        PR.LINS=assign
        cn=ce()-base
        PR.LINS={}
        refit={}
        for li in sorted(const_layers+rank8_layers):
            rr=0 if li in const_layers else r
            PR.LINS=dict(refit)
            refit[li]=fit_rank_live(li,rr)
        PR.LINS=refit
        cr=ce()-base
        PR.LINS={}
        params=len(rank8_layers)*2*D*r
        curve[r]={'naive':cn,'refit':cr,'params':params}
        print(f'rank {r:3d}: naive +{cn:.3f} | refit +{cr:.3f} '
              f'({params/1e6:.2f}M)',flush=True)
    pa=curve[64]['refit']<=1.40
    pb=all(curve[r]['refit']<curve[r]['naive'] for r in curve)
    ks=sorted(curve)
    pc=all(curve[ks[i+1]]['refit']<=curve[ks[i]]['refit']+0.02 and
           curve[ks[i+1]]['naive']<=curve[ks[i]]['naive']+0.02
           for i in range(len(ks)-1))
    out={'curve':{str(k):v for k,v in curve.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),'base':base}
    print(f"(a) refit-r64 <= 1.40: {'HELD' if pa else 'FAILED'}")
    print(f"(b) refit dominates naive: {'HELD' if pb else 'FAILED'}")
    print(f"(c) monotone in rank: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
