"""The causal complement of the nonlinearity map: section 101 says the middle
(L6-10, linear R^2 0.52-0.62) is where quadratic computation lives, and section
102 showed linearizing L17 costs only +0.10 base CE. If the map is causal,
linearizing a middle layer should be expensive.

Per layer li in (2,4,7,9,13,16): replace the MLP with its ridge-fitted linear map
(fit rows 0-60), measure held-out base CE jump. REGISTERED PREDICTIONS:
(a) every middle layer (7,9) costs >= 3x the L16 cost; (b) rank order of
linearization cost anti-correlates with the section-101 linear R^2 (Spearman <=
-0.7 across the six layers); (c) L4 (most linear of the front, 0.89) is the
cheapest front linearization."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_effective_linearity import fwd_all
D=1152
LAYERS=(2,4,7,9,13,16)
R2={2:0.695,4:0.892,7:0.548,9:0.601,13:0.731,16:0.974}
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_linearize_middle_results.json')
LIN={}

@torch.no_grad()
def ce_eval():
    hs=[]
    if LIN:
        li=LIN['li']; blk=m.transformer.h[li]; state={}
        hs.append(blk.register_forward_pre_hook(
            lambda mod,inp: state.__setitem__('x',inp[0].detach())))
        def mlp_hook(mod,i_,o_):
            x=state['x'].reshape(-1,D).float()
            return ((x-LIN['bx'])@LIN['W']+LIN['by']).to(o_.dtype).view_as(o_)
        hs.append(blk.mlp.register_forward_hook(mlp_hook))
    tot,n=0.0,0
    for i in range(300,380,4):
        b=FW[i:i+4,:257].to(DEV)
        loss=m(b[:,:-1].contiguous(), b[:,1:].contiguous())
        ntok=(b.shape[1]-1)*b.shape[0]
        tot+=float(loss)*ntok; n+=ntok
    for h in hs: h.remove()
    return tot/n

@torch.no_grad()
def main():
    global LIN
    t0=time.time()
    base=ce_eval()
    print(f'base {base:.4f}\n',flush=True)
    tri=[]
    for i in range(0,60,6):
        tri.append(fwd_all(FW[i:i+6,:257].to(DEV)))
    costs={}
    for li in LAYERS:
        X=torch.cat([a[li] for a,b in tri]); Y=torch.cat([b[li] for a,b in tri])
        bx=X.mean(0); by=Y.mean(0)
        Xc=X-bx; Yc=Y-by
        lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
        W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                             Xc.T@Yc/Xc.shape[0])
        LIN={'li':li,'W':W,'bx':bx,'by':by}
        costs[li]=ce_eval()-base
        LIN={}
        print(f'L{li:2d}: linearization cost +{costs[li]:.4f} (linear R^2 {R2[li]:.2f})',
              flush=True)
    a=torch.tensor([costs[li] for li in LAYERS])
    b=torch.tensor([R2[li] for li in LAYERS])
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std()
    sp=float((ra*rb).mean())
    pa=all(costs[li]>=3*costs[16] for li in (7,9))
    pb=sp<=-0.7
    pc=costs[4]==min(costs[li] for li in (2,4))
    out={'base':base,'costs':{str(k):v for k,v in costs.items()},
         'spearman_vs_r2':sp,'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) middle >=3x L16: {'HELD' if pa else 'FAILED'}")
    print(f"(b) cost anti-tracks R^2 (<=-0.7): {'HELD' if pb else 'FAILED'} ({sp:+.2f})")
    print(f"(c) L4 cheapest front: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
