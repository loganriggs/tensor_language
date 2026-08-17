"""Is the model a few nonlinear layers on a linear pipe? Section 103: individual
linearization of middle/tail layers is cheap (+0.015-0.08 each), front (L2) is
expensive (+0.23). Joint test: replace the MLPs of MANY layers at once with their
ridge-fitted linear maps.

Arms (held-out CE, rows 300-380): (i) each layer 5-17 individually (13 costs);
(ii) joint 5-17 (all thirteen at once); (iii) joint 6-10 (the middle block);
(iv) joint 5-17 PLUS L2.

REGISTERED PREDICTIONS: (a) subadditivity of linearization: joint(5-17) <= 1.3x
the sum of individual costs -- linearizing removes the quadratic machinery that
makes damages interact, so no product-law blowup; (b) the pipe is viable:
joint(5-17) cost <= 0.6 nats; (c) the front is different: adding L2 to the joint
adds >= 0.15 nats on top."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_effective_linearity import fwd_all
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_linear_pipe_results.json')
LINS={}

@torch.no_grad()
def ce_eval():
    hs=[]
    for li,mp in LINS.items():
        blk=m.transformer.h[li]; state={}
        def mkpre(state=state):
            return lambda mod,inp: state.__setitem__('x',inp[0].detach())
        def mkmlp(mp=mp,state=state):
            def hook(mod,i_,o_):
                x=state['x'].reshape(-1,D).float()
                return ((x-mp['bx'])@mp['W']+mp['by']).to(o_.dtype).view_as(o_)
            return hook
        hs.append(blk.register_forward_pre_hook(mkpre()))
        hs.append(blk.mlp.register_forward_hook(mkmlp()))
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
    global LINS
    t0=time.time()
    base=ce_eval()
    print(f'base {base:.4f}\n',flush=True)
    tri=[]
    for i in range(0,60,6):
        tri.append(fwd_all(FW[i:i+6,:257].to(DEV)))
    maps={}
    for li in [2]+list(range(5,18)):
        X=torch.cat([a[li] for a,b in tri]); Y=torch.cat([b[li] for a,b in tri])
        bx=X.mean(0); by=Y.mean(0)
        Xc=X-bx; Yc=Y-by
        lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
        W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                             Xc.T@Yc/Xc.shape[0])
        maps[li]={'W':W,'bx':bx,'by':by}
    indiv={}
    for li in range(5,18):
        LINS={li:maps[li]}
        indiv[li]=ce_eval()-base
        print(f'L{li:2d} alone: +{indiv[li]:.4f}',flush=True)
    LINS={li:maps[li] for li in range(5,18)}
    joint=ce_eval()-base
    LINS={li:maps[li] for li in range(6,11)}
    joint_mid=ce_eval()-base
    LINS={li:maps[li] for li in [2]+list(range(5,18))}
    joint_p2=ce_eval()-base
    LINS={}
    ssum=sum(indiv.values())
    pa=joint<=1.3*ssum; pb=joint<=0.6; pc=(joint_p2-joint)>=0.15
    out={'base':base,'individual':{str(k):v for k,v in indiv.items()},
         'sum_individual':ssum,'joint_5_17':joint,'joint_6_10':joint_mid,
         'joint_plus_L2':joint_p2,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc)}
    print(f'\nsum of individual: +{ssum:.4f} | joint 5-17: +{joint:.4f} | '
          f'middle block 6-10: +{joint_mid:.4f} | +L2: +{joint_p2:.4f}')
    print(f"(a) subadditive (<=1.3x sum): {'HELD' if pa else 'FAILED'} "
          f"({joint/max(ssum,1e-9):.2f}x)")
    print(f"(b) pipe viable (<=0.6): {'HELD' if pb else 'FAILED'}")
    print(f"(c) L2 adds >=0.15: {'HELD' if pc else 'FAILED'} (+{joint_p2-joint:.3f})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
