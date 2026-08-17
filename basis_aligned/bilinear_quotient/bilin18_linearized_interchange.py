"""Does the 16->17 product law live in L17's nonlinear residue? Section 101: L17's
MLP write is linearly predictable from its input at R^2 0.946, yet the 16->17
interchange is the model's strongest interaction. If the interaction is the
quadratic cross-term the composition arc always said it was, then REPLACING L17's
MLP with its fitted linear map should kill the excess: linear readers of summed
errors have no cross-term.

Measurement (composition-arc style): damage d16 = mean-ablate L16's top-8 span;
damage d17 = mean-ablate L17's top-8 span; excess = CE(joint) - CE(base) -
[CE(d16)-CE(base)] - [CE(d17)-CE(base)]. Computed twice: with the real L17 MLP,
and with L17's MLP replaced by its ridge-fitted linear map (fit on rows 0-60).

REGISTERED PREDICTIONS: (a) the real-model excess is positive (>= +0.003,
reproducing the interchange on these spans); (b) under linearized L17 the excess
drops by >= 70% (the interaction lives in the nonlinear residue). Control: the
linearized model's own base CE must stay within 0.15 nats of the real base (the
linear map is a faithful stand-in; if not, the comparison is void and says so)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_effective_linearity import fwd_all
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_linearized_interchange_results.json')

LIN17={}   # {'W':..., 'bx':..., 'by':...} -> replace L17 mlp write
SPANS={}   # {li:(Q,cbar)}

@torch.no_grad()
def ce_eval():
    hs=[]
    for li,(Q,cbar) in SPANS.items():
        def mk(Q=Q,cbar=cbar):
            def hook(mod,i_,o_):
                c=o_.float()@Q
                return (o_-((c-cbar)@Q.T).to(o_.dtype))
            return hook
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    if LIN17:
        blk=m.transformer.h[17]
        def lin_hook(mod,inp,o_):
            pass
        # replace via pre/post: we hook the mlp with a function of the BLOCK input
        # captured by a pre-hook on the block
        state={}
        h_pre=blk.register_forward_pre_hook(
            lambda mod,inp: state.__setitem__('x',inp[0].detach()))
        def mlp_hook(mod,i_,o_):
            x=state['x'].reshape(-1,D).float()
            pred=(x-LIN17['bx'])@LIN17['W']+LIN17['by']
            return pred.to(o_.dtype).view_as(o_)
        h_mlp=blk.mlp.register_forward_hook(mlp_hook)
        hs+= [h_pre,h_mlp]
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
    global LIN17
    t0=time.time()
    spans={}
    for li in (16,17):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        spans[li]=(Q,Ybar@Q)
    # fit L17 linear map on rows 0-60 (block input -> mlp write)
    ins=[]; mos=[]
    for i in range(0,60,6):
        a_,b_=fwd_all(FW[i:i+6,:257].to(DEV))
        ins.append(a_[17]); mos.append(b_[17])
    X=torch.cat(ins); Y=torch.cat(mos)
    bx=X.mean(0); by=Y.mean(0)
    Xc=X-bx; Yc=Y-by
    lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
    W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                         Xc.T@Yc/Xc.shape[0])
    out={}
    for tag,lin in (('real',{}),
                    ('linearized',{'W':W,'bx':bx,'by':by})):
        LIN17=lin
        SPANS.clear(); base=ce_eval()
        SPANS[16]=spans[16]; d16=ce_eval()-base
        SPANS.clear(); SPANS[17]=spans[17]; d17=ce_eval()-base
        SPANS[16]=spans[16]; joint=ce_eval()-base
        SPANS.clear()
        exc=joint-d16-d17
        out[tag]={'base':base,'d16':d16,'d17':d17,'joint':joint,'excess':exc}
        print(f'{tag:10s}: base {base:.4f} | d16 {d16:+.4f} | d17 {d17:+.4f} | '
              f'joint {joint:+.4f} | excess {exc:+.4f}',flush=True)
    LIN17={}
    faithful=abs(out['linearized']['base']-out['real']['base'])<=0.15
    pa=out['real']['excess']>=0.003
    drop=1-out['linearized']['excess']/out['real']['excess'] \
         if out['real']['excess']>1e-6 else float('nan')
    pb=faithful and pa and drop>=0.7
    out['ctrl_faithful']=bool(faithful); out['pred_a']=bool(pa)
    out['excess_drop']=drop; out['pred_b']=bool(pb)
    print(f"\ncontrol (linear stand-in within 0.15): {'HELD' if faithful else 'VIOLATED'}")
    print(f"(a) excess reproduces (>=+0.003): {'HELD' if pa else 'FAILED'}")
    print(f"(b) linearization kills >=70% of excess: {'HELD' if pb else 'FAILED'}"
          f" (drop {drop if drop==drop else 0:.0%})" if pa else
          "(b) moot -- no excess to kill")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
