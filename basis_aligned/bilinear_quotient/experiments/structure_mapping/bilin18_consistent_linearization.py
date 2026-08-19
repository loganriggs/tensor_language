"""Consistent-protocol rerun after the section-105 lambda-mixing correction. All
stand-ins are fit AND applied inside the same manual forward (post-mix input),
via the fwd_lin machinery from bilin18_pipe_refit.

Measurements: (1) individual linearization cost for layers 2,4,5,7,9,13,16,17;
(2) the 16->17 interchange excess (span-ablation composition, section 102 design)
under real vs consistently-linearized L17.

REGISTERED PREDICTIONS: (a2) individual costs <=0.1 for all layers except L2, and
L2 >= 3x the median of the others (the front-loading claim, retested on a clean
instrument); (b2) linearizing L17 kills >= 70% of the 16->17 excess (the section
102 conclusion, re-verified); (c2) the consistent-L17 stand-in's base cost is
<= 0.05 (tighter than the contaminated +0.10)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
import bilin18_pipe_refit as PR
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_consistent_linearization_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    PR.LINS={}
    base=PR.ce_eval()
    print(f'base {base:.4f}\n',flush=True)
    costs={}
    for li in (2,4,5,7,9,13,16,17):
        PR.LINS={li:PR.fit_layer(li)}
        costs[li]=PR.ce_eval()-base
        print(f'L{li:2d}: consistent cost +{costs[li]:.4f}',flush=True)
    lin17=PR.fit_layer(17) if 17 not in costs else None
    PR.LINS={17:PR.fit_layer(17)}
    c17=PR.ce_eval()-base
    # interchange excess: span ablations at 16/17 outputs, real vs linearized L17
    spans={}
    for li in (16,17):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        spans[li]=(Q,Ybar@Q)
    # patch spans inside fwd_lin: easiest -- wrap mo via LINS-like registry
    # extend: monkey-patch a SPANS dict into PR.fwd_lin by post-processing mo.
    # Simpler: use hooks on the real model for span ablation ONLY when L17 is
    # real; for the linearized arm we ablate the stand-in's output by composing
    # the projection into the linear map (exact for a linear map).
    def ce_hooked(span_lis, lin17=None):
        if lin17 is None:
            hs=[]
            for li in span_lis:
                Q,cbar=spans[li]
                def mk(Q=Q,cbar=cbar):
                    def hook(mod,i_,o_):
                        c=o_.float()@Q
                        return (o_-((c-cbar)@Q.T).to(o_.dtype))
                    return hook
                hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
            PR.LINS={}
            ce=PR.ce_eval()
            for h in hs: h.remove()
            return ce
        # linearized L17: ablate span-17 by projecting the linear map's output
        mp=dict(lin17)
        if 17 in span_lis:
            Q,cbar=spans[17]
            P=torch.eye(D,device=DEV)-Q@Q.T
            mp={'W':mp['W']@P,'bx':mp['bx'],
                'by':(mp['by']-((mp['by']@Q)-cbar)@Q.T)}
        hs=[]
        if 16 in span_lis:
            Q,cbar=spans[16]
            def hook16(mod,i_,o_):
                c=o_.float()@Q
                return (o_-((c-cbar)@Q.T).to(o_.dtype))
            hs.append(m.transformer.h[16].mlp.register_forward_hook(hook16))
        PR.LINS={17:mp}
        ce=PR.ce_eval()
        for h in hs: h.remove()
        PR.LINS={}
        return ce
    L17MAP=PR.fit_layer(17)
    out={'costs':{str(k):v for k,v in costs.items()},'base':base}
    res={}
    for tag,l17 in (('real',None),('linearized',L17MAP)):
        b_=ce_hooked([],l17)
        d16=ce_hooked([16],l17)-b_
        d17=ce_hooked([17],l17)-b_
        joint=ce_hooked([16,17],l17)-b_
        res[tag]={'base':b_,'d16':d16,'d17':d17,'joint':joint,
                  'excess':joint-d16-d17}
        print(f'{tag:10s}: base {b_:.4f} | d16 {d16:+.4f} | d17 {d17:+.4f} | '
              f'excess {joint-d16-d17:+.4f}',flush=True)
    others=[v for k,v in costs.items() if k!=2]
    med=sorted(others)[len(others)//2]
    pa=all(v<=0.1 for k,v in costs.items() if k!=2) and costs[2]>=3*med
    drop=1-res['linearized']['excess']/res['real']['excess'] \
         if res['real']['excess']>1e-6 else float('nan')
    pb=drop>=0.7
    pc=abs(res['linearized']['base']-res['real']['base'])<=0.05
    out['interchange']=res; out['excess_drop']=drop
    out['pred_a2']=bool(pa); out['pred_b2']=bool(pb); out['pred_c2']=bool(pc)
    print(f"\n(a2) front-loading survives: {'HELD' if pa else 'FAILED'} "
          f"(L2 +{costs[2]:.3f} vs median +{med:.3f})")
    print(f"(b2) interaction-kill survives (>=70%): {'HELD' if pb else 'FAILED'} "
          f"({drop if drop==drop else 0:.0%})")
    print(f"(c2) consistent stand-in tighter (<=0.05): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
