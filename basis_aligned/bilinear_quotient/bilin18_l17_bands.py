"""L17 band scan. Section 123: deleting L17's top-32 output-PCA span hurts LESS
than the top-8 (raw +0.471 vs +0.489) -- ranks 9-32 contain beneficial-to-delete
directions. Per-band deletion at L17 (ranks 1-8, 9-32, 33-128), raw and
content-level (frozen per-arm gain). REGISTERED: (a) at least one band beyond
rank 8 has NEGATIVE content-level damage; (b) random-24 control within +-0.005.

Prior context -- causal closure of the mezzanine story. Section 110: L1's functionally necessary
nonlinearity concentrates in ranks ~50-500 of its output spectrum, while the
top-48 interface is nearly linear. If the mezzanine carries the function, DELETING
it (mean-ablate, the standard operator) should hurt more than deleting the loud
interface.

Arms: mean-ablate L1-output components on (i) top-48 span, (ii) mezzanine span
(ranks 129-512, 384 dims), (iii) random-384 span, (iv) ranks 49-128 (upper
mezzanine, 80 dims) for the band profile. REGISTERED PREDICTIONS: (a) mezzanine
damage >= 2x top-48 damage; (b) random-384 <= 0.3x mezzanine; (c) caveat check --
if top-48 ablation is catastrophic (>= 1.0 nats) the comparison is dominated by
interface removal (readers starve) rather than computation removal, and the
verdict must be stated conditionally."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_l17_bands_results.json')

@torch.no_grad()
def ce_eval(Q, cbar):
    hs=[]
    if Q is not None:
        def hook(mod,i_,o_):
            c=o_.float()@Q
            return (o_-((c-cbar)@Q.T).to(o_.dtype))
        hs.append(m.transformer.h[17].mlp.register_forward_hook(hook))
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
    t0=time.time()
    accs=[]
    for i in range(0,60,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=17, acc=acc); accs.append(acc[0])
    Y=torch.cat(accs); Ybar=Y.mean(0); Yc=Y-Ybar
    _,_,Vh=torch.linalg.svd(Yc.float(), full_matrices=False)
    base=ce_eval(None,None)
    print(f'base {base:.4f}\n',flush=True)
    arms={'band1_8':orth(Vh[:8].T),
          'band9_32':orth(Vh[8:32].T),
          'band33_128':orth(Vh[32:128].T)}
    g=torch.Generator(device=DEV).manual_seed(0)
    arms['random24']=orth(torch.randn(D,24,device=DEV,generator=g))
    res={}
    for name,Q in arms.items():
        d=ce_eval(Q,Ybar@Q)-base
        res[name]=d
        print(f'{name:18s}: +{d:.4f}',flush=True)
    pa=any(res[k]<0 for k in ('band9_32','band33_128'))
    pb=abs(res['random24'])<=0.005
    out={'base':base,**res,'pred_a':bool(pa),'ctrl_b':bool(pb)}
    print(f"\n(a) some band beyond 8 negative: {'HELD' if pa else 'FAILED'}")
    print(f"(b) random control inert: {'HELD' if pb else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
