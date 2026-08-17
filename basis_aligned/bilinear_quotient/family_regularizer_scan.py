"""Verification for section 206's cross-model assertion (which outran its
evidence): do the no-slack siblings really lack deletion-improves spans?
bilin18's regularizers appeared as TOP-8 SPAN deletions improving CE (L9
-0.021, L15 -0.011 replicated) -- that scan never ran on the siblings.
Top-8 output-PCA span deletion costs for bilin12 (layers 3-10) and swiglu18
(layers 5-15).

REGISTERED PREDICTIONS (the section-206 slack-regularizer identity): (a)
bilin12 has <= 1 span with deletion benefit <= -0.01; (b) swiglu18 has <= 1
(its lone slack layer L15 the only candidate); a violation (>= 2 negatives in
a no-slack model) breaks the identity and corrects section 206."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV, orth
from tier2_model import load_elriggs
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'family_regularizer_results.json')

@torch.no_grad()
def scan(name, layers):
    m2,_=load_elriggs(name, device=DEV)
    D=m2.transformer.wte.weight.shape[1]
    def ce(patch):
        hs=[]
        if patch is not None:
            li,Q,cbar=patch
            def hook(mod,i_,o_):
                c=o_.float()@Q
                return (o_-((c-cbar)@Q.T).to(o_.dtype))
            hs.append(m2.transformer.h[li].mlp.register_forward_hook(hook))
        tot,n=0.0,0
        for i in range(384,448,4):
            b=FW[i:i+4,:257].to(DEV)
            loss=m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
            ntok=(b.shape[1]-1)*b.shape[0]
            tot+=float(loss)*ntok; n+=ntok
        for h in hs: h.remove()
        return tot/n
    caps={li:[] for li in layers}
    hs=[]
    for li in layers:
        def mk(li=li):
            return lambda mod,i_,o_: caps[li].append(
                o_.detach().reshape(-1,D).float())
        hs.append(m2.transformer.h[li].mlp.register_forward_hook(mk()))
    for i in range(0,24,6):
        b=FW[i:i+6,:513].to(DEV)
        m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    base=ce(None)
    neg=[]; costs={}
    for li in layers:
        Y=torch.cat(caps[li]); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        c=ce((li,Q,Ybar@Q))-base
        costs[li]=c
        if c<=-0.01: neg.append(li)
        print(f'{name} L{li:2d}: span-8 deletion {c:+.4f}'
              f'{"  <-- IMPROVES" if c<=-0.01 else ""}',flush=True)
    del m2; torch.cuda.empty_cache()
    return neg,costs

@torch.no_grad()
def main():
    t0=time.time()
    n12,c12=scan('bilin12', list(range(3,11)))
    nsw,csw=scan('swiglu18', list(range(5,16)))
    pa=len(n12)<=1; pb=len(nsw)<=1
    out={'bilin12_neg':n12,'swiglu18_neg':nsw,
         'bilin12':{str(k):v for k,v in c12.items()},
         'swiglu18':{str(k):v for k,v in csw.items()},
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"\nbilin12 improves: {n12} | swiglu18 improves: {nsw}")
    print(f"(a) bilin12 <=1: {'HELD' if pa else 'FAILED'}")
    print(f"(b) swiglu18 <=1: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
