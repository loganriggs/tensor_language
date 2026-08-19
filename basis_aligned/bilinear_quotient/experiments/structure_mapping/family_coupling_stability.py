"""Coupling-constant stability check for section 192's normalized comparison
(single-cell c estimates are fragile; the product law's c is per-damage-family,
section 123). Measure c = excess/(d1*d2) at TWO span sizes (k=4 and k=8) for
bilin18 L16/L17, swiglu18 L16/L17, bilin12 L10/L11. REGISTERED: (a) within each
model, c stable within 2x across sizes; (b) bilin12's c >= 2x swiglu18's at
both sizes (the section-192 ordering is size-robust); (c) all c positive.

Prior context -- second instrument for section 191's depth-feature claim: the CAUSAL
signature. In bilin18, the interchange shows as composition excess -- jointly
ablating the L16 and L17 top-8 MLP spans costs more than the sum (+0.143 raw).
If the interchange is a depth feature, swiglu18 (18L) should show a real excess
at its L16/L17 and bilin12 (12L) should not at its L10/L11.

REGISTERED PREDICTIONS: (a) swiglu18 excess(L16,L17) >= 0.05; (b) bilin12
excess(L10,L11) <= 0.02; (c) control: swiglu18's unlinked pair (L8,L13) excess
<= 0.02 (the excess is edge-specific, not generic to joint ablation)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV, orth
from tier2_model import load_elriggs
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'family_coupling_results.json')

@torch.no_grad()
def excess(m2, D, li1, li2, k=8):
    # spans from stats rows
    spans={}
    for li in (li1,li2):
        caps=[]
        h=m2.transformer.h[li].mlp.register_forward_hook(
            lambda mod,i_,o_: caps.append(o_.detach().reshape(-1,D).float()))
        for i in range(0,24,6):
            b=FW[i:i+6,:513].to(DEV)
            m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        Y=torch.cat(caps); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:k].T)
        spans[li]=(Q,Ybar@Q)
    def ce(dmg):
        hs=[]
        for li in dmg:
            Q,cbar=spans[li]
            def mk(Q=Q,cbar=cbar):
                def hook(mod,i_,o_):
                    c=o_.float()@Q
                    return (o_-((c-cbar)@Q.T).to(o_.dtype))
                return hook
            hs.append(m2.transformer.h[li].mlp.register_forward_hook(mk()))
        tot,n=0.0,0
        for i in range(300,364,4):
            b=FW[i:i+4,:257].to(DEV)
            loss=m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
            ntok=(b.shape[1]-1)*b.shape[0]
            tot+=float(loss)*ntok; n+=ntok
        for h in hs: h.remove()
        return tot/n
    base=ce([])
    d1=ce([li1])-base; d2=ce([li2])-base; j=ce([li1,li2])-base
    return d1,d2,j,j-d1-d2

@torch.no_grad()
def main():
    t0=time.time()
    out={}
    import bilin18_joint_removal as JR
    cs={}
    specs=[('bilin18',None,16,17),('swiglu18','swiglu18',16,17),
           ('bilin12','bilin12',10,11)]
    for name,key,l1,l2 in specs:
        if key is None:
            m2=JR.m; D=1152
        else:
            m2,_=load_elriggs(key, device=DEV)
            D=m2.transformer.wte.weight.shape[1]
        row={}
        for k in (4,8):
            d1,d2,j,e=excess(m2,D,l1,l2,k=k)
            c=e/max(d1*d2,1e-6)
            row[k]={'d1':d1,'d2':d2,'excess':e,'c':c}
            print(f'{name:9s} k={k}: d {d1:+.3f}/{d2:+.3f} excess {e:+.4f} '
                  f'c={c:.2f}',flush=True)
        cs[name]=row
        out[name]=row
        if key is not None:
            del m2; torch.cuda.empty_cache()
    pa=all(max(cs[n][4]['c'],1e-3)/max(cs[n][8]['c'],1e-3)<=2 and
           max(cs[n][8]['c'],1e-3)/max(cs[n][4]['c'],1e-3)<=2 for n in cs)
    pb=all(cs['bilin12'][k]['c']>=2*cs['swiglu18'][k]['c'] for k in (4,8))
    pc=all(cs[n][k]['c']>0 for n in cs for k in (4,8))
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"\n(a) c stable within 2x: {'HELD' if pa else 'FAILED'}")
    print(f"(b) bilin12 c >= 2x swiglu18 both sizes: {'HELD' if pb else 'FAILED'}")
    print(f"(c) all c positive: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
