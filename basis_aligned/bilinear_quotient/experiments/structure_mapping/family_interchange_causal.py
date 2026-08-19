"""Second instrument for section 191's depth-feature claim: the CAUSAL
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
     'family_interchange_causal_results.json')

@torch.no_grad()
def excess(m2, D, li1, li2):
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
        Q=orth(Vh[:8].T)
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
    msw,_=load_elriggs('swiglu18', device=DEV)
    Dsw=msw.transformer.wte.weight.shape[1]
    d1,d2,j,e=excess(msw,Dsw,16,17)
    out['swiglu18_16_17']={'d1':d1,'d2':d2,'joint':j,'excess':e}
    print(f'swiglu18 L16/L17: d16 +{d1:.3f} d17 +{d2:.3f} joint +{j:.3f} '
          f'excess {e:+.4f}',flush=True)
    d1,d2,j,ec=excess(msw,Dsw,8,13)
    out['swiglu18_8_13']={'excess':ec}
    print(f'swiglu18 L8/L13 (control): excess {ec:+.4f}',flush=True)
    del msw; torch.cuda.empty_cache()
    m12,_=load_elriggs('bilin12', device=DEV)
    D12=m12.transformer.wte.weight.shape[1]
    d1,d2,j,e12=excess(m12,D12,10,11)
    out['bilin12_10_11']={'d1':d1,'d2':d2,'joint':j,'excess':e12}
    print(f'bilin12 L10/L11: d10 +{d1:.3f} d11 +{d2:.3f} joint +{j:.3f} '
          f'excess {e12:+.4f}',flush=True)
    del m12; torch.cuda.empty_cache()
    pa=out['swiglu18_16_17']['excess']>=0.05
    pb=out['bilin12_10_11']['excess']<=0.02
    pc=abs(out['swiglu18_8_13']['excess'])<=0.02
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['ctrl_c']=bool(pc)
    print(f"\n(a) swiglu18 has the causal interchange (>=0.05): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) bilin12 lacks it (<=0.02): {'HELD' if pb else 'FAILED'}")
    print(f"(c) unlinked control (<=0.02): {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
