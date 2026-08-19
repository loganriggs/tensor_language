"""Instrument disagreement to resolve: family_regularizer_scan (section 207)
measured bilin12 L7's top-8 span deletion at +0.0305 (no regularizer); both
scorecard validations measured it <= -0.01, replicating on disjoint stats
fits. If L7 is a genuine deletion-improves span, bilin12's regularizer set is
NOT empty while its licensed-constant set is -- violating the slack-
regularizer identity (section 207, 'exact in all three models') and earning a
ledger entry. Four independent span fits (rows 0-30, 30-60, 60-90, 90-120)
plus an exact rerun of the family-scan construction (rows 0-24, its
orthonormalization path).

REGISTERED PREDICTIONS: (a) if >= 3/4 independent fits give deletion
<= -0.01, the identity FALLS in bilin12 (ledger); (b) if <= 1/4, the
scorecard fits were the outliers and P2 machinery needs review; (c) the
family-scan exact rerun reproduces its published +0.03 within 0.02 (else the
original run itself is in question)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from bilin18_joint_removal import FW, orth, DEV
from tier2_model import load_elriggs
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin12_l7_span_replication_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    m2,_=load_elriggs('bilin12', device=DEV)
    D=m2.transformer.wte.weight.shape[1]
    def caps(r0,r1,step=6,ctx=513):
        outs=[]
        h=m2.transformer.h[7].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(r0,r1,step):
            b=FW[i:i+step,:ctx].to(DEV)
            m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    def ce(Q=None,cbar=None):
        hs=[]
        if Q is not None:
            def hook(mod,i_,o_):
                c=o_.float().reshape(-1,D)@Q
                return o_-((c-cbar)@Q.T).to(o_.dtype).view_as(o_)
            hs.append(m2.transformer.h[7].mlp.register_forward_hook(hook))
        tot,n=0.0,0
        for i in range(384,448,4):
            b=FW[i:i+4,:257].to(DEV)
            loss=m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
            ntok=(b.shape[1]-1)*b.shape[0]
            tot+=float(loss)*ntok; n+=ntok
        for h in hs: h.remove()
        return tot/n
    base=ce()
    deltas=[]
    for r0,r1 in ((0,30),(30,60),(60,90),(90,120)):
        Y=caps(r0,r1); Yb=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        d=ce(Q,Yb.float()@Q)-base
        deltas.append(d)
        print(f'fit rows {r0}-{r1}: span-8 deletion {d:+.4f}',flush=True)
    Y=caps(0,24); Yb=Y.mean(0)
    _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
    Q=orth(Vh[:8].T)
    d_family=ce(Q,Yb.float()@Q)-base
    print(f'family-scan construction (rows 0-24): {d_family:+.4f} '
          f'(published +0.0305)',flush=True)
    n_imp=sum(1 for d in deltas if d<=-0.01)
    pa=n_imp>=3; pb=n_imp<=1
    pc=abs(d_family-0.0305)<=0.02
    out={'deltas':deltas,'family_rerun':d_family,'n_improve':n_imp,
         'identity_falls':bool(pa),'scorecard_outlier':bool(pb),
         'family_reproduces':bool(pc)}
    print(f"\n{n_imp}/4 fits improve")
    print(f"(a) identity falls (>=3/4): {'YES -- ledger' if pa else 'no'}")
    print(f"(b) scorecard outlier (<=1/4): {'YES' if pb else 'no'}")
    print(f"(c) family rerun reproduces: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
