"""Out-of-sample test of the section-35 product law at an untried corner.

The law, fit on six configurations (R=4 with k swept; k=2 with R swept): the 16-17
composition excess = c * d16 * d17, c = 22.9 +/- 2.4. The corner (R=8, k=8) was never
measured and varies BOTH knobs off the fitted axes. REGISTERED PREDICTION: measure d16
and d17 solo at (8,8), predict excess = 22.9*d16*d17, then measure the joint; bar =
prediction within 25% of measurement. Also (R=16, k=8) as a second corner."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, eval_ce
from bilin18_identifiable import form_for_direction, mlp_inputs
from bilin18_whitened import sqrtm_psd, truncate
from bilin18_layer17 import Truncated, out_pcs
DEV='cuda'; C_LAW=22.9
model,cfg=load_elriggs('bilin18',device=DEV)
tokens=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/bilin18_eval_tokens.pt')
base=eval_ce(model,tokens,batch=4)
CACHE={}
def build(li,R,k):
    if li not in CACHE:
        mlp=model.transformer.h[li].mlp
        V,mu,ev=out_pcs(model,tokens,li,512)
        X=mlp_inputs(model,tokens,(li,),6000)[li].to(DEV)
        S=X.T@X/X.shape[0]; Sh,Sih=sqrtm_psd(S)
        CACHE[li]=(V,mu,Sh,Sih,mlp)
    V,mu,Sh,Sih,mlp=CACHE[li]
    P=V[:R]; bias=mlp.Down_bias.detach().float()
    forms=torch.stack([form_for_direction(mlp,P[p]) for p in range(R)])
    Fw=torch.stack([Sih@truncate(Sh@forms[p]@Sh,k)@Sih for p in range(R)])
    return Truncated(P.float(),Fw.float(),(mu-bias).float(),bias.float()).to(DEV)
def main():
    t0=time.time()
    orig={li:model.transformer.h[li].mlp.forward for li in (16,17)}
    def ce_with(pairs):
        for li,f in pairs: model.transformer.h[li].mlp.forward=f.forward
        try: return eval_ce(model,tokens,batch=4)-base
        finally:
            for li,_ in pairs: model.transformer.h[li].mlp.forward=orig[li]
    out={'c_law':C_LAW,'corners':{}}
    for R,k in ((8,8),(16,8)):
        r16=build(16,R,k); r17=build(17,R,k)
        d16=ce_with([(16,r16)]); d17=ce_with([(17,r17)])
        pred=C_LAW*d16*d17
        j=ce_with([(16,r16),(17,r17)]); exc=j-d16-d17
        err=abs(pred-exc)/max(abs(exc),1e-9)
        out['corners'][f'{R},{k}']={'d16':d16,'d17':d17,'excess_pred':pred,
                                    'excess_meas':exc,'rel_err':err}
        print(f'(R={R},k={k}): d16 {d16:+.4f} d17 {d17:+.4f} | predicted excess '
              f'{pred:+.4f} | measured {exc:+.4f} | error {100*err:.0f}%',flush=True)
    ok=all(v['rel_err']<0.25 for v in out['corners'].values())
    out['prediction_held']=bool(ok)
    print(f'\nregistered bar (both corners within 25%): {"HELD" if ok else "FAILED"}')
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_product_law_test_results.json','w'),indent=1)
    print(f'wrote bilin18_product_law_test_results.json ({out["runtime_s"]:.0f}s)')
if __name__=='__main__': main()
