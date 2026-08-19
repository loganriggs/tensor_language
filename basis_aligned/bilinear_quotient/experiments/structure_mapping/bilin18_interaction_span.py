"""Third hypothesis for the 16-17 interaction: the R=4 output-span confinement.
Registered prediction: the excess shrinks by more than half from R=4 to R=16 at fixed
k=2 (if not, the interaction is not span coverage either and the elimination continues)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, eval_ce
from bilin18_identifiable import form_for_direction, mlp_inputs
from bilin18_whitened import sqrtm_psd, truncate
from bilin18_layer17 import Truncated, out_pcs
DEV='cuda'
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
    out={'rows':{}}
    print(f"  {'R':>3} {'d16':>8} {'d17':>8} {'joint':>8} {'excess':>8}")
    for R in (4,8,16):
        r16=build(16,R,2); r17=build(17,R,2)
        d16=ce_with([(16,r16)]); d17=ce_with([(17,r17)])
        j=ce_with([(16,r16),(17,r17)]); exc=j-d16-d17
        out['rows'][R]={'d16':d16,'d17':d17,'joint':j,'excess':exc}
        print(f"  {R:>3} {d16:>+8.4f} {d17:>+8.4f} {j:>+8.4f} {exc:>+8.4f}",flush=True)
    e4=out['rows'][4]['excess']; e16=out['rows'][16]['excess']
    held=e16<e4/2
    out['prediction_held']=bool(held)
    print(f'\nexcess R=4 {e4:+.4f} -> R=16 {e16:+.4f}; prediction (halves): '
          f'{"HELD" if held else "FAILED"}')
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_interaction_span_results.json','w'),indent=1)
    print(f'wrote bilin18_interaction_span_results.json ({out["runtime_s"]:.0f}s)')
if __name__=='__main__': main()
