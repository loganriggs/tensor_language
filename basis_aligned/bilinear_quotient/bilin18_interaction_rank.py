"""Why didn't scheduled refitting close the 16-17 interaction? Rank sweep diagnostic.

§33's refit closed only 21% of the +0.0649 interaction, so 'fit on the wrong upstream
distribution' is a minor mechanism. The alternative: BOTH replacements truncate the
same bus signal (the 16->17 syntax axis rides both layers), and losses on a shared wire
compound through L17's quadratic readout. If that is right, the interaction excess
should shrink rapidly as the form rank k grows (less truncation loss to compound),
faster than the individual damages shrink. REGISTERED PREDICTION: at k=8 the excess is
under a third of its k=2 value, while individual damages fall more slowly.
"""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, eval_ce
from bilin18_identifiable import form_for_direction, mlp_inputs
from bilin18_whitened import sqrtm_psd, truncate
from bilin18_layer17 import Truncated, out_pcs

DEV='cuda'
model, cfg = load_elriggs('bilin18', device=DEV)
tokens = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/bilin18_eval_tokens.pt')
base = eval_ce(model, tokens, batch=4)

CACHE={}
def build(li, R, k):
    key=(li,)
    if key not in CACHE:
        mlp = model.transformer.h[li].mlp
        V, mu, ev = out_pcs(model, tokens, li, 512)
        X = mlp_inputs(model, tokens, (li,), 6000)[li].to(DEV)
        S = X.T @ X / X.shape[0]
        Sh, Sih = sqrtm_psd(S)
        CACHE[key]=(V,mu,Sh,Sih,mlp)
    V,mu,Sh,Sih,mlp=CACHE[key]
    P=V[:R]; bias=mlp.Down_bias.detach().float()
    forms=torch.stack([form_for_direction(mlp,P[p]) for p in range(R)])
    Fw=torch.stack([Sih@truncate(Sh@forms[p]@Sh,k)@Sih for p in range(R)])
    return Truncated(P.float(),Fw.float(),(mu-bias).float(),bias.float()).to(DEV)

def main():
    t0=time.time()
    orig={li: model.transformer.h[li].mlp.forward for li in (16,17)}
    def ce_with(pairs):
        for li,f in pairs: model.transformer.h[li].mlp.forward=f.forward
        try: return eval_ce(model,tokens,batch=4)-base
        finally:
            for li,_ in pairs: model.transformer.h[li].mlp.forward=orig[li]
    out={'rows':{}}
    print(f"  {'k':>3} {'d16':>8} {'d17':>8} {'joint':>8} {'excess':>8}")
    for k in (2,4,8,16):
        r16=build(16,4,k); r17=build(17,4,k)
        d16=ce_with([(16,r16)]); d17=ce_with([(17,r17)])
        j=ce_with([(16,r16),(17,r17)])
        exc=j-d16-d17
        out['rows'][k]={'d16':d16,'d17':d17,'joint':j,'excess':exc}
        print(f"  {k:>3} {d16:>+8.4f} {d17:>+8.4f} {j:>+8.4f} {exc:>+8.4f}",flush=True)
    e2=out['rows'][2]['excess']; e8=out['rows'][8]['excess']
    held = e8 < e2/3
    out['prediction_held']=bool(held)
    print(f'\nexcess k=2 {e2:+.4f} -> k=8 {e8:+.4f}; registered prediction '
          f'(k=8 excess < 1/3 of k=2): {"HELD" if held else "FAILED"}')
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_interaction_rank_results.json','w'),indent=1)
    print(f'wrote bilin18_interaction_rank_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
