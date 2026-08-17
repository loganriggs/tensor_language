"""The mechanism-prescribed fix for §32: refit L17's replacement DOWNSTREAM of L16's.

§32 localised the composition failure to the 16->17 bus edge: L17's replacement was fit
on inputs produced by an intact layer 16. The fix scheduled fitting: install L16's
replacement first, then fit L17's (output PCs, input second moment, whitened forms) on
the MODIFIED model. REGISTERED PREDICTION: the refit recovers most of the +0.0649
interaction -- joint(16, 17-refit) <= 0.155 vs the naive joint's 0.1974 (i.e. at least
two thirds of the excess closed).
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

def build_repl(li, R=4, k=2):
    mlp = model.transformer.h[li].mlp
    V, mu, ev = out_pcs(model, tokens, li, 512)
    P = V[:R]
    X = mlp_inputs(model, tokens, (li,), 6000)[li].to(DEV)
    S = X.T @ X / X.shape[0]
    Sh, Sih = sqrtm_psd(S)
    bias = mlp.Down_bias.detach().float()
    forms = torch.stack([form_for_direction(mlp, P[p]) for p in range(R)])
    Fw = torch.stack([Sih @ truncate(Sh @ forms[p] @ Sh, k) @ Sih for p in range(R)])
    return Truncated(P.float(), Fw.float(), (mu - bias).float(), bias.float()).to(DEV)

def main():
    t0=time.time()
    orig = {li: model.transformer.h[li].mlp.forward for li in (16,17)}
    r16 = build_repl(16)
    # naive L17 replacement (fit on intact model), for the reference arm
    r17_naive = build_repl(17)
    # refit arm: fit L17's replacement WITH L16 already replaced
    model.transformer.h[16].mlp.forward = r16.forward
    r17_refit = build_repl(17)
    model.transformer.h[16].mlp.forward = orig[16]

    def ce_with(pairs):
        for li,f in pairs: model.transformer.h[li].mlp.forward = f.forward
        try: return eval_ce(model, tokens, batch=4) - base
        finally:
            for li,_ in pairs: model.transformer.h[li].mlp.forward = orig[li]

    naive = ce_with([(16,r16),(17,r17_naive)])
    refit = ce_with([(16,r16),(17,r17_refit)])
    solo16 = ce_with([(16,r16)])
    refit_alone = ce_with([(17,r17_refit)])
    out={'base_ce':base,'joint_naive':naive,'joint_refit':refit,
         'solo16':solo16,'refit17_on_intact':refit_alone}
    print(f'joint, naive fit:      +{naive:.4f}   (§32: +0.1974)')
    print(f'joint, scheduled fit:  +{refit:.4f}')
    exc_naive = naive - solo16 - 0.1018
    exc_refit = refit - solo16 - 0.1018
    closed = 1 - (refit - (solo16 + 0.1018)) / max(naive - (solo16 + 0.1018),1e-9)
    out['excess_closed_frac']=closed
    print(f'interaction excess: naive {exc_naive:+.4f} -> refit {exc_refit:+.4f} '
          f'({100*closed:.0f}% closed)')
    print(f'(sanity: refit-17 installed alone on the intact model: '
          f'+{refit_alone:.4f} -- expected worse than the naive 0.1018, since it '
          f'was fit for a different upstream)')
    held = refit <= 0.155
    out['prediction_held']=bool(held)
    print(f'registered prediction (joint refit <= 0.155): '
          f'{"HELD" if held else "FAILED"}')
    out['runtime_s']=time.time()-t0
    json.dump(out, open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                        'bilin18_sequential_refit_results.json','w'), indent=1)
    print(f'wrote bilin18_sequential_refit_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
