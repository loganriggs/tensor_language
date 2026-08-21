"""MASSIVE DIM CONSTANCY -- confirm 678's DC/bias reading: are the massive
block-17 dims near-CONSTANT across positions (a literal learned bias) or
do they carry per-position signal? For the top massive dims, measure the
mean, std, coefficient of variation (std/|mean|), and sign-consistency
(fraction of positions matching the majority sign) across all positions.
Compare to median-magnitude dims.

REGISTERED PREDICTIONS:
  (0) SANITY: the chosen dims are massive (RMS >> median);
  (a) DC/BIAS: the top massive dims have LOW coefficient of variation
      (std/|mean| < 0.5) and high sign-consistency (>90% one sign) --
      they are near-constant offsets, a learned bias substrate;
  (b) report per-dim mean, std, CV, sign-consistency for top dims vs
      median dims;
  NULL: median-magnitude dims have HIGHER CV (they carry more per-position
      variation relative to their mean) -- constancy is specific to the
      massive dims."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'massive_dim_constancy_results.json'
NFRESH = 48


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    acts = []
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        acts.append(x.detach().float().reshape(-1, D).cpu())
    X = torch.cat(acts, 0).numpy()                      # (Npos, D)
    rms = np.sqrt((X ** 2).mean(0)); med = float(np.median(rms))
    order = np.argsort(-rms)
    top_dims = order[:5]
    med_dims = order[D // 2 - 2:D // 2 + 3]             # around-median-magnitude dims

    def stats(dims):
        res = []
        for d in dims:
            col = X[:, d]
            mean = float(col.mean()); std = float(col.std())
            cv = std / (abs(mean) + 1e-9)
            sign = np.sign(mean) if mean != 0 else 1
            signcons = float((np.sign(col) == sign).mean())
            res.append({'dim': int(d), 'mean': round(mean, 1), 'std': round(std, 1),
                        'cv': round(cv, 3), 'sign_consistency': round(signcons, 3),
                        'rms': round(float(rms[d]), 1)})
        return res

    top = stats(top_dims); meds = stats(med_dims)
    print('TOP massive dims:', flush=True)
    for r in top:
        print(f"  dim {r['dim']:4d} mean {r['mean']:+.0f} std {r['std']:.0f} "
              f"cv {r['cv']:.2f} sign-cons {r['sign_consistency']:.2f}", flush=True)
    print('MEDIAN-magnitude dims:', flush=True)
    for r in meds:
        print(f"  dim {r['dim']:4d} mean {r['mean']:+.1f} std {r['std']:.1f} "
              f"cv {r['cv']:.2f} sign-cons {r['sign_consistency']:.2f}", flush=True)

    top_cv = float(np.mean([r['cv'] for r in top]))
    top_sign = float(np.mean([r['sign_consistency'] for r in top]))
    med_cv = float(np.mean([r['cv'] for r in meds]))
    p0 = top[0]['rms'] > 5 * med
    pa = top_cv < 0.5 and top_sign > 0.9
    null_ok = med_cv > top_cv
    print(f'\n(0) massive: {p0}', flush=True)
    print(f'(a) DC/bias (top CV {top_cv:.2f}<0.5, sign-cons {top_sign:.2f}>0.9): {pa}',
          flush=True)
    print(f'NULL median dims higher CV ({med_cv:.2f} > {top_cv:.2f}): {null_ok}', flush=True)

    out = {'median_rms': med, 'top_dims': top, 'median_dims': meds,
           'top_mean_cv': round(top_cv, 3), 'top_mean_signcons': round(top_sign, 3),
           'median_mean_cv': round(med_cv, 3),
           'pred_0': bool(p0), 'pred_a_dc_bias': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
