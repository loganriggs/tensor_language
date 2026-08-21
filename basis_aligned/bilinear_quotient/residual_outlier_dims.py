"""RESIDUAL OUTLIER DIMS -- fresh structural ground. Do this model's
residual streams have MASSIVE-ACTIVATION / outlier dimensions (a few
dims with hugely-larger-than-typical magnitude, a documented LLM
phenomenon tied to attention sinks and quantization difficulty)? If so,
which dims, at what depth, and do they carry anything interpretable
(e.g. the frequency-calibration axis, or a constant/sink)?

Method: capture the residual after each block over real text; for each
dimension compute its RMS magnitude across positions; flag dims whose
magnitude is >> the median (outlier ratio). Report the count and the top
outlier dims per depth, and whether the top outliers overlap the
frequency-calibration direction w_freq.

REGISTERED PREDICTIONS:
  (0) SANITY: the residual has a nonuniform magnitude spectrum (max-dim
      RMS >> median-dim RMS);
  (a) OUTLIER DIMS EXIST: at least one depth has dims with RMS >= 8x the
      median dim (massive activations), and report how many and where;
  (b) STABILITY: the top outlier dims are consistent across depths (a few
      persistent channels), i.e. the same dims dominate at multiple
      layers;
  (c) report the outlier count and top dims by depth; check |cos| of the
      top-outlier one-hot directions with w_freq;
  NULL: a random Gaussian matrix of the same shape has no dims >= 8x
      median (outliers are real model structure, not sampling)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'residual_outlier_dims_results.json'
NFRESH = 48
DEPTHS = [0, 4, 8, 12, 16, 17]


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]

    # accumulate sum of squares per dim per depth
    ss = {li: torch.zeros(D, dtype=torch.float64) for li in DEPTHS}
    n = 0
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
            if li in DEPTHS:
                ss[li] += (x.detach().float() ** 2).reshape(-1, D).sum(0).double().cpu()
        n += idx.numel()

    rms = {li: torch.sqrt(ss[li] / n).numpy() for li in DEPTHS}

    # w_freq (mlp17) for overlap check
    cap = []
    hk = m.transformer.h[17].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    hk.remove()
    O = torch.cat(cap, 0).numpy()
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V).astype(np.float64)
    tgt_lf = np.log(freq[nxt] + 1.0); Oc = O - O.mean(0); tc = tgt_lf - tgt_lf.mean()
    w = Oc.T @ tc; w = w / (np.linalg.norm(w) + 1e-9)

    out = {'by_depth': {}}
    top_sets = []
    for li in DEPTHS:
        r = rms[li]; med = float(np.median(r)); mx = float(r.max())
        ratio = mx / med
        n_out = int((r >= 8 * med).sum())
        top = np.argsort(-r)[:5]
        out['by_depth'][li] = {'median_rms': round(med, 3), 'max_rms': round(mx, 3),
                               'max_over_median': round(ratio, 2), 'n_outlier_8x': n_out,
                               'top5_dims': [int(d) for d in top],
                               'top5_rms': [round(float(r[d]), 2) for d in top]}
        top_sets.append(set(top.tolist()))
        print(f'depth {li:2d}: median RMS {med:.3f}  max {mx:.3f}  ({ratio:.1f}x)  '
              f'#>=8x {n_out}  top {list(top)}', flush=True)

    # stability: overlap of top-5 dim sets across depths
    inter = set.intersection(*top_sets) if top_sets else set()
    # w_freq overlap with top outlier dims (one-hot): abs weight of w on those dims
    all_top = set().union(*top_sets)
    wfreq_masson_top = float(sum(w[d] ** 2 for d in all_top))

    # NULL: random gaussian
    g = np.random.default_rng(0); R = g.standard_normal((NFRESH * T, D))
    rr = np.sqrt((R ** 2).mean(0)); rand_ratio = float(rr.max() / np.median(rr))
    rand_nout = int((rr >= 8 * np.median(rr)).sum())

    p0 = out['by_depth'][DEPTHS[-1]]['max_over_median'] > 2
    pa = any(out['by_depth'][li]['n_outlier_8x'] >= 1 for li in DEPTHS)
    pb = len(inter) >= 1
    null_ok = rand_nout == 0
    print(f'\n(0) nonuniform spectrum: {p0}', flush=True)
    print(f'(a) outlier dims (>=8x median) exist: {pa}', flush=True)
    print(f'(b) persistent across depths (top-5 intersection {sorted(inter)}): {pb}',
          flush=True)
    print(f'    w_freq squared-mass on top-outlier dims: {wfreq_masson_top:.3f}', flush=True)
    print(f'NULL random has no 8x outliers: {null_ok} (rand ratio {rand_ratio:.2f}, '
          f'n {rand_nout})', flush=True)

    out.update({'persistent_top_dims': sorted(inter),
                'wfreq_mass_on_top_outliers': round(wfreq_masson_top, 4),
                'random_max_over_median': round(rand_ratio, 3), 'random_n_outlier': rand_nout,
                'pred_0': bool(p0), 'pred_a_outliers': bool(pa), 'pred_b_persistent': bool(pb),
                'null_ok': bool(null_ok), 'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
