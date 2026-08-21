"""LAYERWISE CALIB AXIS -- do the 5 calibrator layers (662: L4,5,6,16,17)
write along the SAME frequency axis, or different directions? If they all
align with the unembedding's log-frequency readout direction (freq_dir,
656), the distributed calibration is ONE shared frequency direction
applied at 5 layers -- a single reused mechanism.

For each layer L, w_freq_L = cov(mlp-L output, target log-freq). Compare
each to freq_dir = W_U^+ log_freq (656, computed from the unembedding,
INDEPENDENT of per-layer activations -- so cross-layer comparison via
this shared reference avoids the 659 activation-covariance confound).

REGISTERED PREDICTIONS:
  (0) SANITY: freq_dir readout-vs-logfreq corr reproduces 656 (~0.5);
      random directions give |cos(freq_dir)| ~ 0.03;
  (a) SHARED AXIS: the 5 calibrator layers (L4,5,6,16,17) all have
      w_freq_L aligned with freq_dir (|cos| >= 0.3, well above random) --
      the distributed calibration writes along one shared unembedding
      frequency axis;
  (b) report cos(w_freq_L, freq_dir) for ALL 18 layers (calibrators
      should stand out) and pairwise cos among the 5 calibrators;
  NULL: writer layers (e.g. L0-3) and random directions have low
      |cos(freq_dir)| -- alignment is specific to the calibrator layers."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'layerwise_calib_axis_results.json'
NFRESH = 48
CALIB = [4, 5, 6, 16, 17]


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    WU = m.lm_head.weight.detach().float().cpu().numpy()
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V).astype(np.float64)
    lfv = np.log(freq + 1.0); lfc = lfv - lfv.mean()

    # freq_dir via pinv (656)
    G = WU.T @ WU; b = WU.T @ lfc
    freq_dir = np.linalg.solve(G + 1e-3 * np.eye(D), b)
    freq_dir = freq_dir / (np.linalg.norm(freq_dir) + 1e-9)
    r0 = float(np.corrcoef(WU @ freq_dir, lfv)[0, 1])

    NL = len(m.transformer.h)
    caps = {li: [] for li in range(NL)}
    hks = [m.transformer.h[li].mlp.register_forward_hook(
        (lambda li: lambda mo, i_, o_: caps[li].append(
            o_.detach().float().reshape(-1, D).cpu()))(li)) for li in range(NL)]
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    for h in hks:
        h.remove()

    tgt_lf = np.log(freq[nxt] + 1.0); tc = tgt_lf - tgt_lf.mean()
    wdirs = {}
    cosf = {}
    for li in range(NL):
        O = torch.cat(caps[li], 0).numpy(); Oc = O - O.mean(0)
        w = Oc.T @ tc; w = w / (np.linalg.norm(w) + 1e-9)
        wdirs[li] = w
        cosf[li] = round(float(w @ freq_dir), 4)
        tag = 'CAL' if li in CALIB else ''
        print(f'L{li:2d}: cos(w_freq_L, freq_dir) {cosf[li]:+.3f} {tag}', flush=True)

    # pairwise cos among calibrators
    pair = {}
    for a in range(len(CALIB)):
        for b_ in range(a + 1, len(CALIB)):
            i, j = CALIB[a], CALIB[b_]
            pair[f'{i}-{j}'] = round(float(wdirs[i] @ wdirs[j]), 3)

    g = np.random.default_rng(0)
    rc = [abs(float((lambda r: r/np.linalg.norm(r))(g.standard_normal(D)) @ freq_dir))
          for _ in range(200)]
    rand_mean = float(np.mean(rc))

    calib_cos = [abs(cosf[li]) for li in CALIB]
    writer_cos = [abs(cosf[li]) for li in [0, 1, 2, 3]]
    p0 = abs(r0 - 0.53) < 0.15 and rand_mean < 0.1
    pa = all(c >= 0.3 for c in calib_cos)
    null_ok = np.mean(writer_cos) < np.mean(calib_cos)
    print(f'\n(0) freq_dir sane (corr {r0:.2f}, rand {rand_mean:.3f}): {p0}', flush=True)
    print(f'(a) all 5 calibrators |cos(freq_dir)|>=0.3: {pa} '
          f'({[round(c,2) for c in calib_cos]})', flush=True)
    print(f'(b) pairwise cos among calibrators: {pair}', flush=True)
    print(f'NULL writers < calibrators (mean |cos| {np.mean(writer_cos):.2f} vs '
          f'{np.mean(calib_cos):.2f}): {null_ok}', flush=True)

    out = {'freqdir_corr': round(r0, 4), 'random_abscos_mean': round(rand_mean, 4),
           'cos_with_freqdir': cosf, 'pairwise_calib_cos': pair,
           'pred_0': bool(p0), 'pred_a_shared_axis': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
