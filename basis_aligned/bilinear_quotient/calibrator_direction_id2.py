"""CALIBRATOR DIRECTION ID v2 -- fix 655's method bug. To get the
residual direction whose UNEMBEDDING readout is proportional to token
log-frequency, solve the least-squares system (W_U rows are correlated),
freq_dir = W_U^+ log_freq = solve(W_U^T W_U, W_U^T log_freq), NOT
W_U^T log_freq. Then test whether the isolated calibration direction
w_freq (650) is (anti-)aligned with it.

REGISTERED PREDICTIONS:
  (0) SANITY (must pass this time): freq_dir = W_U^+ log_freq reads out
      as log-frequency, corr(W_U @ freq_dir, log_freq) >= 0.8;
  (a) THE CALIBRATION IS THE FREQUENCY AXIS: |cos(w_freq, freq_dir)| is
      large (>= 0.4) and >> random -- block 17's isolated rank-1
      calibration writes a log-frequency-proportional bias; sign should
      be ANTI (it suppresses frequent tokens);
  (b) report cos and readout corr;
  NULL: random directions have |cos(., freq_dir)| ~ 0.03."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'calibrator_direction_id2_results.json'
NFRESH = 48


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    WU = m.lm_head.weight.detach().float().cpu().numpy()      # (V,D)
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V).astype(np.float64)
    lfv = np.log(freq + 1.0)
    lfc = lfv - lfv.mean()

    # freq_dir = W_U^+ log_freq via normal equations (D x D solve)
    G = WU.T @ WU                                             # (D,D)
    b = WU.T @ lfc                                            # (D,)
    freq_dir = np.linalg.solve(G + 1e-3 * np.eye(D), b)
    freq_dir = freq_dir / (np.linalg.norm(freq_dir) + 1e-9)
    readout = WU @ freq_dir
    r0 = float(np.corrcoef(readout, lfv)[0, 1])

    # w_freq from mlp17 output (as 650)
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
    tgt_lf = np.log(freq[nxt] + 1.0); Oc = O - O.mean(0); tc = tgt_lf - tgt_lf.mean()
    w_freq = Oc.T @ tc; w_freq = w_freq / (np.linalg.norm(w_freq) + 1e-9)

    cos = float(w_freq @ freq_dir)
    g = np.random.default_rng(0)
    rc = [abs(float((lambda r: r/np.linalg.norm(r))(g.standard_normal(D)) @ freq_dir))
          for _ in range(200)]
    rand_mean = float(np.mean(rc))
    print(f'(0) freq_dir readout vs log-freq corr {r0:.3f}', flush=True)
    print(f'(a) cos(w_freq, freq_dir) {cos:+.3f}; random |cos| mean {rand_mean:.3f}',
          flush=True)

    p0 = r0 >= 0.8
    pa = abs(cos) >= 0.4 and abs(cos) > 5 * rand_mean
    null_ok = rand_mean < 0.1
    print(f'\n(0) freq_dir is the frequency axis (corr>=0.8): {p0}', flush=True)
    print(f'(a) w_freq aligned with freq axis (|cos|>=0.4): {pa} '
          f'(sign {"anti" if cos < 0 else "pos"})', flush=True)
    print(f'NULL random tiny: {null_ok}', flush=True)

    out = {'freqdir_readout_corr': round(r0, 4), 'cos_wfreq_freqdir': round(cos, 4),
           'abs_cos': round(abs(cos), 4), 'sign': 'anti' if cos < 0 else 'pos',
           'random_abscos_mean': round(rand_mean, 4),
           'pred_0': bool(p0), 'pred_a_is_freq_axis': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
