"""CALIBRATOR DIRECTION ID -- fully characterize the one rank-1 component
we isolated (650-651): WHAT is w_freq geometrically? Hypothesis: block
17's calibration writes a bias proportional to token log-frequency, i.e.
w_freq (in mlp17's output/residual space) is aligned with the
UNEMBEDDING's frequency direction -- the residual direction that, through
W_U, shifts logits by log-frequency.

freq_dir = normalized sum_t (logfreq(t) - mean) * W_U[t]  -- the residual
direction whose logit readout is proportional to token log-frequency.
w_freq = cov(mlp17 output, target log-freq) from 650. Since block 17
SUPPRESSES frequent tokens, w_freq should be ANTI-aligned with freq_dir
(it writes the negative frequency bias), or aligned depending on sign
convention -- what matters is |cos| >> random.

REGISTERED PREDICTIONS:
  (0) SANITY: freq_dir's logit readout correlates with log-frequency
      across the vocabulary (corr >= 0.8) -- it really is the frequency
      axis;
  (a) THE CALIBRATION IS THE FREQUENCY AXIS: |cos(w_freq, freq_dir)| is
      large (>= 0.5) and far above random-direction cosines -- block 17's
      isolated calibration direction is (anti-)aligned with the
      unembedding frequency direction, i.e. it writes a frequency-
      proportional bias;
  (b) report cos(w_freq, freq_dir), its sign, and the random baseline;
  NULL: random directions have |cos| ~ 1/sqrt(D) ~ 0.03 with freq_dir."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'calibrator_direction_id_results.json'
NFRESH = 48


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    WU = m.lm_head.weight.detach().float().cpu().numpy()      # (V,D)

    # corpus log-frequency per vocab token
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V).astype(np.float64)
    lfv = np.log(freq + 1.0)
    lfc = lfv - lfv.mean()

    # freq_dir: residual direction whose logit readout ~ log-frequency
    freq_dir = (lfc[:, None] * WU).sum(0)                      # (D,)
    freq_dir = freq_dir / (np.linalg.norm(freq_dir) + 1e-9)
    # (0) sanity: does WU @ freq_dir correlate with log-freq?
    readout = WU @ freq_dir
    r0 = np.corrcoef(readout, lfv)[0, 1]

    # w_freq: capture mlp17 output, cov with target log-freq (as in 650)
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
    tgt_lf = np.log(freq[nxt] + 1.0)
    Oc = O - O.mean(0); tc = tgt_lf - tgt_lf.mean()
    w_freq = Oc.T @ tc; w_freq = w_freq / (np.linalg.norm(w_freq) + 1e-9)

    cos = float(w_freq @ freq_dir)
    g = np.random.default_rng(0)
    rand_cos = []
    for _ in range(200):
        r = g.standard_normal(D); r /= np.linalg.norm(r)
        rand_cos.append(abs(float(r @ freq_dir)))
    rand_mean = float(np.mean(rand_cos)); rand_max = float(np.max(rand_cos))

    print(f'(0) freq_dir readout vs log-freq corr {r0:.3f}', flush=True)
    print(f'(a) cos(w_freq, freq_dir) = {cos:+.3f} (|{abs(cos):.3f}|); '
          f'random |cos| mean {rand_mean:.3f} max {rand_max:.3f}', flush=True)

    p0 = r0 >= 0.8
    pa = abs(cos) >= 0.5 and abs(cos) > 5 * rand_mean
    null_ok = rand_mean < 0.1
    print(f'\n(0) freq_dir is the frequency axis: {p0}', flush=True)
    print(f'(a) w_freq (anti-)aligned with freq_dir: {pa} '
          f'(sign {"anti" if cos < 0 else "pos"})', flush=True)
    print(f'NULL random |cos| tiny: {null_ok}', flush=True)

    out = {'freqdir_readout_corr': round(float(r0), 4),
           'cos_wfreq_freqdir': round(cos, 4), 'abs_cos': round(abs(cos), 4),
           'sign': 'anti' if cos < 0 else 'pos',
           'random_abscos_mean': round(rand_mean, 4),
           'random_abscos_max': round(rand_max, 4),
           'pred_0': bool(p0), 'pred_a_is_freq_axis': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
