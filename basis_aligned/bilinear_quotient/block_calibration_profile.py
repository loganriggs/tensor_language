"""BLOCK CALIBRATION PROFILE -- is frequency calibration localized to
block 17, or do the last few blocks all calibrate?

624-626 established block 17 is a net-beneficial frequency calibrator
(suppresses common tokens, corr +0.64), while blocks 1 and 9 are
writers (corr ~ -0.28). This finishes the picture across ALL 18 blocks:
for each block, corr(log token frequency, per-token removal-delta). A
positive corr = calibrator (suppresses frequent tokens); negative =
writer (builds frequent tokens). Also quantify how much of block 17's
action is PURE frequency by the R^2 of a log-frequency-only linear fit
to its per-token deltas (corr 0.64 -> R^2 ~ 0.41 if purely linear).

REGISTERED PREDICTIONS:
  (0) SANITY: block 17's corr reproduces 625 (+0.64 +/- 0.05);
  (a) LOCALIZED CALIBRATION: block 17 is the strongest positive-corr
      block; report the full 18-block profile and how many blocks have
      positive corr (calibrators) vs negative (writers). Registered
      guess: calibration is localized to the last block(s) -- at most
      2-3 blocks positive, the rest writers;
  (b) PURITY: R^2 of a log-frequency-only fit to block 17's per-token
      delta -- how much of the readout layer's action is explained by
      frequency alone vs context/other structure;
  (c) report the per-block corr profile;
  NULL: early writer blocks (0-4) are all negative-corr (they build
      frequent tokens, they do not calibrate)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'block_calibration_profile_results.json'
NFRESH = 48
NB = 18


@torch.no_grad()
def mean_logit_per_token(fresh, ablate_block):
    V = m.lm_head.weight.shape[0]
    acc = torch.zeros(V, dtype=torch.float64)
    n = 0
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x_in = x
            x, v1 = blk(x, v1, x0)
            if ablate_block is not None and li == ablate_block:
                delta = x - x_in
                x = x_in + delta.mean(dim=(0, 1), keepdim=True)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        lg = lg.reshape(-1, lg.shape[-1])
        acc += lg.sum(0).double().cpu()
        n += lg.shape[0]
    return (acc / n).numpy()


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    V = m.lm_head.weight.shape[0]
    freq = np.bincount(nxt, minlength=V).astype(float)
    keep = freq >= 20
    lf = np.log(freq[keep])

    base = mean_logit_per_token(fresh, None)
    corrs = []
    for L in range(NB):
        delta = mean_logit_per_token(fresh, L) - base
        c = pearson(lf, delta[keep])
        corrs.append(round(c, 4))
        print(f'  block {L:2d}: corr(log-freq, delta) {c:+.4f} '
              f'{"CALIBRATOR" if c > 0.2 else ("writer" if c < -0.2 else "")}',
              flush=True)

    # purity of block 17: R^2 of log-freq-only linear fit to its deltas
    d17 = mean_logit_per_token(fresh, 17) - base
    y = d17[keep]
    x = lf
    b1 = np.cov(x, y, bias=True)[0, 1] / np.var(x)
    b0 = y.mean() - b1 * x.mean()
    yhat = b0 + b1 * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / (ss_tot + 1e-12)

    n_pos = sum(1 for c in corrs if c > 0.2)
    n_neg = sum(1 for c in corrs if c < -0.2)
    argmax = int(np.argmax(corrs))
    p0 = abs(corrs[17] - 0.64) < 0.08
    pa = argmax == 17 and n_pos <= 3
    early_neg = all(corrs[L] < -0.2 for L in range(5))
    print(f'\n(0) block17 corr ~0.64: {p0} ({corrs[17]})', flush=True)
    print(f'(a) block17 strongest positive & <=3 calibrators: {pa} '
          f'(argmax block {argmax}, {n_pos} pos, {n_neg} neg)', flush=True)
    print(f'(b) block17 log-freq-only R^2 {r2:.3f} '
          f'(fraction of its action explained by frequency alone)', flush=True)
    print(f'NULL early blocks 0-4 all writers (neg): {early_neg} '
          f'({[corrs[L] for L in range(5)]})', flush=True)

    out = {'corrs': corrs, 'block17_corr': corrs[17], 'pred_0': bool(p0),
           'argmax_block': argmax, 'n_calibrators': n_pos, 'n_writers': n_neg,
           'pred_a_localized': bool(pa), 'block17_logfreq_R2': round(r2, 4),
           'null_early_writers': bool(early_neg), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
