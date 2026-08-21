"""EARLY BAND AXIS -- identify the early calibration band's direction
(663: L4,5,6 share a frequency-calibration direction that is NOT the
unembedding readout axis and is orthogonal to the late band). Hypothesis:
the early band corrects the CURRENT token's frequency in the
representation -- so its direction should align with the INPUT/embedding
frequency direction (the direction in residual space that encodes the
current token's log-frequency), not the output readout axis.

w_early = mean of the L4/5/6 w_freq_L directions (their shared axis).
Compare to:
  - emb_freq_dir: the direction encoding the CURRENT token's log-freq in
    the early residual = cov(residual-after-block-3, current-token
    log-freq). (What the stream carries about how frequent the current
    token is.)
  - readout freq_dir (656): expect LOW (663 showed ~0.06).

REGISTERED PREDICTIONS:
  (0) SANITY: emb_freq_dir reads out current-token log-freq from the
    early residual (probe AUC / corr clearly above chance);
  (a) EARLY BAND = INPUT-FREQUENCY DIRECTION: |cos(w_early, emb_freq_dir)|
    is substantially higher than |cos(w_early, readout freq_dir)| and
    higher than random -- the early band operates on the current token's
    frequency representation, not the output;
  (b) report all three cosines;
  NULL: a random direction has low |cos| with emb_freq_dir (~0.03)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'early_band_axis_results.json'
NFRESH = 48
EARLY = [4, 5, 6]
CAP_AFTER = 3                     # capture residual after block 3 (entering the early band)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    WU = m.lm_head.weight.detach().float().cpu().numpy()
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    cur = fresh[:, :256].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V).astype(np.float64)
    lfc_vocab = np.log(freq + 1.0) - np.log(freq + 1.0).mean()

    # readout freq_dir (656)
    G = WU.T @ WU; b = WU.T @ lfc_vocab
    readout_dir = np.linalg.solve(G + 1e-3 * np.eye(D), b)
    readout_dir = readout_dir / (np.linalg.norm(readout_dir) + 1e-9)

    # capture mlp outputs of L4/5/6 (for w_freq_L) AND residual after block 3
    caps = {li: [] for li in EARLY}
    resid3 = []
    hks = [m.transformer.h[li].mlp.register_forward_hook(
        (lambda li: lambda mo, i_, o_: caps[li].append(
            o_.detach().float().reshape(-1, D).cpu()))(li)) for li in EARLY]

    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
            if li == CAP_AFTER:
                resid3.append(x.detach().float().reshape(-1, D).cpu())
    for h in hks:
        h.remove()

    tgt_lf = np.log(freq[nxt] + 1.0); tc = tgt_lf - tgt_lf.mean()
    ws = []
    for li in EARLY:
        O = torch.cat(caps[li], 0).numpy(); Oc = O - O.mean(0)
        w = Oc.T @ tc; w = w / (np.linalg.norm(w) + 1e-9); ws.append(w)
    w_early = np.mean(ws, 0); w_early = w_early / (np.linalg.norm(w_early) + 1e-9)

    # emb_freq_dir: current-token log-freq direction in the residual after block 3
    X3 = torch.cat(resid3, 0).numpy(); X3c = X3 - X3.mean(0)
    cur_lf = np.log(freq[cur] + 1.0); clc = cur_lf - cur_lf.mean()
    emb_dir = X3c.T @ clc; emb_dir = emb_dir / (np.linalg.norm(emb_dir) + 1e-9)
    # sanity: does emb_dir read out current-token log-freq?
    r0 = float(np.corrcoef(X3 @ emb_dir, cur_lf)[0, 1])

    cos_emb = float(w_early @ emb_dir)
    cos_readout = float(w_early @ readout_dir)
    g = np.random.default_rng(0)
    rc = [abs(float((lambda r: r/np.linalg.norm(r))(g.standard_normal(D)) @ emb_dir))
          for _ in range(200)]
    rand_mean = float(np.mean(rc))

    print(f'(0) emb_dir reads current-token log-freq, corr {r0:.3f}', flush=True)
    print(f'cos(w_early, emb_freq_dir)   {cos_emb:+.3f}', flush=True)
    print(f'cos(w_early, readout_freq)   {cos_readout:+.3f}', flush=True)
    print(f'random |cos(emb_dir)| mean   {rand_mean:.3f}', flush=True)

    p0 = abs(r0) >= 0.5
    pa = abs(cos_emb) > abs(cos_readout) and abs(cos_emb) > 5 * rand_mean
    null_ok = rand_mean < 0.1
    print(f'\n(0) emb_dir sane: {p0}', flush=True)
    print(f'(a) early band = input-frequency dir (|cos_emb|>|cos_readout| & >>random): '
          f'{pa}', flush=True)
    print(f'NULL random low: {null_ok}', flush=True)

    out = {'emb_dir_readout_corr': round(r0, 4), 'cos_wearly_embfreq': round(cos_emb, 4),
           'cos_wearly_readout': round(cos_readout, 4),
           'random_abscos_mean': round(rand_mean, 4),
           'pred_0': bool(p0), 'pred_a_input_freq': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
