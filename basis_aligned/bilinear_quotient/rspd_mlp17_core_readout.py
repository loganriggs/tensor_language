"""RSPD MLP17 CORE READOUT -- name the rank-4 functional core (694). The
A-SVD gives Down's data-conditioned surrogate W_r = A[:,:r]@B[:r,:]; the
columns of A_fac (M=1152 = residual space) are the ORDERED output
directions of mlp17's core -- direction j is where the j-th core component
WRITES into the residual. Project each of the top-4 onto the unembedding
(W_U) to read which tokens it boosts/suppresses, and correlate each with
log-frequency (does a core direction = the calibration axis?). This is the
finer-grained component isolation Q5 was ultimately after: not just 'the
core is rank 4' but WHAT the 4 pieces do.

For each core direction a_j (unit, in residual space):
  - top +/- tokens by (rms_norm-then-)unembedding projection;
  - correlation of its per-token unembedding logit-shift with log-freq
    (calibration signature = strong negative corr: suppress frequent);
  - its cosine with w_freq (the known rank-1 calibration direction, 650).

REGISTERED PREDICTIONS:
  (0) SANITY: the 4 directions are orthonormal (|cos| between them < 0.05,
      they are SVD left-vectors);
  (a) CALIBRATION IN THE CORE: at least ONE of the top-4 core directions is
      a frequency-calibration axis -- |corr(its token logit-shift,
      log-freq)| >= 0.4 AND |cos with w_freq| >= 0.4 (the dominant rank-1
      calibrator lives inside the rank-4 core, consistent with 662/676);
  (b) report top +/- tokens, freq-corr, and cos(w_freq) for each of the 4;
  NULL: a RANDOM residual-space direction has |corr(logit-shift,log-freq)|
      < 0.15 and |cos(w_freq)| < 0.15 (the calibration signature is
      specific to the core directions, not generic)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/workspace/rspd')
from rspd.asvd import generate_lowrank_approximation
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_mlp17_core_readout_results.json'
NCAP = 12
R = 4


@torch.no_grad()
def capture_gate_and_wfreq(fresh):
    cap = []; O = []; lf = []
    mlp = m.transformer.h[17].mlp
    V = m.lm_head.weight.shape[0]
    freq = np.bincount(fresh[:, 1:257].reshape(-1).numpy(), minlength=V).astype(np.float64)
    h1 = mlp.Down.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID).cpu()))
    h2 = mlp.register_forward_hook(
        lambda mo, i_, o_: O.append(o_.detach().float().reshape(-1, D).cpu()))
    for i in range(0, NCAP, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        lf.append(torch.tensor(np.log(freq[tgt.reshape(-1).cpu().numpy()] + 1.0)))
    h1.remove(); h2.remove()
    Xg = torch.cat(cap, 0); Oc = torch.cat(O, 0); lfc = torch.cat(lf, 0)
    wf = ((Oc - Oc.mean(0)) * (lfc - lfc.mean())[:, None]).mean(0)
    wf = (wf / wf.norm()).float()
    return Xg, wf


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NCAP)
    W = m.transformer.h[17].mlp.Down.weight.data.float().cpu()
    Xg, wf = capture_gate_and_wfreq(fresh)

    A_fac, B_fac = generate_lowrank_approximation(W, Xg, target=Xg @ W.T)
    core = A_fac[:, :R]                                   # (1152, 4) output dirs
    core = core / core.norm(dim=0, keepdim=True)          # unit columns

    # unembedding readout: logit shift for pushing +core_j through rms_norm-ish
    # readout. Use W_U @ core_j directly (residual -> logits linear part).
    W_U = m.lm_head.weight.data.float().cpu()             # (V, 1152)
    V = W_U.shape[0]
    freq = np.bincount(fresh[:, 1:257].reshape(-1).numpy(), minlength=V).astype(np.float64)
    log_freq = np.log(freq + 1.0)
    valid = freq > 0

    def d1(t):
        try:
            return cl.d1(int(t))
        except Exception:
            return f'<{t}>'

    # pairwise orthogonality
    G = (core.T @ core).numpy()
    off = G[~np.eye(R, dtype=bool)]
    p0 = np.abs(off).max() < 0.05

    dirs = []
    for j in range(R):
        shift = (W_U @ core[:, j]).numpy()                # (V,) logit shift
        order = np.argsort(-shift)
        toppos = [d1(t) for t in order[:8]]
        topneg = [d1(t) for t in order[::-1][:8]]
        corr = float(np.corrcoef(shift[valid], log_freq[valid])[0, 1])
        cw = float(torch.dot(core[:, j], wf))
        dirs.append({'j': j, 'freq_corr': round(corr, 3), 'cos_wfreq': round(cw, 3),
                     'top_boost': toppos, 'top_suppress': topneg})
        print(f'dir {j}: freq_corr {corr:+.3f}  cos(w_freq) {cw:+.3f}', flush=True)
        print(f'   boost:    {toppos}', flush=True)
        print(f'   suppress: {topneg}', flush=True)

    # random-direction null
    g = torch.Generator().manual_seed(0)
    rd = torch.randn(D, generator=g); rd = rd / rd.norm()
    rshift = (W_U @ rd).numpy()
    rcorr = float(np.corrcoef(rshift[valid], log_freq[valid])[0, 1])
    rcw = float(torch.dot(rd, wf))
    print(f'\nNULL random dir: freq_corr {rcorr:+.3f}  cos(w_freq) {rcw:+.3f}', flush=True)

    best = max(dirs, key=lambda dd: abs(dd['freq_corr']))
    pa = abs(best['freq_corr']) >= 0.4 and abs(best['cos_wfreq']) >= 0.4
    null_ok = abs(rcorr) < 0.15 and abs(rcw) < 0.15
    print(f'\n(0) core dirs orthonormal (max|off-diag|={np.abs(off).max():.3f}<0.05): {p0}',
          flush=True)
    print(f'(a) calibration axis in the core (best dir {best["j"]}: freq_corr '
          f'{best["freq_corr"]}, cos_wfreq {best["cos_wfreq"]}): {pa}', flush=True)
    print(f'NULL random dir uninformative: {null_ok}', flush=True)

    out = {'directions': dirs, 'null_random': {'freq_corr': round(rcorr, 3),
           'cos_wfreq': round(rcw, 3)}, 'max_offdiag': round(float(np.abs(off).max()), 4),
           'best_dir': best['j'], 'pred_0': bool(p0), 'pred_a_calibration_in_core': bool(pa),
           'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
