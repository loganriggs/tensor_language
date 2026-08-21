"""ROUTING RANK CURVE -- confirm 652's boundary. The rank-1 removal did
not touch the newline routing. Is that because the routing is HIGH-RANK
in the post-front residual (removing enough directions would collapse
it), or because it is NOT a residual feature at all (front-attention-
computed, downstream-reconstructed)? And is w_route a good PROBE despite
being causally inert (read!=write)?

Remove the top-r behavior-conditioned directions (r=1,2,4,8,16,32) from
the residual after block 2 and measure the routing R (643). Also report
the out-of-sample AUC of w_route as a line-end PROBE.

REGISTERED PREDICTIONS:
  (0) SANITY: baseline routing R positive (~0.26);
  (a) READOUT, NOT CAUSE: w_route is a GOOD probe -- projecting the
      post-front residual onto it separates end-punct->newline from
      ->not at AUC >= 0.75 out of sample -- yet (652) removing it does
      nothing. Decode != cause.
  (b) NOT A LOW-RANK RESIDUAL FEATURE: even removing the top-32
      behavior-conditioned directions leaves the routing largely intact
      (< 50% lost) -- the routing is not stored in a low-rank residual
      subspace, it is computed by front attention (644) / reconstructed;
  (c) report R at each r and the probe AUC;
  NULL: removing 32 RANDOM directions also barely changes R (so any
      effect at r=32 is not specific)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'routing_rank_curve_results.json'
NFRESH = 48
NL1, NL2 = 198, 628
REMOVE_AFTER = 2


def auc(scores, labels):
    order = np.argsort(scores); ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos = labels == 1; npos = pos.sum(); nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    return float((ranks[pos].sum() - npos * (npos - 1) / 2) / (npos * nneg))


@torch.no_grad()
def forward(fresh, remove_Q, capture=False):
    """remove_Q: (D,k) orthonormal directions to project out after block 2."""
    pnl = torch.zeros(NFRESH, T)
    cap = [] if capture else None
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
            if li == REMOVE_AFTER:
                if capture:
                    cap.append(x.detach().float().reshape(-1, D).cpu())
                if remove_Q is not None:
                    x = x - (x @ remove_Q) @ remove_Q.T
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        pnl[i:i + B] = (p[..., NL1] + p[..., NL2]).cpu()
    pnl = pnl.reshape(-1).numpy()
    return (pnl, torch.cat(cap, 0).numpy()) if capture else pnl


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    cur = fresh[:, :256].reshape(-1).numpy()
    nxt = fresh[:, 1:257].reshape(-1).numpy()

    def end_punct(t):
        s = cl.d1(int(t)).strip()
        return len(s) > 0 and s[-1] in '.!?'
    endp = np.array([end_punct(t) for t in cur])
    follows = np.array([chr(10) in cl.d1(int(t)) for t in nxt])
    A = endp & follows; Bm = endp & ~follows

    base, X2 = forward(fresh, None, capture=True)

    def routing(pnl):
        return float(pnl[A].mean() - pnl[Bm].mean())
    R_base = routing(base)

    # train/test split of end-punct positions for the probe + directions
    epi = np.where(endp)[0]
    rng = np.random.default_rng(0); rng.shuffle(epi)
    half = len(epi) // 2
    tr, te = epi[:half], epi[half:]
    ind = follows.astype(np.float64)
    # top-r behavior-conditioned dirs, fit on TRAIN end-punct positions
    Xtr = X2[tr]; Xc = Xtr - Xtr.mean(0); ic = ind[tr] - ind[tr].mean()
    dirs = []; R = Xc.copy()
    for _ in range(32):
        w = R.T @ ic; w = w / (np.linalg.norm(w) + 1e-9)
        dirs.append(w); R = R - (R @ w)[:, None] * w[None, :]
    Dfull = np.stack(dirs, 1)                       # (D,32)

    # (a) probe AUC of the rank-1 direction, out of sample
    s_te = X2[te] @ Dfull[:, 0]
    probe_auc = auc(s_te, follows[te].astype(int))

    def Q(r):
        q, _ = np.linalg.qr(Dfull[:, :r])
        return torch.tensor(q, dtype=torch.float32, device=DEV)

    curve = {}
    for r in [1, 2, 4, 8, 16, 32]:
        curve[r] = round(routing(forward(fresh, Q(r))), 4)
        print(f'remove top-{r:2d}: R {curve[r]:+.4f}  (lost '
              f'{100*(1-curve[r]/R_base):.0f}%)', flush=True)
    # NULL: remove 32 random
    g = np.random.default_rng(1); rr = g.standard_normal((D, 32))
    qr, _ = np.linalg.qr(rr)
    R_rand32 = routing(forward(fresh, torch.tensor(qr, dtype=torch.float32, device=DEV)))
    print(f'\nbaseline R {R_base:+.4f}; probe AUC (rank-1, OOS) {probe_auc:.3f}',
          flush=True)
    print(f'remove random-32: R {R_rand32:+.4f} (lost {100*(1-R_rand32/R_base):.0f}%)',
          flush=True)

    p0 = R_base > 0.1
    pa = probe_auc >= 0.75
    lost32 = 1 - curve[32] / R_base
    pb = lost32 < 0.5
    null_ok = abs(1 - R_rand32 / R_base) < 0.25
    print(f'\n(0) baseline positive: {p0}', flush=True)
    print(f'(a) w_route good probe (AUC>=0.75) yet causally inert: {pa} '
          f'(AUC {probe_auc:.2f})', flush=True)
    print(f'(b) not low-rank residual (top-32 <50% lost): {pb} '
          f'({100*lost32:.0f}% lost)', flush=True)
    print(f'NULL random-32 barely matters: {null_ok}', flush=True)

    out = {'R_baseline': round(R_base, 4), 'probe_auc_oos': round(probe_auc, 4),
           'remove_topr_R': {str(k): v for k, v in curve.items()},
           'remove_random32_R': round(R_rand32, 4),
           'top32_lost_frac': round(float(lost32), 4),
           'pred_0': bool(p0), 'pred_a_readout_not_cause': bool(pa),
           'pred_b_not_lowrank_residual': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
