# OBSERVABILITY QUOTIENT v1, companion: the DEPTH PROFILE of the observable rank at all 18 stream sites.
#
# observability_gramian_v1.py measures the loss-gradient Gramian G_k = E[g g^T], g = dCE_t/dx_k(t), at three
# sites with a causal perturbation test. This companion is spectra only, at EVERY site 0..17, from one backward
# pass per batch (every block input is a leaf), so the quotient's size can be read as a function of depth and
# compared with the activation covariance at the same site. It is the map the alternate entry point needs
# before choosing WHERE to factor: a site whose observable rank is a small fraction of its activation rank is
# where "factor only the quotient" buys the most.
#
# WHAT IS ALREADY KNOWN, stated before registering: the three-site run's first site landed while this was being
# written -- at block 2 the observable r90 is 737 of 1152 while the activation covariance's r90 is 264 (its r50
# is ONE: a single massive direction carries half the activation energy, so activation r90 is not a fair
# denominator). The first-order quotient at block 2 is NOT small: the loss is sensitive to ~64% of the stream's
# dimensions. Predictions below are therefore about the UNSEEN sites and about depth, with 737 as the anchor.
#
# REGISTERED PREDICTIONS (r90 = eigenvalues to 90% of trace):
#   (a) THE QUOTIENT SHRINKS TOWARD THE READOUT: r90(G_17) <= 0.5 * 737. Near the output the gradient is
#       J_lm^T (p - onehot) and the residual stream has fewer blocks left to mix it, so fewer directions should
#       matter. If FALSE, the observable subspace stays >= 2/3 of the stream to the end, and "factor only the
#       quotient" buys at most a third at first order anywhere.
#   (b) DEPTH ORDERS IT: |Spearman rho(k, r90(G_k))| >= 0.5 over k = 0..17, sign reported. If FALSE the
#       observable rank is flat in depth and the quotient could be factored once rather than per site.
#   (c) DOCUMENT STABILITY EVERYWHERE: the r90 observable subspace fitted on the first 128 rows captures >= 0.80
#       of gradient energy on the other 128 at every site (the §2098 standard). If it fails at a site, that
#       site's quotient is not a fixed object and cannot be priced.
#
# Descriptive: r50/r90/r99 of both spectra per site and their ratio; gradient trace per site (how much the loss
# is sensitive to the stream at all, by depth); overlap of each site's top-8 observable directions with the
# next site's.
#
# Self-reviewed. Writes observability_depth_profile_v1_results.json (create-only).
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BQ = os.path.join(ROOT, 'basis_aligned', 'bilinear_quotient')
for p in (ROOT, HERE, BQ):
    sys.path.insert(0, p)
os.chdir(HERE)

ROWS_PATH = os.path.join(BQ, 'bilin18_eval_tokens_large.pt')
NROWS, T, SKIP, BATCH = 256, 256, 64, 4
OUT = os.path.join(HERE, 'observability_depth_profile_v1_results.json')
if os.environ.get('BQLIB_DRYRUN') == '1':
    if not os.path.exists(ROWS_PATH):
        print(f'DRYRUN FAIL: missing {ROWS_PATH}'); raise SystemExit(1)
    if os.path.exists(OUT):
        print(f'DRYRUN FAIL: {OUT} exists (create-only)'); raise SystemExit(1)
    print(f'DRYRUN OK: 18 sites, {NROWS} rows, batch {BATCH}')
    raise SystemExit(0)

import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402

import bilin18_observed_model_facade as FAC                               # noqa: E402

T0 = time.time()
M, RECEIPT = FAC.load_bilin18()
DEV = next(M.parameters()).device
D = M.config.n_embd
H = M.transformer.h
NL = len(H)
ROWS = torch.load(ROWS_PATH, map_location='cpu')[:, :T + 1].long()
print(f'model {RECEIPT.weights_sha256[:12]} | rows {tuple(ROWS.shape)} | {NL} sites', flush=True)

G = [torch.zeros(D, D, device=DEV, dtype=torch.float64) for _ in range(NL)]
GA = [torch.zeros(D, D, device=DEV, dtype=torch.float64) for _ in range(NL)]
GB = [torch.zeros(D, D, device=DEV, dtype=torch.float64) for _ in range(NL)]
C = [torch.zeros(D, D, device=DEV, dtype=torch.float64) for _ in range(NL)]
n = 0
for s in range(0, NROWS, BATCH):
    idx = ROWS[s:s + BATCH].to(DEV)
    x = F.rms_norm(M.transformer.wte(idx), (D,))
    x0, v1, leaves = x, None, []
    for blk in H:
        x = x.detach().requires_grad_(True)
        leaves.append(x)
        x, v1 = blk(x, v1, x0)
    logits = M.lm_head(F.rms_norm(x, (D,)))
    logits = 30 * torch.tanh(logits / 30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                         idx[:, 1:].reshape(-1), reduction='none').view(idx.shape[0], -1)
    ce[:, SKIP:].sum().backward()
    for k, leaf in enumerate(leaves):
        g = leaf.grad[:, SKIP:-1].reshape(-1, D).double()
        a = leaf.detach()[:, SKIP:-1].reshape(-1, D).double()
        G[k] += g.T @ g
        (GA if s < NROWS // 2 else GB)[k].add_(g.T @ g)
        C[k] += (a - a.mean(0)).T @ (a - a.mean(0))
    n += (idx.shape[1] - 1 - SKIP) * idx.shape[0]
    M.zero_grad(set_to_none=True)
    if s % 64 == 0:
        print(f'  rows {s + BATCH}/{NROWS} {time.time() - T0:.0f}s', flush=True)


def r_at(eig, frac):
    c = torch.cumsum(eig, 0) / eig.sum()
    return int((c < frac).sum()) + 1


sites = []
prev_top = None
for k in range(NL):
    eg, Q = torch.linalg.eigh(G[k] / n)
    eg, Q = eg.flip(0).clamp_min(0), Q.flip(1)
    ec = torch.linalg.eigvalsh(C[k] / n).flip(0).clamp_min(0)
    r90 = r_at(eg, 0.9)
    QA = torch.linalg.eigh(GA[k])[1].flip(1)[:, :r90]
    transfer = float((QA.T @ GB[k] @ QA).trace() / GB[k].trace())
    top = Q[:, :8]
    overlap = None if prev_top is None else float(((prev_top.T @ top) ** 2).sum() / 8)
    prev_top = top
    rec = {'site': k, 'gramian': {f'r{int(f * 100)}': r_at(eg, f) for f in (0.5, 0.9, 0.99)},
           'activation': {f'r{int(f * 100)}': r_at(ec, f) for f in (0.5, 0.9, 0.99)},
           'ratio_r90': round(r90 / r_at(ec, 0.9), 4), 'gradient_trace': float(eg.sum()),
           'observable_transfer_A_to_B': round(transfer, 4),
           'top8_overlap_with_previous_site': None if overlap is None else round(overlap, 4)}
    sites.append(rec)
    print(f'site {k:2d}: r90 obs {r90:4d} / act {rec["activation"]["r90"]:4d} = {rec["ratio_r90"]:.3f} | '
          f'trace {rec["gradient_trace"]:.3e} | transfer {transfer:.3f}', flush=True)


def spearman(x, y):
    rx = torch.tensor(x).double().argsort().argsort().double(); ry = torch.tensor(y).double().argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))


ANCHOR_SITE2 = 737
r90s = [s['gramian']['r90'] for s in sites]
rho_depth = spearman(list(range(NL)), r90s)
pa = r90s[NL - 1] <= 0.5 * ANCHOR_SITE2
pb = abs(rho_depth) >= 0.5
pc = all(s['observable_transfer_A_to_B'] >= 0.80 for s in sites)
out = {'model_weights_sha256': RECEIPT.weights_sha256, 'rows': NROWS, 'positions_per_row_scored': T - SKIP,
       'sites': sites, 'anchor_r90_site2_from_three_site_run': ANCHOR_SITE2,
       'r90_obs_by_site': r90s, 'spearman_depth_vs_r90_obs': round(rho_depth, 4), 'self_reviewed': True,
       'pred_a_quotient_shrinks_toward_readout': bool(pa), 'pred_b_depth_orders_it': bool(pb),
       'pred_c_document_stable_everywhere': bool(pc), 'runtime_s': round(time.time() - T0, 1)}
if os.path.exists(OUT):
    print(f'{OUT} exists; refusing'); raise SystemExit(2)
json.dump(out, open(OUT, 'w'), indent=1)
print(f'(a) r90 at site 17 = {r90s[-1]} <= 0.5 * {ANCHOR_SITE2}: {"HELD" if pa else "FAILED"}')
print(f'(b) |rho(depth, r90 obs)| = {abs(rho_depth):.3f} >= 0.5 (sign {rho_depth:+.3f}): {"HELD" if pb else "FAILED"}')
print(f'(c) transfer >= 0.80 at every site: {"HELD" if pc else "FAILED"}')
print(f'wrote {OUT} ({time.time() - T0:.0f}s)')
