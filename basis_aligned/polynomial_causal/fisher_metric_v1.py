# FISHER METRIC v1: is the eight-direction object the (label-free) Fisher, does the price obey it at small norm,
# and does it compose through a block by the chain rule?
#
# MATHEMATICAL_REVIEW_2026-08-30_1905.md. The selector that certified on eight fresh windows (lane 1 §2116/§2119)
# ranks mlp4/mlp5 units by their write into the top-8 eigenvectors of G_k = E[dCE/dx_k dCE/dx_k^T] at blocks 5/6 --
# the EMPIRICAL Fisher of the loss pulled back to the stream. Three mathematical claims about it are testable cheaply:
#   (i)  the TRUE Fisher F_k = E_x E_{y~p}[g g^T] (labels sampled from the model's own prediction) has the same top
#        eigen-directions -- the eight are a property of the model and its inputs, not of the labels;
#   (ii) the price of a small stream error is the quadratic form 1/2 delta^T F delta: at r <= 1/4 the measured CE
#        increase for random directions should scale as r^2 and match tr(F) rho^2 / (2 D) within a small factor;
#   (iii) the Fisher composes by the chain rule: pulling G_6's top-8 back through block 5 (VJPs) should land on G_5's
#        top-8 far better than the raw 0.47 overlap of §2111.
#
# SITES: 5 and 6 (the cliff), 256 fresh rows for the Gramians (rows 0-255 as observability_gramian_v1), MC true
# Fisher with 2 label samples per position, price sweep on rows 320-383 at r in {1/16, 1/8, 1/4} with 4 draws.
#
# REGISTERED PREDICTIONS:
#   (a) LABEL-FREE: top-8 subspace overlap (mean cos^2 of principal angles) between the true-Fisher and the
#       empirical-Fisher eight >= 0.7 at both sites. If FALSE the selector depends on the labels and the eight are
#       a data object, not a model object (Kunstner et al. 2019's regime).
#   (b) QUADRATIC REGIME: the log2 price ratio between r = 1/8 and r = 1/4 lies in [1.6, 2.6] at both sites.
#   (c) FISHER PRICES IT: measured mean CE increase at r = 1/8 is within a factor of 2 of the second-order
#       prediction rho^2 * tr(F_k) / (2 D) with rho = r * mean stream norm (random unit directions average the
#       quadratic form to tr/D). If FALSE the empirical Fisher's scale is off even where its shape is right.
#   (d) COMPOSES: overlap between (G_6's top-8 pulled back through block 5 by VJP, then orthonormalised) and G_5's
#       top-8 >= 0.7, against the raw overlap 0.47 (§2111) recomputed here as the reference.
#
# Self-reviewed. Writes fisher_metric_v1_results.json (create-only).
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
NROWS, T, SKIP, BATCH = 256, 256, 64, 8
PRICE_ROWS, PRICE_OFF, NDRAW = 64, 320, 4
RELS = (1 / 16, 1 / 8, 1 / 4)
SITES = (5, 6)
OUT = os.path.join(HERE, 'fisher_metric_v1_results.json')
if os.environ.get('BQLIB_DRYRUN') == '1':
    if not os.path.exists(ROWS_PATH):
        print('DRYRUN FAIL: rows missing'); raise SystemExit(1)
    if os.path.exists(OUT):
        print(f'DRYRUN FAIL: {OUT} exists (create-only)'); raise SystemExit(1)
    print(f'DRYRUN OK: sites {SITES}, empirical + MC-true Fisher on {NROWS} rows, price sweep {RELS}, pullback 6->5')
    raise SystemExit(0)

import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402

import bilin18_observed_model_facade as FAC                               # noqa: E402

T0 = time.time()
M, RECEIPT = FAC.load_bilin18()
DEV = next(M.parameters()).device
D = M.config.n_embd
H = M.transformer.h
ROWS = torch.load(ROWS_PATH, map_location='cpu')[:, :T + 1].long()
GEN = torch.Generator(device=DEV).manual_seed(5)
print(f'model {RECEIPT.weights_sha256[:12]} | rows {tuple(ROWS.shape)}', flush=True)


def forward_from(idx, site, delta=None, leaf_sites=()):
    """Return logits and the dict of leaves (stream tensors with retained grad) at requested sites."""
    x = F.rms_norm(M.transformer.wte(idx), (D,))
    x0, v1, leaves = x, None, {}
    for li, blk in enumerate(H):
        if li == site and delta is not None:
            x = x + delta
        if li in leaf_sites:
            x = x.detach().requires_grad_(True)
            leaves[li] = x
        x, v1 = blk(x, v1, x0)
    logits = M.lm_head(F.rms_norm(x, (D,)))
    return 30 * torch.tanh(logits / 30), leaves


def ce_per_pos(logits, idx):
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                           idx[:, 1:].reshape(-1), reduction='none').view(idx.shape[0], -1)


def gramians(site):
    """Empirical Fisher (true labels) and MC true Fisher (labels ~ p, 2 samples) at one site; also the stream norm."""
    Ge = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    Gt = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    n = 0; nt = 0; xnorm = 0.0
    for s in range(0, NROWS, BATCH):
        idx = ROWS[s:s + BATCH].to(DEV)
        # empirical
        logits, leaves = forward_from(idx, site, leaf_sites=(site,))
        ce_per_pos(logits, idx)[:, SKIP:].sum().backward()
        g = leaves[site].grad[:, SKIP:-1].reshape(-1, D).double(); Ge += g.T @ g; n += g.shape[0]
        xnorm += float(leaves[site].detach()[:, SKIP:-1].norm(dim=-1).sum())
        M.zero_grad(set_to_none=True)
        # true (sampled labels), 2 samples
        with torch.no_grad():
            p = torch.softmax(logits[:, :-1].float(), -1)
        for _ in range(2):
            y = torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=GEN).view(p.shape[0], p.shape[1])
            logits2, leaves2 = forward_from(idx, site, leaf_sites=(site,))
            lp = F.log_softmax(logits2[:, :-1].float(), -1)
            nll = -lp.gather(-1, y[..., None]).squeeze(-1)
            nll[:, SKIP:].sum().backward()
            g = leaves2[site].grad[:, SKIP:-1].reshape(-1, D).double(); Gt += g.T @ g; nt += g.shape[0]
            M.zero_grad(set_to_none=True)
    return Ge / n, Gt / nt, xnorm / n


def top8(G):
    e, Q = torch.linalg.eigh(G)
    return e.flip(0).clamp_min(0), Q.flip(1)[:, :8].float()


def overlap(P, Q):
    return float(((P.T @ Q) ** 2).sum() / 8)


res = {'sites': {}}
G = {}; Ptrue = {}; Pemp = {}; XN = {}
for site in SITES:
    ts = time.time()
    Ge, Gt, xn = gramians(site)
    e_e, P_e = top8(Ge); e_t, P_t = top8(Gt)
    G[site] = Ge; Pemp[site] = P_e; Ptrue[site] = P_t; XN[site] = xn
    res['sites'][str(site)] = {'trace_empirical': float(Ge.trace()), 'trace_true': float(Gt.trace()),
                               'top8_share_empirical': [round(float(v), 4) for v in (e_e[:8] / e_e.sum())],
                               'top8_share_true': [round(float(v), 4) for v in (e_t[:8] / e_t.sum())],
                               'overlap_true_vs_empirical_top8': round(overlap(P_t, P_e), 4),
                               'mean_stream_norm': round(xn, 2), 'seconds': round(time.time() - ts, 1)}
    print(f'site {site}: tr(emp) {float(Ge.trace()):.3e} tr(true) {float(Gt.trace()):.3e} | true-vs-empirical top-8 overlap '
          f'{overlap(P_t, P_e):.3f} | {time.time() - ts:.0f}s', flush=True)

# ---- (b)/(c): small-norm price sweep vs the second-order prediction
IDX = ROWS[PRICE_OFF:PRICE_OFF + PRICE_ROWS].to(DEV)
with torch.no_grad():
    base = float(ce_per_pos(forward_from(IDX, None)[0], IDX)[:, SKIP:].mean())
gen = torch.Generator(device='cpu').manual_seed(56)
for site in SITES:
    prices = {}
    for r in RELS:
        vals = []
        for _ in range(NDRAW):
            u = torch.randn(IDX.shape[0], IDX.shape[1], D, generator=gen).to(DEV)
            u = u / u.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                logits, _ = forward_from(IDX, site, delta=None)
                xs = None
            # perturb: need the stream at site to scale by its own norm -> recompute inside forward
            with torch.no_grad():
                x = F.rms_norm(M.transformer.wte(IDX), (D,)); x0, v1 = x, None
                for li, blk in enumerate(H):
                    if li == site:
                        x = x + r * x.norm(dim=-1, keepdim=True) * u
                    x, v1 = blk(x, v1, x0)
                lg = 30 * torch.tanh(M.lm_head(F.rms_norm(x, (D,))) / 30)
                vals.append(float(ce_per_pos(lg, IDX)[:, SKIP:].mean()) - base)
        prices[str(r)] = round(sum(vals) / NDRAW, 6)
    rho = (1 / 8) * XN[site]
    pred = float(rho ** 2 * G[site].trace() / (2 * D))
    ratio_18_14 = prices[str(1 / 4)] / max(prices[str(1 / 8)], 1e-9)
    res['sites'][str(site)].update({'price_by_r': prices, 'log2_ratio_quarter_over_eighth': round(float(torch.log2(torch.tensor(ratio_18_14))), 3),
                                    'fisher_prediction_at_eighth': pred, 'measured_over_predicted_at_eighth': round(prices[str(1 / 8)] / max(pred, 1e-12), 3)})
    print(f'site {site}: prices {prices} | log2(1/4 over 1/8) {float(torch.log2(torch.tensor(ratio_18_14))):.2f} | '
          f'Fisher prediction at 1/8: {pred:.4e}, measured/predicted {prices[str(1 / 8)] / max(pred, 1e-12):.2f}', flush=True)

# ---- (d): pull G_6's top-8 back through block 5 by VJP, compare with G_5's top-8
pull = torch.zeros(D, 8, device=DEV, dtype=torch.float64); n = 0
for s in range(0, NROWS, BATCH):
    idx = ROWS[s:s + BATCH].to(DEV)
    x = F.rms_norm(M.transformer.wte(idx), (D,)); x0, v1 = x, None
    for li in range(5):
        x, v1 = H[li](x, v1, x0)
    x5 = x.detach().requires_grad_(True)
    x6, _ = H[5](x5, v1, x0)
    for k in range(8):
        q = Pemp[6][:, k]
        (x6[:, SKIP:-1] @ q).sum().backward(retain_graph=(k < 7))
        pull[:, k] += x5.grad[:, SKIP:-1].reshape(-1, D).double().sum(0); x5.grad = None
    n += 1
Ppull = torch.linalg.qr(pull.float())[0]
raw_overlap = overlap(Pemp[5], Pemp[6]); pull_overlap = overlap(Pemp[5], Ppull)
res.update({'raw_overlap_5_6_top8': round(raw_overlap, 4), 'pullback_overlap_5_top8': round(pull_overlap, 4)})
print(f'overlap of G_5 top-8 with G_6 top-8: raw {raw_overlap:.3f} | pulled back through block 5 {pull_overlap:.3f}', flush=True)
pa = all(res['sites'][str(s)]['overlap_true_vs_empirical_top8'] >= 0.7 for s in SITES)
pb = all(1.6 <= res['sites'][str(s)]['log2_ratio_quarter_over_eighth'] <= 2.6 for s in SITES)
pc = all(0.5 <= res['sites'][str(s)]['measured_over_predicted_at_eighth'] <= 2.0 for s in SITES)
pd = pull_overlap >= 0.7
res.update({'model_weights_sha256': RECEIPT.weights_sha256, 'self_reviewed': True,
            'pred_a_label_free': bool(pa), 'pred_b_quadratic_regime': bool(pb), 'pred_c_fisher_prices_it': bool(pc),
            'pred_d_composes_by_chain_rule': bool(pd), 'runtime_s': round(time.time() - T0, 1)})
if os.path.exists(OUT):
    print('output exists; refusing'); raise SystemExit(2)
json.dump(res, open(OUT, 'w'), indent=1)
print(f"(a) true-vs-empirical top-8 overlap >= 0.7 at both sites: {'HELD' if pa else 'FAILED'}")
print(f"(b) log2 ratio in [1.6, 2.6] at both sites: {'HELD' if pb else 'FAILED'}")
print(f"(c) measured/predicted at r=1/8 within [0.5, 2]: {'HELD' if pc else 'FAILED'}")
print(f"(d) pullback overlap {pull_overlap:.3f} >= 0.7 (raw {raw_overlap:.3f}): {'HELD' if pd else 'FAILED'}")
print(f'wrote {OUT} ({time.time() - T0:.0f}s)')
