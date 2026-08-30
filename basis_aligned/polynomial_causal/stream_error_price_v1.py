# STREAM ERROR PRICE v1: what a unit of early-program error COSTS at every depth, and whether scale is free.
#
# observability_gramian_v1 (OBSERVABILITY_QUOTIENT_V1_RESULT.md) found that the first-order observable subspace
# is two-thirds of the stream and that direction matters far less than magnitude: a relative-norm-0.5 random
# perturbation costs 0.015 / 0.067 / 1.07 nat at blocks 2 / 5 / 9. That is the number a compressed program's
# error budget is priced in, and it exists for three sites. This measures it at all 18 sites and three norms,
# plus the one perturbation the pre-RMSNorm architecture should make nearly free -- a pure RESCALING of the
# stream -- because a program that gets the stream's direction right and its scale wrong should, if so, pay
# almost nothing, and lane 1's §1818/§1819 (head 5.7 writes the same vector 159x too large) says scale errors
# are exactly what compiled programs make.
#
# ARMS at site k in 0..17 (perturbation added to the stream entering block k, 64 fresh rows, 8 draws):
#   random     delta = r * ||x|| * u,  u a random unit direction per position
#   rescale    delta = r * x           (the stream scaled by 1 + r; same per-position norm as `random`)
#   sign-flip  delta = -2x on a random 10% of positions (the stream inverted there; a shape-preserving null of
#              large magnitude, descriptive only)
# at relative norms r in {0.25, 0.5, 1.0}. Price = mean CE increase over positions >= 64.
#
# REGISTERED PREDICTIONS:
#   (a) PRICE RISES WITH DEPTH: Spearman rho(k, price_random(k, r=0.5)) >= 0.8 over the 18 sites. The three
#       measured points order that way (0.015 < 0.067 < 1.07); the bar asks the whole curve to. If FALSE, there
#       is a cheap-error band somewhere in the middle, which is where a lossy program should live.
#   (b) PRICE IS SUPERLINEAR IN NORM: price(r=1.0) >= 2 * price(r=0.5) at every site. First order gives 2x;
#       the three measured sites gave 23x / 7.6x / 3.0x. If FALSE at some site the loss is locally linear in
#       error there and a first-order budget is exact at that depth.
#   (c) SCALE IS CHEAP: price_rescale(k, r) <= 0.2 * price_random(k, r) at every site for r = 0.5. Pre-norm
#       blocks see F.rms_norm(x); a rescaled stream re-enters every block identically except through the
#       residual sum and the lambdas mixing with x0, so the cost should be a small fraction of a random error
#       of the same norm. If FALSE, scale is NOT a free gauge of the stream and scale-error programs (§1818)
#       pay in full -- important for pricing, and the opposite of what the architecture suggests.
#
# Self-reviewed. Writes stream_error_price_v1_results.json (create-only).
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
NROWS, T, SKIP, NDRAW = 64, 256, 64, 8
RELS = (0.25, 0.5, 1.0)
OUT = os.path.join(HERE, 'stream_error_price_v1_results.json')
if os.environ.get('BQLIB_DRYRUN') == '1':
    if not os.path.exists(ROWS_PATH):
        print(f'DRYRUN FAIL: missing {ROWS_PATH}'); raise SystemExit(1)
    if os.path.exists(OUT):
        print(f'DRYRUN FAIL: {OUT} exists (create-only)'); raise SystemExit(1)
    print(f'DRYRUN OK: 18 sites x {RELS} x {NDRAW} draws x 3 arms on {NROWS} rows')
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
# rows 256.. are untouched by the Gramian scripts' first 320 rows? They used rows 0-255 (Gramian) and 256-319
# (perturbation). Use 320-383 here so the price curve is on rows no earlier observability artifact scored.
IDX = torch.load(ROWS_PATH, map_location='cpu')[320:320 + NROWS, :T + 1].long().to(DEV)
print(f'model {RECEIPT.weights_sha256[:12]} | rows {tuple(IDX.shape)} | {NL} sites', flush=True)


@torch.no_grad()
def run(site=None, fn=None):
    x = F.rms_norm(M.transformer.wte(IDX), (D,))
    x0, v1 = x, None
    for li, blk in enumerate(H):
        if li == site:
            x = fn(x)
        x, v1 = blk(x, v1, x0)
    logits = M.lm_head(F.rms_norm(x, (D,)))
    logits = 30 * torch.tanh(logits / 30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                         IDX[:, 1:].reshape(-1), reduction='none').view(IDX.shape[0], -1)
    return float(ce[:, SKIP:].mean())


BASE = run()
gen = torch.Generator(device='cpu').manual_seed(1)
sites = []
for k in range(NL):
    ts = time.time()
    rec = {'site': k}
    for r in RELS:
        prices = {'random': [], 'rescale': [], 'signflip': []}
        for draw in range(NDRAW):
            u = torch.randn(IDX.shape[0], IDX.shape[1], D, generator=gen).to(DEV)
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            mask = (torch.rand(IDX.shape[0], IDX.shape[1], generator=gen) < 0.10).to(DEV)[..., None]
            prices['random'].append(run(k, lambda x, u=u, r=r: x + r * x.norm(dim=-1, keepdim=True) * u) - BASE)
            if draw == 0:
                prices['rescale'].append(run(k, lambda x, r=r: (1 + r) * x) - BASE)
            if draw < 2 and r == 0.5:
                prices['signflip'].append(run(k, lambda x, m=mask: torch.where(m, -x, x)) - BASE)
        rec[str(r)] = {a: {'mean': round(sum(v) / len(v), 5), 'min': round(min(v), 5), 'max': round(max(v), 5)}
                       for a, v in prices.items() if v}
    sites.append(rec)
    print(f'site {k:2d}: ' + ' | '.join(f'r{r}: rnd {rec[str(r)]["random"]["mean"]:+.4f} scale '
                                       f'{rec[str(r)]["rescale"]["mean"]:+.4f}' for r in RELS)
          + f' | flip {rec["0.5"]["signflip"]["mean"]:+.4f} | {time.time() - ts:.0f}s', flush=True)


def spearman(x, y):
    rx = torch.tensor(x).double().argsort().argsort().double(); ry = torch.tensor(y).double().argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))


p05 = [s['0.5']['random']['mean'] for s in sites]
p10 = [s['1.0']['random']['mean'] for s in sites]
sc05 = [s['0.5']['rescale']['mean'] for s in sites]
rho = spearman(list(range(NL)), p05)
pa = rho >= 0.8
pb = all(b >= 2 * a for a, b in zip(p05, p10))
pc = all(s <= 0.2 * max(r, 1e-9) for s, r in zip(sc05, p05))
out = {'model_weights_sha256': RECEIPT.weights_sha256, 'rows': NROWS, 'row_offset': 320, 'base_ce': round(BASE, 5),
       'rel_norms': list(RELS), 'sites': sites, 'price_random_r05_by_site': [round(v, 5) for v in p05],
       'price_random_r10_by_site': [round(v, 5) for v in p10], 'price_rescale_r05_by_site': [round(v, 5) for v in sc05],
       'spearman_depth_vs_price_r05': round(rho, 4),
       'superlinear_ratio_r10_over_r05': [round(b / max(a, 1e-9), 3) for a, b in zip(p05, p10)],
       'rescale_over_random_r05': [round(s / max(r, 1e-9), 3) for s, r in zip(sc05, p05)],
       'self_reviewed': True, 'pred_a_price_rises_with_depth': bool(pa),
       'pred_b_superlinear_in_norm': bool(pb), 'pred_c_scale_is_cheap': bool(pc),
       'runtime_s': round(time.time() - T0, 1)}
if os.path.exists(OUT):
    print(f'{OUT} exists; refusing'); raise SystemExit(2)
json.dump(out, open(OUT, 'w'), indent=1)
print(f'(a) rho(depth, price r=0.5) {rho:+.3f} >= 0.8: {"HELD" if pa else "FAILED"}')
print(f'(b) price(1.0) >= 2 price(0.5) at every site: {"HELD" if pb else "FAILED"}  min ratio '
      f'{min(out["superlinear_ratio_r10_over_r05"]):.2f}')
print(f'(c) rescale <= 0.2 random at r=0.5 every site: {"HELD" if pc else "FAILED"}  max ratio '
      f'{max(out["rescale_over_random_r05"]):.3f}')
print(f'wrote {OUT} ({time.time() - T0:.0f}s)')
