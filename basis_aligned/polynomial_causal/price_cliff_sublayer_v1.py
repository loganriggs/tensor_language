# PRICE CLIFF, SUBLAYER-RESOLVED: does the 25x jump in the price of stream error happen at an ATTENTION write or
# an MLP write, and is the cost conducted across positions?
#
# STREAM_ERROR_PRICE_V1_RESULT.md: a half-norm random error costs 0.058 nat at the input of block 5 and 1.48 nat
# at the input of block 6 -- a 25x jump across one block -- then 1.81 at block 7 and a slow decay. Block inputs
# are the only sites that run measured; each block is attention THEN MLP, so the jump could sit at attn5 (the
# last gatherer head band, lane 1 s998/s1044), at mlp5, at attn6, or be smeared. That decides what a program
# must reproduce exactly: the gatherer's write, or the first mid-band MLP's.
#
# SITES (perturbation added to the stream at that point, 64 fresh rows 384-447 -- untouched by every earlier
# observability artifact -- 8 draws, relative norms 0.25 / 0.5 / 1.0):
#   b5.in    input of block 5          b5.attn   after block 5's attention residual add (before mlp5)
#   b6.in    input of block 6          b6.attn   after block 6's attention residual add
#   b7.in    input of block 7
# plus, at b6.in and r = 0.5, a SINGLE-POSITION arm: one random position t0 >= 64 per row is perturbed, and the
# CE increase is split into the perturbed position itself and all later positions (conduction through attention).
#
# REGISTERED PREDICTIONS:
#   (a) THE CLIFF IS AN ATTENTION WRITE: price(b5.attn, r=0.5) >= 5 * price(b5.in, r=0.5), i.e. most of the 25x
#       is already present once attn5 has written and before mlp5 runs. If FALSE the jump is at mlp5 (or attn6)
#       and the gatherer band's LAST head write is not the expensive one.
#   (b) THE MLP DOES NOT ADD A SECOND CLIFF: price(b6.in) <= 2 * price(b5.attn) at r = 0.5. If FALSE, mlp5's
#       write is itself a cliff and the two sublayers must both be reproduced exactly.
#   (c) THE COST IS CONDUCTED: at b6.in, r = 0.5, single-position arm, the CE increase summed over LATER positions
#       is >= 0.5 of the increase at the perturbed position itself (per row, mean over rows and draws). A
#       per-position error at block 6 that only hurt its own position would be a local error; one that hurts the
#       rest of the row is read by later attention and is what makes the band expensive. If FALSE the block-6
#       price is local and a program's error there is contained.
#
# Self-reviewed. Writes price_cliff_sublayer_v1_results.json (create-only).
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
NROWS, T, SKIP, NDRAW, ROW0 = 64, 256, 64, 8, 384
RELS = (0.25, 0.5, 1.0)
SITES = ('b5.in', 'b5.attn', 'b6.in', 'b6.attn', 'b7.in')
OUT = os.path.join(HERE, 'price_cliff_sublayer_v1_results.json')
PRIOR = os.path.join(HERE, 'stream_error_price_v1_results.json')
if os.environ.get('BQLIB_DRYRUN') == '1':
    if not os.path.exists(ROWS_PATH) or not os.path.exists(PRIOR):
        print('DRYRUN FAIL: missing rows or the price-curve artifact'); raise SystemExit(1)
    if os.path.exists(OUT):
        print(f'DRYRUN FAIL: {OUT} exists (create-only)'); raise SystemExit(1)
    p = json.load(open(PRIOR))['price_random_r05_by_site']
    print(f'DRYRUN OK: prior block-5/6 prices {p[5]:.4f}/{p[6]:.4f}; 5 sublayer sites x {RELS} x {NDRAW} draws')
    raise SystemExit(0)

import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402

import bilin18_observed_model_facade as FAC                               # noqa: E402

T0 = time.time()
M, RECEIPT = FAC.load_bilin18()
DEV = next(M.parameters()).device
D = M.config.n_embd
H = M.transformer.h
IDX = torch.load(ROWS_PATH, map_location='cpu')[ROW0:ROW0 + NROWS, :T + 1].long().to(DEV)
PRIOR_P = json.load(open(PRIOR))['price_random_r05_by_site']
print(f'model {RECEIPT.weights_sha256[:12]} | rows {tuple(IDX.shape)} | prior block5/6 r=0.5 prices '
      f'{PRIOR_P[5]:.4f}/{PRIOR_P[6]:.4f}', flush=True)


@torch.no_grad()
def run(site=None, fn=None):
    """Block.forward replicated so a perturbation can be applied between the attention and MLP residual adds."""
    x = F.rms_norm(M.transformer.wte(IDX), (D,))
    x0, v1 = x, None
    for li, blk in enumerate(H):
        if site == f'b{li}.in':
            x = fn(x)
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        x1, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        x = x + x1
        if site == f'b{li}.attn':
            x = fn(x)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    logits = M.lm_head(F.rms_norm(x, (D,)))
    logits = 30 * torch.tanh(logits / 30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                         IDX[:, 1:].reshape(-1), reduction='none').view(IDX.shape[0], -1)
    return ce


BASE_CE = run()
BASE = float(BASE_CE[:, SKIP:].mean())
check = float(run('b0.in', lambda x: x)[:, SKIP:].mean())
if abs(check - BASE) > 1e-6:
    print('replicated block forward differs from the model; refusing'); raise SystemExit(2)
gen = torch.Generator(device='cpu').manual_seed(56)
prices = {}
for site in SITES:
    prices[site] = {}
    for r in RELS:
        vals = []
        for _ in range(NDRAW):
            u = torch.randn(IDX.shape[0], IDX.shape[1], D, generator=gen).to(DEV)
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            ce = run(site, lambda x, u=u, r=r: x + r * x.norm(dim=-1, keepdim=True) * u)
            vals.append(float(ce[:, SKIP:].mean()) - BASE)
        prices[site][str(r)] = {'mean': round(sum(vals) / NDRAW, 5), 'min': round(min(vals), 5), 'max': round(max(vals), 5)}
    print(f'{site:8s}: ' + ' | '.join(f'r{r} {prices[site][str(r)]["mean"]:+.4f}' for r in RELS) + f' | {time.time() - T0:.0f}s', flush=True)

# single-position conduction arm at b6.in, r = 0.5
own_tot, later_tot = [], []
for _ in range(NDRAW):
    t0s = torch.randint(SKIP, T - 32, (IDX.shape[0],), generator=gen).to(DEV)
    u = torch.randn(IDX.shape[0], D, generator=gen).to(DEV)
    u = u / u.norm(dim=-1, keepdim=True)

    def single(x, t0s=t0s, u=u):
        x = x.clone()
        rows = torch.arange(x.shape[0], device=DEV)
        x[rows, t0s] = x[rows, t0s] + 0.5 * x[rows, t0s].norm(dim=-1, keepdim=True) * u
        return x

    ce = run('b6.in', single)
    d = ce - BASE_CE                                             # (rows, T) per-position increase
    rows = torch.arange(IDX.shape[0], device=DEV)
    own = d[rows, t0s - 0]                                       # CE at position t0 predicts token t0+1
    later = torch.stack([d[i, t0s[i] + 1:].sum() for i in range(IDX.shape[0])])
    own_tot.append(float(own.mean())); later_tot.append(float(later.mean()))
own_m, later_m = sum(own_tot) / NDRAW, sum(later_tot) / NDRAW
p5in, p5at, p6in = prices['b5.in']['0.5']['mean'], prices['b5.attn']['0.5']['mean'], prices['b6.in']['0.5']['mean']
pa = p5at >= 5 * p5in
pb = p6in <= 2 * p5at
pc = later_m >= 0.5 * own_m
out = {'model_weights_sha256': RECEIPT.weights_sha256, 'rows': NROWS, 'row_offset': ROW0, 'base_ce': round(BASE, 5),
       'prices': prices, 'prior_block_input_prices_r05': {'5': PRIOR_P[5], '6': PRIOR_P[6], '7': PRIOR_P[7]},
       'single_position_b6_r05': {'own_position_mean_dCE': round(own_m, 5), 'later_positions_sum_mean_dCE': round(later_m, 5),
                                  'later_over_own': round(later_m / max(own_m, 1e-9), 4)},
       'self_reviewed': True, 'pred_a_cliff_is_attention_write': bool(pa),
       'pred_b_mlp_adds_no_second_cliff': bool(pb), 'pred_c_cost_is_conducted': bool(pc),
       'runtime_s': round(time.time() - T0, 1)}
if os.path.exists(OUT):
    print(f'{OUT} exists; refusing'); raise SystemExit(2)
json.dump(out, open(OUT, 'w'), indent=1)
print(f'single position @ b6.in r=0.5: own {own_m:+.4f} | later (sum) {later_m:+.4f} | ratio {later_m / max(own_m, 1e-9):.3f}')
print(f'(a) b5.attn {p5at:+.4f} >= 5 x b5.in {p5in:+.4f}: {"HELD" if pa else "FAILED"}')
print(f'(b) b6.in {p6in:+.4f} <= 2 x b5.attn: {"HELD" if pb else "FAILED"}')
print(f'(c) later/own {later_m / max(own_m, 1e-9):.3f} >= 0.5: {"HELD" if pc else "FAILED"}')
print(f'wrote {OUT} ({time.time() - T0:.0f}s)')
