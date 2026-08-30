# MECHANISM OF THE SELECTION GAIN, v2 (rung 15b): the HEAVY HEAD of the Gramian, not its 90% span.
#
# S2108 refuted the r90 version: every unit has ~0.68 of its Down energy in the r90 observable span (which is two-thirds
# of the stream), swapped-in 0.685 vs swapped-out 0.670. But the metric ranks on ||G^{1/2} Down_u||, dominated by G's
# top eigenvalues (r50 = 70 at block 5). This tests the head: energy fraction of Down columns in the top-r50 and top-8
# eigen-directions of the site's Gramian, IN vs OUT, same rankings and keep-sets as S2108.
#
# REGISTERED PREDICTIONS (each for mlp4 AND mlp5, K = 2304):
#   (a) IN units load on the heavy head: median top-r50 energy fraction of IN >= 1.5 x that of OUT.
#   (b) and on its very top: median top-8 energy fraction of IN >= 1.5 x that of OUT.
#   (c) INSTRUMENT: the shared fraction reproduces S2108's 0.750 / 0.749 within 0.01.
# If (a) fails too, the metric's gain is not a matter of which directions the kept units write into at all, and the
# weights-side story for S2105-S2107 stays open.
#
# ORIGINAL HEADER FOLLOWS.
#
# BENCHMARK_BACKLOG rung 15. §2105-§2107: selecting mlp4/mlp5's kept CP units by importance under the block-5/6
# first-order observability metric instead of raw output norm buys 0.124 / 0.075 nat at equal stored values on two
# windows, and metric-1152 matches norm-2304. The claim behind it is mechanical: the metric keeps units whose Down
# columns point into the subspace the loss reads at the block after the MLP (which, for mlp4, is what attn5 reads
# and amplifies 8.6x, §2102). This checks that directly on the weights, with no fits and no arms.
#
# METHOD. For mlp4 (metric at block 5's input) and mlp5 (block 6's input): recompute the two importance rankings on the
# FIT rows' Gramian exactly as ops/metric_units_ksweep.py does, take the K = 2304 keep-sets, and split units into
# IN (metric keeps, norm drops), OUT (norm keeps, metric drops), BOTH. For every unit u report the observable fraction
# f_u = ||P^T Down[:,u]||^2 / ||Down[:,u]||^2 with P the top-r90 eigenvectors of the site's Gramian, and the raw
# importance ||Down_u|| ||L_u|| ||R_u||.
#
# REGISTERED PREDICTIONS (each for mlp4 AND mlp5):
#   (a) IN units write into the observable subspace: median f(IN) >= 2 x median f(OUT). This is the mechanism; if
#       FALSE the metric is choosing on something other than direction (e.g. scale interactions) and §2105's
#       reading of the gain is wrong even though the gain stands.
#   (b) OUT units are louder: median raw importance of OUT >= median raw importance of IN. True by construction of the
#       two rankings if the instrument is right; a failed (b) means the rankings were recomputed wrongly and (a) is
#       not readable.
#   (c) THE SWAP IS A MINORITY: |IN| = |OUT| <= 30% of 2304 (>= 70% of kept units are shared). If FALSE the two
#       selectors disagree on most of the layer and "a targeted swap" is the wrong description.
#
# Descriptive: the same split at K = 1152; the observable fraction distribution of all 4608 units per layer.
# Self-reviewed. Writes metric_units_mechanism_results.json.
import json, sys, time, torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import os

if os.environ.get('BQLIB_DRYRUN') == '1':
    _bq = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    if not os.path.exists(_bq + 'metric_units_mechanism_results.json'):
        print('DRYRUN FAIL: S2108 artifact absent'); raise SystemExit(1)
    print('DRYRUN OK: weights + two Gramians, no fits')
    raise SystemExit(0)

import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV

D = 1152; TT = 256; SKIP = 64; CA, CB = 300, 512
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'metric_units_mechanism2_results.json'
t0 = time.time()
TOKF = torch.cat([FW[i:i + 4, :257] for i in range(CA, CB, 4)]).to(DEV)


def gramian(site):
    G = torch.zeros(D, D, device=DEV, dtype=torch.float64); n = 0
    for b0 in range(0, TOKF.shape[0], 4):
        idx = TOKF[b0:b0 + 4, :-1]; tg = TOKF[b0:b0 + 4, 1:].reshape(-1)
        with torch.enable_grad():
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None; leaf = None
            for li, blk in enumerate(m.transformer.h):
                if li == site:
                    x = x.detach().requires_grad_(True); leaf = x
                x, v1 = blk(x, v1, x0)
            lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
            ce = F.cross_entropy(lg.view(-1, lg.size(-1)), tg, reduction='none').view(idx.shape[0], TT)
            ce[:, SKIP:].sum().backward()
        g = leaf.grad[:, SKIP:].reshape(-1, D).double(); G += g.T @ g; n += g.shape[0]
    return G / n


res = {}
for li, site in ((4, 5), (5, 6)):
    G = gramian(site)
    e, Q = torch.linalg.eigh(G); e, Q = e.flip(0).clamp_min(0), Q.flip(1)
    c = torch.cumsum(e, 0) / e.sum(); r90 = int((c < 0.9).sum()) + 1
    c50 = int((c < 0.5).sum()) + 1
    P = Q[:, :r90].float(); P50 = Q[:, :c50].float(); P8 = Q[:, :8].float()
    ef = e.clamp_min(1e-3 * float(e.max()))
    half = ((Q * ef.sqrt()[None, :]) @ Q.T).float()
    mlp = m.transformer.h[li].mlp
    L = mlp.Left.weight.detach().float(); Rw = mlp.Right.weight.detach().float(); Dw = mlp.Down.weight.detach().float()
    imp_norm = Dw.norm(dim=0) * L.norm(dim=1) * Rw.norm(dim=1)
    imp_metric = (half @ Dw).norm(dim=0) * L.norm(dim=1) * Rw.norm(dim=1)
    den = (Dw ** 2).sum(0).clamp_min(1e-12)
    frac = ((P.T @ Dw) ** 2).sum(0) / den
    frac50 = ((P50.T @ Dw) ** 2).sum(0) / den
    frac8 = ((P8.T @ Dw) ** 2).sum(0) / den
    rec = {'site': site, 'r90': r90, 'r50': c50, 'units': int(Dw.shape[1]),
           'observable_fraction_all': {'median': round(float(frac.median()), 4), 'p10': round(float(frac.quantile(0.1)), 4),
                                       'p90': round(float(frac.quantile(0.9)), 4)}}
    for K in (2304, 1152):
        kn = set(imp_norm.argsort(descending=True)[:K].tolist()); km = set(imp_metric.argsort(descending=True)[:K].tolist())
        IN = sorted(km - kn); OUTu = sorted(kn - km); BOTH = sorted(kn & km)
        def med(idx, v): return round(float(v[torch.tensor(idx)].median()), 5) if idx else None
        rec[str(K)] = {'n_in': len(IN), 'n_out': len(OUTu), 'n_both': len(BOTH), 'shared_fraction': round(len(BOTH) / K, 4),
                       'obs_fraction_median': {'in': med(IN, frac), 'out': med(OUTu, frac), 'both': med(BOTH, frac)},
                       'top_r50_fraction_median': {'in': med(IN, frac50), 'out': med(OUTu, frac50), 'both': med(BOTH, frac50)},
                       'top8_fraction_median': {'in': med(IN, frac8), 'out': med(OUTu, frac8), 'both': med(BOTH, frac8)},
                       'raw_importance_median': {'in': med(IN, imp_norm), 'out': med(OUTu, imp_norm), 'both': med(BOTH, imp_norm)}}
    res[f'mlp{li}'] = rec
    r = rec['2304']
    print(f"mlp{li} (site {site}, r90 {r90}, r50 {c50}): K=2304 shared {r['shared_fraction']:.3f} | top-r50 in {r['top_r50_fraction_median']['in']} out {r['top_r50_fraction_median']['out']} | top-8 in {r['top8_fraction_median']['in']} out {r['top8_fraction_median']['out']} | r90 in {r['obs_fraction_median']['in']} "
          f"out {r['obs_fraction_median']['out']} both {r['obs_fraction_median']['both']} | raw importance in {r['raw_importance_median']['in']:.4g} "
          f"out {r['raw_importance_median']['out']:.4g} | all-units obs median {rec['observable_fraction_all']['median']}", flush=True)
pa = all(res[k]['2304']['top_r50_fraction_median']['in'] >= 1.5 * res[k]['2304']['top_r50_fraction_median']['out'] for k in res)
pb = all(res[k]['2304']['top8_fraction_median']['in'] >= 1.5 * res[k]['2304']['top8_fraction_median']['out'] for k in res)
pc = abs(res['mlp4']['2304']['shared_fraction'] - 0.750) <= 0.01 and abs(res['mlp5']['2304']['shared_fraction'] - 0.749) <= 0.01
out = {'layers': res, 'pred_a_in_units_load_on_top_r50': bool(pa), 'pred_b_in_units_load_on_top8': bool(pb),
       'pred_c_reproduces_S2108_shares': bool(pc), 'self_reviewed': True, 'runtime_s': round(time.time() - t0, 1)}
json.dump(out, open(OUT, 'w'), indent=1)
print(f"(a) top-r50 fraction IN >= 1.5 x OUT at both layers: {'HELD' if pa else 'FAILED'}")
print(f"(b) top-8 fraction IN >= 1.5 x OUT at both layers: {'HELD' if pb else 'FAILED'}")
print(f"(c) shared fractions reproduce S2108: {'HELD' if pc else 'FAILED'}")
print(f'wrote {OUT} ({time.time() - t0:.0f}s)')
