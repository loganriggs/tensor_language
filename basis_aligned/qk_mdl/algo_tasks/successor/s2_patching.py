"""Step 2: single-component activation patching (clean -> corrupted) over all
180 components (162 heads + 18 MLPs). Metric: recovered fraction of the
correct-successor logit margin  (sum over pairs of m_patch - m_corr) /
(sum of m_clean - m_corr), margin = logit[clean_ans] - logit[corr_ans] at the
prediction position. Per-family vectors, rank correlations, cumulative top-k,
atlas comparison."""
import json
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/successor')
from successor_lib import HERE, DEV, PRED_POS, load_model, load_stimuli, run, pairs_tensors

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_model()
NH, NL = cfg['n_head'], cfg['n_layer']
COMPS = [('h', li, h) for li in range(NL) for h in range(NH)] + \
        [('m', li) for li in range(NL)]
stim = load_stimuli()
ci, xi, ca, xa, rows = pairs_tensors(stim, split='analysis')
fams = [r['family'] for r in rows]
N = len(rows)
print(f'{N} analysis pairs', flush=True)


def margins(lg, ca, xa):
    n = lg.shape[0]
    return (lg[range(n), PRED_POS, ca] - lg[range(n), PRED_POS, xa])


BS = 8
m_clean, m_corr = [], []
caches = []   # per batch clean cache
for i in range(0, N, BS):
    lg_c, cache = run(m, cfg, ci[i:i + BS], collect=True)
    lg_x, _ = run(m, cfg, xi[i:i + BS])
    m_clean.append(margins(lg_c, ca[i:i + BS], xa[i:i + BS]))
    m_corr.append(margins(lg_x, ca[i:i + BS], xa[i:i + BS]))
    caches.append(cache)
m_clean = torch.cat(m_clean)
m_corr = torch.cat(m_corr)
den = (m_clean - m_corr)
print(f'denominator sum {den.sum().item():.1f}  (per-pair median {den.median().item():.2f})', flush=True)


def patched_margins(patch_comps):
    """Patch the listed components' full outputs from clean cache into the
    corrupted run; returns per-pair margins."""
    out = []
    for bi, i in enumerate(range(0, N, BS)):
        cache = caches[bi]
        ph, pm = {}, {}
        for c in patch_comps:
            if c[0] == 'h':
                ph[(c[1], c[2])] = cache[('h', c[1])][:, :, c[2], :]
            else:
                pm[c[1]] = cache[('m', c[1])]
        lg, _ = run(m, cfg, xi[i:i + BS], patch_head=ph or None,
                    patch_mlp=pm or None)
        out.append(margins(lg, ca[i:i + BS], xa[i:i + BS]))
    return torch.cat(out)


fam_mask = {f: torch.tensor([x == f for x in fams], device=DEV) for f in
            ['weekday', 'month', 'alphabet']}


def recov(m_p, mask=None):
    d, n = den, (m_p - m_corr)
    if mask is not None:
        d, n = d[mask], n[mask]
    return (n.sum() / d.sum()).item()


imp_all, imp_fam = {}, {f: {} for f in fam_mask}
for j, c in enumerate(COMPS):
    m_p = patched_margins([c])
    imp_all[c] = recov(m_p)
    for f in fam_mask:
        imp_fam[f][c] = recov(m_p, fam_mask[f])
    if j % 30 == 0:
        print(f'{j}/{len(COMPS)}', flush=True)

order = sorted(COMPS, key=lambda c: -imp_all[c])
top10 = [(str(c), round(imp_all[c], 4)) for c in order[:10]]
print('TOP-10 overall:', top10, flush=True)

cum = {}
for k in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]:
    m_p = patched_margins(order[:k])
    cum[k] = {'overall': round(recov(m_p), 4),
              **{f: round(recov(m_p, fam_mask[f]), 4) for f in fam_mask}}
    print(f'cumulative top-{k}: {cum[k]}', flush=True)

vec_all = np.array([imp_all[c] for c in COMPS])
vecs = {f: np.array([imp_fam[f][c] for c in COMPS]) for f in fam_mask}
pair_rho = {}
for a in fam_mask:
    for b in fam_mask:
        if a < b:
            r, p = spearmanr(vecs[a], vecs[b])
            pair_rho[f'{a}~{b}'] = {'spearman': round(float(r), 3),
                                    'p': float(p)}
# also on the top-20 union (where signal lives)
top20u = set()
for f in fam_mask:
    top20u |= set(sorted(range(len(COMPS)), key=lambda i: -vecs[f][i])[:20])
top20u = sorted(top20u)
pair_rho_top = {}
for a in fam_mask:
    for b in fam_mask:
        if a < b:
            r, _ = spearmanr(vecs[a][top20u], vecs[b][top20u])
            pair_rho_top[f'{a}~{b}'] = round(float(r), 3)
print('family rank correlations (all 180):', pair_rho, flush=True)
print('family rank correlations (top-20 union):', pair_rho_top, flush=True)
top5_fam = {f: [(str(COMPS[i]), round(float(vecs[f][i]), 4)) for i in
                sorted(range(len(COMPS)), key=lambda i: -vecs[f][i])[:5]]
            for f in fam_mask}
print('per-family top-5:', json.dumps(top5_fam, indent=1), flush=True)

# --- atlas comparison ---
atlas = json.load(open(f'{QK}/qk_circuit_atlas.json'))
atl_cmp = {}
for t in ['capital', 'funcword', 'induction', 'digit']:
    av = np.array([atlas['importance_matrix'][t][str(c)] for c in COMPS])
    r, p = spearmanr(vec_all, av)
    ours10 = set(str(c) for c in order[:10])
    theirs10 = set(sorted(atlas['importance_matrix'][t],
                          key=lambda k: -atlas['importance_matrix'][t][k])[:10])
    atl_cmp[t] = {'spearman_vs_successor': round(float(r), 3), 'p': float(p),
                  'top10_overlap': sorted(ours10 & theirs10)}
print('atlas comparison:', json.dumps(atl_cmp, indent=1), flush=True)

res = {'n_analysis': N,
       'denominator_median': round(den.median().item(), 3),
       'importance_overall': {str(c): round(imp_all[c], 5) for c in COMPS},
       'importance_per_family': {f: {str(c): round(imp_fam[f][c], 5)
                                     for c in COMPS} for f in fam_mask},
       'top10_overall': top10, 'top5_per_family': top5_fam,
       'cumulative_topk': cum,
       'family_rank_corr_all': pair_rho,
       'family_rank_corr_top20union': pair_rho_top,
       'atlas_comparison': atl_cmp,
       'order': [str(c) for c in order]}
json.dump(res, open(f'{HERE}/patching.json', 'w'), indent=1)
print('saved patching.json', flush=True)
