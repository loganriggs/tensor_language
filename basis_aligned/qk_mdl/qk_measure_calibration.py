"""TICK 209 (from the measure discussion with Logan): MEASURE CALIBRATION — which
candidate measure's residual best predicts causal damage, across the program's stored
intervention library?

Family A (9 points): layer-0 whole-head zeros, full-audit dCE (tick 192). Measures
already computed there (weights+unigram only): key-table Frobenius (identical across
heads — the uniform-weight measure), expected output magnitude ov_norm, third-moment
core scale, expected squared pattern. Spearman recomputed here for the record.

Family B (90 points, the powerful one): per-archetype key-channel ablations with
64-document per-position dCE (tick 190). For each ablation, computed now from weights
+ unigram: (1) PATTERN ENERGY removed E_{p x p}[(dP)^2] (sampled 4096^2 pairs);
(2) weight-Frobenius fraction of key tables removed (uniform measure); (3) p-weighted
Frobenius fraction removed (unigram measure); (4) the archetype's core mass lambda
(the mechanism-ledger measure). Spearman of each vs measured mean dCE, within heads
(partialing out head identity by rank-correlating pooled z-scores per head) and pooled.

Output: qk_measure_calibration.json + fig_measure_calibration.png (scatter panels).
Expected story, if the discussion's thesis holds: uniform-weight measures calibrate
worst, unigram pattern-energy best among cheap ones — quantifying "the measure is the
message" as a measurement instead of a slogan.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
from tier2_folding import branch_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
TAB = {}
for br, (qn, kn) in ((1, ('q1', 'k1')), (2, ('q2', 'k2'))):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()

mh_pt = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
mh_js = json.load(open(f'{QK}/qk_minimal_heads.json'))
polish = {0: torch.load(f'{QK}/qk_h0_polish_g025.pt', map_location=DEV),
          4: torch.load(f'{QK}/qk_h04_polish.pt', map_location=DEV)}
abl = json.load(open(f'{QK}/qk_arch_ablation.json'))
imp = json.load(open(f'{QK}/qk_head_importance.json'))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------- Family A (recomputed from stored) ----------
whole = [imp[f'h{h}_whole_dce'] for h in range(NH)]
famA = {}
for name in ('k1_norm', 'ov_norm', 'core_scale', 'pattern_sq', 'pattern_ov'):
    famA[name] = round(spearman(imp['correlates'][name]['values'], whole), 3)
print('Family A (9 whole-head zeros):', famA, flush=True)

# ---------- Family B (90 archetype ablations) ----------


def detectors(h, r):
    if h in (0, 4):
        bb = polish[h]
        D1 = bb[f'h{h}_k1_Dm'].to(DEV)
        D2 = bb[f'h{h}_k2_Dm'].to(DEV)
        D1 = D1 / D1.norm(dim=1, keepdim=True).clamp_min(1e-8)
        D2 = D2 / D2.norm(dim=1, keepdim=True).clamp_min(1e-8)
        g1 = D1.T @ bb[f'h{h}_AJ'][:, r].to(DEV)
        g2 = D2.T @ bb[f'h{h}_BJ'][:, r].to(DEV)
        lam = float(bb[f'h{h}_lamJ'][r])
    else:
        P = mh_pt[f'h{h}']
        Dn = P['Dm'].to(DEV)
        Dn = Dn / Dn.norm(dim=1, keepdim=True).clamp_min(1e-8)
        U = P['U'].to(DEV)
        g1 = Dn[:, :HD].T @ U[:, r]
        g2 = Dn[:, HD:2 * HD].T @ U[:, r]
        lam = float(P['lam'][r])
    return g1 / g1.norm().clamp_min(1e-12), g2 / g2.norm().clamp_min(1e-12), lam


@torch.no_grad()
def measures_for(h, g1, g2, n=4096, nb=4, seed=0):
    k1, k2 = TAB['k1'][:, h], TAB['k2'][:, h]
    q1, q2 = TAB['q1'][:, h], TAB['q2'][:, h]
    c1 = (k1 @ g1)
    c2 = (k2 @ g2)
    wfrac = float((c1 ** 2).sum() + (c2 ** 2).sum()) / float((k1 ** 2).sum() + (k2 ** 2).sum())
    pfrac = float((QP * c1 ** 2).sum() + (QP * c2 ** 2).sum()) / \
        float((QP * (k1 ** 2).sum(1)).sum() + (QP * (k2 ** 2).sum(1)).sum())
    g = torch.Generator().manual_seed(seed)
    tot = 0.0
    k1a = k1 - c1[:, None] * g1[None]
    k2a = k2 - c2[:, None] * g2[None]
    for _ in range(nb):
        si = torch.multinomial(QP.cpu(), n, replacement=True, generator=g).to(DEV)
        ti = torch.multinomial(QP.cpu(), n, replacement=True, generator=g).to(DEV)
        s1 = q1[si] @ k1[ti].T / HD
        s2 = q2[si] @ k2[ti].T / HD
        a1 = q1[si] @ k1a[ti].T / HD
        a2 = q2[si] @ k2a[ti].T / HD
        tot += float(((s1 * s2 - a1 * a2) ** 2).mean())
    return wfrac, pfrac, tot / nb


rows = []
for h in [1, 2, 3, 5, 6, 7, 8, 0, 4]:
    for rec in abl[f'h{h}']:
        r = rec['r']
        g1, g2, lam = detectors(h, r)
        wf, pf, pe = measures_for(h, g1, g2)
        rows.append({'h': h, 'r': r, 'dce': rec['mean_dce'], 'wfrac': wf,
                     'pfrac': pf, 'penergy': pe, 'lam': abs(lam)})
    print(f'h{h} measured', flush=True)
famB_pooled = {}
for name in ('wfrac', 'pfrac', 'penergy', 'lam'):
    famB_pooled[name] = round(spearman([x[name] for x in rows], [x['dce'] for x in rows]), 3)
# within-head (rank inside each head, then pool)
famB_within = {}
for name in ('wfrac', 'pfrac', 'penergy', 'lam'):
    ra, rd = [], []
    for h in range(NH):
        sub = [x for x in rows if x['h'] == h]
        if len(sub) < 3:
            continue
        va = np.argsort(np.argsort([x[name] for x in sub])).astype(float)
        vd = np.argsort(np.argsort([x['dce'] for x in sub])).astype(float)
        va = (va - va.mean()) / (va.std() + 1e-9)
        vd = (vd - vd.mean()) / (vd.std() + 1e-9)
        ra += va.tolist()
        rd += vd.tolist()
    famB_within[name] = round(float(np.corrcoef(ra, rd)[0, 1]), 3)
print('Family B pooled (90 archetype ablations):', famB_pooled, flush=True)
print('Family B within-head:', famB_within, flush=True)

out = {'familyA_wholehead': famA, 'familyB_pooled': famB_pooled,
       'familyB_within_head': famB_within,
       'rows': [{k: (round(v, 6) if isinstance(v, float) else v)
                 for k, v in x.items()} for x in rows]}
json.dump(out, open(f'{QK}/qk_measure_calibration.json', 'w'), indent=1)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
names = [('wfrac', 'uniform weight fraction'), ('pfrac', 'unigram-weighted fraction'),
         ('penergy', 'pattern energy removed'), ('lam', 'core mass (mechanism)')]
for ax, (name, label) in zip(axes, names):
    xs = [x[name] for x in rows]
    ys = [max(x['dce'], 1e-6) for x in rows]
    cs = [x['h'] for x in rows]
    ax.scatter(xs, ys, c=cs, cmap='tab10', s=18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(label)
    ax.set_ylabel('mean ΔCE (ablation)')
    ax.set_title(f'ρ pooled {famB_pooled[name]} · within-head {famB_within[name]}')
plt.tight_layout()
plt.savefig(f'{QK}/fig_measure_calibration.png', dpi=130)
print('MEASURE CALIBRATION DONE', flush=True)
