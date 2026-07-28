"""ECG circuit minimality + interpretability METRICS (Logan Q: how minimal / interpretable,
with a metric). Separates three notions and scores each:
  MINIMALITY (sufficiency, MDL frame): fit an EXPLICIT standalone readout on the top-K
    interaction features (feature act -> logistic -> 28 codes), sweep K, report behavioral
    RETENTION vs the full model. K at 90/95/99% retention = the description length.
    Random-K control shows the ranked features carry it.
  MINIMALITY (necessity): from §34 (no collapse removing 10) -> report as unbounded/redundant.
  INTERPRETABILITY: (a) grounding = physiological top-lead match to textbook diagnostic leads;
    (b) morphology-specificity (real vs scrambled, §36); (c) selectivity = codes served per
    feature (monosemantic vs polysemantic); (d) the explicit model's retention IS the
    faithfulness of the human-readable description.
"""
import ast, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
KNOWN = {'CRBBB': {'V1', 'V2'}, 'CLBBB': {'V1', 'V6', 'I'}, 'IRBBB': {'V1', 'V2'},
         'LAFB': {'I', 'aVL', 'III', 'aVF'}, 'LPFB': {'III', 'aVF', 'I'},
         'INJAS': {'V1', 'V2', 'V3'}, 'INJAL': {'I', 'aVL', 'V5', 'V6'},
         'IMI': {'II', 'III', 'aVF'}, 'AMI': {'V1', 'V2', 'V3', 'V4'},
         'ISCIN': {'II', 'III', 'aVF'}, 'LVH': {'V5', 'V6', 'I'}, 'ILMI': {'II', 'III', 'aVF'}}
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; CODES = ck['codes']; NCLS = len(CODES)
D, NP = cfg['D'], cfg['NP']
ib = torch.load(f'{QK}/ecg_interaction_basis.pt', map_location=DEV, weights_only=False)
A = ib['A'].to(DEV); R = ib['rank']; fc = ib['feat_code_auc']
Ahat = A / A.norm(dim=0, keepdim=True).clamp_min(1e-8)
fb = torch.load(f'{QK}/ecg_fold_block0.pt', map_location=DEV, weights_only=False)
Gsqrt_inv = torch.linalg.inv(fb['Gsqrt'].to(DEV))

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
def labels(mask):
    Y = np.zeros((mask.sum(), NCLS), dtype=np.float32)
    for i, cc in enumerate(df.scp_codes.values[mask]):
        for j, c in enumerate(CODES):
            if c in cc: Y[i, j] = 1.0
    return torch.from_numpy(Y).to(DEV)
Ytr, Yte = labels(fold <= 8), labels(fold == 10)

def feats(tag, n_ecg):
    rs = torch.from_numpy(np.load(f'{QK}/ecg_readspace_{tag}.npy')).to(DEV)     # (n_ecg*NP, D) whitened
    hn = rs @ Gsqrt_inv.T
    f = (hn @ Ahat).pow(2).reshape(n_ecg, NP, R)
    return torch.cat([f.mean(1), f.amax(1)], 1)                                 # (n_ecg, 2R) mean+max pool
Ftr = feats('train', Ytr.shape[0]); Fte = feats('test', Yte.shape[0])
# standardize
mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True).clamp_min(1e-6)
Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd

def auc_col(s, y, c):
    lab = y[:, c].bool(); p = lab.sum().float(); n = (~lab).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(s[:, c])).float() + 1
    return float((r[lab].sum()-p*(p+1)/2)/(p*n))
capable = [c for c in range(NCLS) if int(Yte[:, c].sum()) >= 10 and
           ({d['code']: d['auc'] for d in json.load(open(f'{QK}/ecg_codes_train.json'))['per_code']}.get(CODES[c], 0) >= 0.75)]
MODEL_MACRO = 0.925

def fit_explicit(feat_idx):
    """feat_idx into the R features; use both mean(idx) and max(R+idx) pooled columns."""
    cols = feat_idx + [R + i for i in feat_idx]
    Xtr, Xte = Ftr[:, cols], Fte[:, cols]
    lin = nn.Linear(len(cols), NCLS).to(DEV)
    opt = torch.optim.Adam(lin.parameters(), lr=5e-3, weight_decay=1e-3)
    for _ in range(3000):
        bi = torch.randint(0, len(Xtr), (1024,), device=DEV)
        loss = F.binary_cross_entropy_with_logits(lin(Xtr[bi]), Ytr[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s = lin(Xte)
    return float(np.mean([auc_col(s, Yte, c) for c in capable]))

# rank features by discriminativeness (max over codes of |AUC-0.5|)
disc = np.abs(fc - 0.5).max(1)
order = list(np.argsort(-disc))
Ks = [1, 2, 4, 8, 16, 32, 48, 64]
mdl = {}
for K in Ks:
    macro = fit_explicit(order[:K])
    mdl[K] = {'explicit_macro_auc': round(macro, 3), 'retention': round(macro / MODEL_MACRO, 3)}
    print(f'  K={K}: explicit macro {macro:.3f}, retention {macro/MODEL_MACRO:.3f}', flush=True)
# random-K control at K=16
rng = np.random.default_rng(0)
rand16 = fit_explicit(list(rng.choice(R, 16, replace=False)))
def kth(th):
    for K in Ks:
        if mdl[K]['retention'] >= th: return K
    return None

# ---- interpretability metrics ----
# grounding: top-lead of each feature vs textbook, for features that best-serve a KNOWN code
viz = json.load(open(f'{QK}/ecg_viz_data.json'))
phys_hit = 0; phys_tot = 0
for r in range(R):
    bestc = int(np.argmax(fc[:, r])) if False else int(np.argmax(fc[r]))
    code = CODES[bestc]
    if code in KNOWN and fc[r, bestc] >= 0.75:
        tl = set(viz['features'].get(str(r), {}).get('top_leads', []))
        phys_hit += len(tl & KNOWN[code]) > 0; phys_tot += 1
# selectivity: codes served per feature (AUC>=0.72), among features that serve >=1
served_counts = [(fc[r] >= 0.72).sum() for r in range(R)]
active = [n for n in served_counts if n >= 1]
mono = sum(1 for n in active if n == 1)

metrics = {
 'MINIMALITY_sufficiency_MDL': {
   'description': 'explicit K-feature standalone readout, behavioral retention vs full model',
   'K_at_90pct': kth(0.90), 'K_at_95pct': kth(0.95), 'K_at_99pct': kth(0.99),
   'frontier': mdl, 'random_16feat_retention': round(rand16 / MODEL_MACRO, 3),
   'block0_behavioral_rank_from_splice': '32-64 of 192 (RESULTS §28)'},
 'MINIMALITY_necessity': {'min_features_to_collapse_any_code': None,
   'note': 'no code collapses removing top-10 residual dirs (§34) -> necessity circuit is NOT minimal (redundant)'},
 'INTERPRETABILITY': {
   'grounding_physiology_leadmatch': f'{phys_hit}/{phys_tot}',
   'morphology_specific_real_vs_scrambled': '10/11 (§36)',
   'selectivity_mean_codes_per_active_feature': round(float(np.mean(active)), 2),
   'monosemantic_features_frac': round(mono / len(active), 2),
   'n_active_features': len(active),
   'faithfulness_is_the_retention': 'the explicit description reproduces X% of model behavior (see K_at_*)'},
}
json.dump(metrics, open(f'{QK}/ecg_circuit_metrics.json', 'w'), indent=2)
print(json.dumps(metrics, indent=1), flush=True)
print('ECG CIRCUIT METRICS DONE', flush=True)
