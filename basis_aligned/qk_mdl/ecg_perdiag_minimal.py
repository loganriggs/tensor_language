"""ECG PER-DIAGNOSIS minimality (Logan Q): for each diagnosis, what is the minimal set of
features / part of the model that reproduces the SAME output? Two 'same output' notions:
  - ranking: explicit readout AUC_c >= 0.97 x model AUC_c  (reproduce who gets flagged)
  - value:   R^2 of the model's actual logit_c from the features (reproduce the score)
Feature bank = block-0 interaction features (interpretable) + Attn-2 pooled dirs (the two
co-equal parts, §38). Rank per code on TRAIN, fit on TRAIN, eval on TEST (no selection leak).
Also minimal LEADS per diagnosis (minimal part of the input). Necessity is separate (§34:
removal collapses nothing -> per-diagnosis necessary set is large/redundant).
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
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; CODES = ck['codes']; NCLS = len(CODES)
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
ib = torch.load(f'{QK}/ecg_interaction_basis.pt', map_location=DEV, weights_only=False)
A = ib['A'].to(DEV); R = ib['rank']
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
Xtr = torch.from_numpy(np.load(f'{OUT}/ecg_X_train.npy')).to(DEV)
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)


def patch(xn):
    B = xn.shape[0]
    return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def forward(x, keep_leads=None):
    xn = (x - MU) / SD
    if keep_leads is not None:
        m = torch.zeros(NLEAD, device=DEV); m[list(keep_leads)] = 1.0
        xn = xn * m[None, :, None]
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    a2 = None
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        ac = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T
        h = h + ac
        if li == 2: a2 = ac.mean(1)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        h = h + ((hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias'], a2


@torch.no_grad()
def model_logits(x, keep_leads=None):
    ls, a2 = [], []
    for i in range(0, len(x), 2048):
        l, a = forward(x[i:i+2048], keep_leads); ls.append(l.float()); a2.append(a.float())
    return torch.cat(ls), torch.cat(a2)


def auc1(vec, yc):
    lab = yc.bool(); p = lab.sum().float(); n = (~lab).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(vec)).float() + 1
    return float((r[lab].sum()-p*(p+1)/2)/(p*n))


Ltr, A2tr = model_logits(Xtr); Lte, A2te = model_logits(Xte)
ref = {c: auc1(Lte[:, c], Yte[:, c]) for c in range(NCLS)}
capable = [c for c in range(NCLS) if ref[c] >= 0.75 and int(Yte[:, c].sum()) >= 10]

# feature bank: block0 interaction (mean+max) + Attn-2 pooled dirs
def b0(tag, n):
    rs = torch.from_numpy(np.load(f'{QK}/ecg_readspace_{tag}.npy')).to(DEV)
    f = ((rs @ Gsqrt_inv.T) @ Ahat).pow(2).reshape(n, NP, R)
    return torch.cat([f.mean(1), f.amax(1)], 1)
Ftr = torch.cat([b0('train', Ytr.shape[0]), A2tr], 1)
Fte = torch.cat([b0('test', Yte.shape[0]), A2te], 1)
mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True).clamp_min(1e-6)
Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
NF = Ftr.shape[1]
def label_feat(j):
    if j < R: return f'wave#{j}(mean)'
    if j < 2*R: return f'wave#{j-R}(peak)'
    return f'attn2_dir{j-2*R}'

def fit_code(cols, c):
    lin = nn.Linear(len(cols), 1).to(DEV)
    opt = torch.optim.Adam(lin.parameters(), lr=8e-3, weight_decay=1e-3)
    yc = Ytr[:, c:c+1]
    for _ in range(1500):
        bi = torch.randint(0, len(Ftr), (1024,), device=DEV)
        loss = F.binary_cross_entropy_with_logits(lin(Ftr[bi][:, cols]), yc[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s = lin(Fte[:, cols]).squeeze(1)
    a = auc1(s, Yte[:, c])
    # logit-R^2 vs model logit for c
    t = Lte[:, c]; r2 = 1 - (s - (s.mean() + (t - t.mean()) * 0)).pow(2).sum().item()*0  # placeholder
    return a, s

# per-diagnosis: rank features by TRAIN discriminativeness, grow set to reach ranking target
Ks = [1, 2, 3, 5, 8, 12, 20, 32]
perdiag = {}
for c in capable:
    dtr = np.array([abs(auc1(Ftr[:, j], Ytr[:, c]) - 0.5) for j in range(NF)])
    order = list(np.argsort(-dtr))
    target = 0.97 * ref[c]
    kmin = None; curve = []; s_best = None
    for K in Ks:
        a, s = fit_code(order[:K], c)
        curve.append(round(a, 3))
        if a >= target and kmin is None:
            kmin = K; s_best = s
        if a >= 0.99 * ref[c]:
            break
    if s_best is None:
        s_best = s
    # logit R^2 at the chosen (or max) K
    t = Lte[:, c].float(); pr = s_best.float()
    # best affine fit of pr to t
    A_ = torch.stack([pr, torch.ones_like(pr)], 1)
    coef = torch.linalg.lstsq(A_, t.unsqueeze(1)).solution
    fit = (A_ @ coef).squeeze(1)
    r2 = 1 - (t - fit).pow(2).sum().item() / (t - t.mean()).pow(2).sum().item()
    topf = [label_feat(order[0])]
    perdiag[CODES[c]] = {'model_auc': round(ref[c], 3), 'min_features_rank97': kmin,
                         'auc_curve': dict(zip([str(k) for k in Ks[:len(curve)]], curve)),
                         'single_feature_auc': curve[0], 'logit_R2_at_min': round(r2, 3),
                         'top_feature': topf[0]}

# per-diagnosis minimal LEADS: rank leads by single-lead occlusion drop, keep-top-k
single_drop = {}
for L in range(NLEAD):
    lg, _ = model_logits(Xte, keep_leads=[x for x in range(NLEAD) if x != L])  # occlude L
    for c in capable:
        single_drop.setdefault(c, {})[L] = ref[c] - auc1(lg[:, c], Yte[:, c])
for c in capable:
    lead_rank = sorted(range(NLEAD), key=lambda L: -single_drop[c][L])
    kmin_leads = None
    for k in range(1, 7):
        lg, _ = model_logits(Xte, keep_leads=lead_rank[:k])
        a = auc1(lg[:, c], Yte[:, c])
        if a >= 0.95 * ref[c]:
            kmin_leads = k; break
    perdiag[CODES[c]]['min_leads_95'] = kmin_leads
    perdiag[CODES[c]]['top_leads'] = [LEADS[L] for L in lead_rank[:3]]

kf = [v['min_features_rank97'] for v in perdiag.values() if v['min_features_rank97']]
kl = [v['min_leads_95'] for v in perdiag.values() if v['min_leads_95']]
res = {'n_capable': len(capable), 'ranking_target': '0.97 x model per-code AUC',
       'min_features_median': int(np.median(kf)) if kf else None,
       'min_features_dist': {str(k): sum(1 for v in perdiag.values() if v['min_features_rank97'] == k) for k in Ks},
       'codes_single_feature_suffices': [c for c in perdiag if perdiag[c]['min_features_rank97'] == 1],
       'min_leads_median': int(np.median(kl)) if kl else None,
       'mean_logit_R2_at_min': round(float(np.mean([v['logit_R2_at_min'] for v in perdiag.values()])), 3),
       'per_code': perdiag}
json.dump(res, open(f'{QK}/ecg_perdiag_minimal.json', 'w'), indent=2)
print(json.dumps({k: res[k] for k in ('min_features_median', 'min_features_dist', 'codes_single_feature_suffices',
      'min_leads_median', 'mean_logit_R2_at_min')}, indent=1), flush=True)
for c in ['CLBBB', 'CRBBB', 'LAFB', 'INJAS', 'AMI', 'LVH', 'IMI', 'ISCIN']:
    if c in perdiag:
        p = perdiag[c]
        print(f"  {c}: model {p['model_auc']}, minFeat(rank) {p['min_features_rank97']}, "
              f"1-feat {p['single_feature_auc']}, logitR2 {p['logit_R2_at_min']}, minLeads {p['min_leads_95']} {p['top_leads']}", flush=True)
print('ECG PERDIAG MINIMAL DONE', flush=True)
