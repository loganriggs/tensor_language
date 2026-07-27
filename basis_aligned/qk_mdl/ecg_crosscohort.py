"""ECG STAGE 2: cross-country validation. Does the conduction-disturbance signal the
PTB-XL (German) model learned generalize to the Georgia (US) cohort? And do the
individual features separate into true cross-country ones vs PTB-XL-specific?

(A) Model transfer: PTB-XL-trained model's CD-superclass AUC on Georgia (cross-country).
(B) Per-feature CD discriminativeness on PTB-XL test vs Georgia; correlation = how much
    of the learned CD representation is country-independent.
(C) The validation loop: generalizing features (strong on both) vs PTB-XL-specific;
    a CD detector from each, evaluated on Georgia.
(D) The Stage-1 V1/QRS BBB units specifically: do they retain CD-discrimination in the
    US cohort -> is the V1/QRS feature a true cross-country biomarker?
"""
import sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
CD = 3
ck = torch.load(f'{QK}/ecg_model.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
norm = lambda x: (x - MU) / SD

# PTB-XL test (in-domain) and Georgia (cross-country)
Xp = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
Yp_cd = (torch.from_numpy(np.load(f'{OUT}/ecg_Y_test.npy'))[:, CD] == 1).to(DEV)
Xg = torch.from_numpy(np.load(f'{OUT}/georgia_X.npy')).to(DEV)
Yg_cd = (torch.from_numpy(np.load(f'{OUT}/georgia_yCD.npy')) == 1).to(DEV)


def patch(x):
    B = x.shape[0]
    return x.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def forward(x):
    h = patch(norm(x)) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        h = h + (((hn2 @ W[mw+'L.weight'].T)*(hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T)
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


A = W['embed.weight'].T @ W['blocks.1.L.weight'].T
Bm = W['embed.weight'].T @ W['blocks.1.R.weight'].T


@torch.no_grad()
def feats(X):
    o = []
    for i in range(0, len(X), 2048):
        P = patch(norm(X[i:i+2048]))
        o.append((torch.einsum('bnp,pj->bnj', P, A) * torch.einsum('bnp,pj->bnj', P, Bm)).mean(1))
    return torch.cat(o)


def auc(s, lab):
    o = torch.argsort(torch.argsort(s)); r = o.float()+1
    p = lab.sum().float(); n = (~lab).sum().float()
    return 0.5 if p == 0 or n == 0 else float((r[lab].sum()-p*(p+1)/2)/(p*n))


# (A) model transfer
with torch.no_grad():
    scp = torch.cat([forward(Xp[i:i+2048])[:, CD] for i in range(0, len(Xp), 2048)])
    scg = torch.cat([forward(Xg[i:i+2048])[:, CD] for i in range(0, len(Xg), 2048)])
res = {'model_CD_auc_ptbxl': round(auc(scp, Yp_cd), 4),
       'model_CD_auc_georgia': round(auc(scg, Yg_cd), 4),
       'georgia_n': int(len(Xg)), 'georgia_CD_prev': round(float(Yg_cd.float().mean()), 3)}

# (B) per-feature CD AUC on both cohorts
Fp, Fg = feats(Xp), feats(Xg)
ap = np.array([abs(auc(Fp[:, j], Yp_cd)-0.5) for j in range(INNER)])
ag = np.array([abs(auc(Fg[:, j], Yg_cd)-0.5) for j in range(INNER)])
res['feature_CDstrength_corr_ptbxl_vs_georgia'] = round(float(np.corrcoef(ap, ag)[0, 1]), 3)

# (C) validation loop: select by PTB-XL strength vs by generalization (min of both)
def fit_eval(units):
    cols = torch.tensor(units, device=DEV)
    head = nn.Linear(len(units), 1).to(DEV)
    Xf = Fp[:, cols]; mu, sd = Xf.mean(0, keepdim=True), Xf.std(0, keepdim=True).clamp_min(1e-6)
    yb = Yp_cd.float()
    opt = torch.optim.Adam(head.parameters(), 1e-2)
    for s in range(1500):
        bi = torch.randint(0, len(Xf), (2048,), device=DEV)
        l = F.binary_cross_entropy_with_logits(head(((Xf[bi]-mu)/sd)).squeeze(1), yb[bi])
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        return round(auc(head(((Fg[:, cols]-mu)/sd)).squeeze(1), Yg_cd), 4)   # eval on Georgia
K = 16
sel_strength = list(np.argsort(-ap)[:K])
sel_general = list(np.argsort(-np.minimum(ap, ag))[:K])
res['georgia_detector_from_ptbxl_strong_features'] = fit_eval(sel_strength)
res['georgia_detector_from_generalizing_features'] = fit_eval(sel_general)
res['feature_overlap'] = len(set(sel_strength) & set(sel_general))

# (D) the Stage-1 V1/QRS units: retained cross-country?
try:
    s1 = json.load(open(f'{QK}/ecg_analyze.json'))['fold_top_units']
    res['stage1_units_ptbxl_CDstrength'] = [round(float(ap[j]), 3) for j in s1]
    res['stage1_units_georgia_CDstrength'] = [round(float(ag[j]), 3) for j in s1]
except Exception as e:
    res['stage1_units_error'] = str(e)

print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/ecg_crosscohort.json', 'w'), indent=2)
print('ECG CROSSCOHORT DONE', flush=True)
