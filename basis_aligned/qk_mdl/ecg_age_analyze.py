"""ECG PHASE D: discover WHAT the sex-from-ECG model uses, render it, and cross-cohort
validate (Germany PTB-XL -> US Georgia). Known physiology to check against: sex signal
lives in QRS amplitude (precordial leads, higher in men), QT interval and T-wave
morphology (longer QT in women). A rendered feature matching those = partial validation;
one that generalizes but ISN'T those = a discovery lead.
Lead order: I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6.
"""
import glob, sys, json
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
ck = torch.load(f'{QK}/ecg_age_model.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
norm = lambda x: (x - MU) / SD

# PTB-XL test + sex labels
df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); fold = df.strat_fold.values
Xp = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
ya = df.age.values[fold == 10].astype(np.float32)
vp = (ya > 0) & (ya <= 95)
yp = torch.from_numpy(ya).to(DEV)

# Georgia signals + sex from .hea (parse # Sex:)
Xg = torch.from_numpy(np.load(f'{OUT}/georgia_X.npy')).to(DEV)
recs = sorted(glob.glob(f'{OUT}/georgia/*/*.hea'))
gsex = []
for hp in recs:
    s = None
    for line in open(hp):
        if line.lower().startswith('# age') or line.lower().startswith('#age'):
            v = line.split(':', 1)[1].strip()
            try: s = float(v)
            except: s = None
    gsex.append(s if s is not None else -1)
gsex = np.array(gsex, dtype=np.float32)
# georgia_X was saved with a keep-mask; recs order matches pre-mask. Rebuild keep to align.
# (prep dropped only unreadable records; assume alignment holds for the kept set length)
gkeep = gsex >= 0
if len(gsex) != len(Xg):
    gsex = gsex[:len(Xg)]                       # defensive: align lengths
yg = torch.from_numpy(gsex).to(DEV)
gvalid = (yg > 0) & (yg <= 95)
Xg, yg = Xg[gvalid], yg[gvalid]


def patch(x):
    B = x.shape[0]
    return x.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def forward(x, occ_lead=None):
    xn = norm(x)
    if occ_lead is not None:
        xn = xn.clone(); xn[:, occ_lead] = 0.0
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        h = h + (((hn2 @ W[mw+'L.weight'].T)*(hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T)
    return (F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']).squeeze(1)


def auc(s, lab):
    # for continuous target: |Pearson r| between feature s and age lab (lab is float age here)
    s = s.float(); l = lab.float()
    r = torch.corrcoef(torch.stack([s, l]))[0, 1]
    return float(r.abs()) if not torch.isnan(r) else 0.0


# (A) cross-cohort model transfer for sex
Xp = Xp[torch.from_numpy(vp).to(DEV)]; yp = yp[torch.from_numpy(vp).to(DEV)]
with torch.no_grad():
    sp = torch.cat([forward(Xp[i:i+2048]) for i in range(0, len(Xp), 2048)])
    sg = torch.cat([forward(Xg[i:i+2048]) for i in range(0, len(Xg), 2048)])
res = {'age_r_ptbxl': round(auc(sp, yp), 4), 'age_r_georgia': round(auc(sg, yg), 4),
       'georgia_n': int(len(Xg))}

# (B) causal per-lead importance for sex (which leads encode it?)
with torch.no_grad():
    lead_imp = {}
    for L in range(NLEAD):
        s2 = torch.cat([forward(Xp[i:i+2048], occ_lead=L) for i in range(0, len(Xp), 2048)])
        lead_imp[LEADS[L]] = round(res['age_r_ptbxl'] - auc(s2, yp), 4)
res['per_lead_importance'] = lead_imp
res['top_leads'] = sorted(lead_imp, key=lambda k: -lead_imp[k])[:4]

# (C) feature-level cross-cohort generalization + rendering
A = W['embed.weight'].T @ W['blocks.1.L.weight'].T
Bm = W['embed.weight'].T @ W['blocks.1.R.weight'].T


@torch.no_grad()
def feats(X):
    o = []
    for i in range(0, len(X), 2048):
        P = patch(norm(X[i:i+2048]))
        o.append((torch.einsum('bnp,pj->bnj', P, A) * torch.einsum('bnp,pj->bnj', P, Bm)).mean(1))
    return torch.cat(o)


Fp, Fg = feats(Xp), feats(Xg)
ap = np.array([auc(Fp[:, j], yp) for j in range(INNER)])
ag = np.array([auc(Fg[:, j], yg) for j in range(INNER)])
res['feature_agestrength_corr_ptbxl_vs_georgia'] = round(float(np.corrcoef(ap, ag)[0, 1]), 3)
top_units = list(np.argsort(-np.minimum(ap, ag))[:6])       # top GENERALIZING sex features
res['top_generalizing_units'] = [int(u) for u in top_units]
res['top_units_ptbxl_strength'] = [round(float(ap[j]), 3) for j in top_units]
res['top_units_georgia_strength'] = [round(float(ag[j]), 3) for j in top_units]
waves = {}
for j in top_units:
    a_j, b_j = A[:, j], Bm[:, j]
    S = 0.5*(torch.outer(a_j, b_j)+torch.outer(b_j, a_j))
    ev, evec = torch.linalg.eigh(S)
    w = (evec[:, ev.abs().argmax()] * torch.sign(ev[ev.abs().argmax()])).reshape(NLEAD, PT).cpu()
    waves[int(j)] = w
    e = w.pow(2).sum(1).numpy()
    res.setdefault('unit_top_lead', {})[f'u{j}'] = LEADS[int(e.argmax())]
torch.save(waves, f'{QK}/ecg_age_waveforms.pt')
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/ecg_age_analyze.json', 'w'), indent=2)
print('ECG AGE ANALYZE DONE', flush=True)
