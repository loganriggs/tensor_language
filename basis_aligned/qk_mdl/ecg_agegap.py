"""ECG age-gap as a disease proxy (answering Logan: can we find a mortality-linked
signal WITHOUT mortality labels?). The ECG age-gap (predicted age - actual age) is a
validated biomarker of cardiovascular aging / mortality. We compute it on PTB-XL and
test whether it tracks DISEASE using labels we already have: do pathological ECGs
look OLDER than the patient's true age, controlling for actual age?
"""
import sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
SUP = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
ck = torch.load(f'{QK}/ecg_age_model.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
norm = lambda x: (x - MU) / SD
# the age model was trained standardizing the target; recover AMU/ASD from train ages
df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); fold = df.strat_fold.values
tr_age = df.age.values[fold <= 8].astype(np.float32); tr_age = tr_age[(tr_age > 0) & (tr_age <= 95)]
AMU, ASD = float(tr_age.mean()), float(tr_age.std())

Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
Yte = np.load(f'{OUT}/ecg_Y_test.npy')                       # (N,5) superclass multihot
age = df.age.values[fold == 10].astype(np.float32)
valid = (age > 0) & (age <= 95)


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
    return (F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']).squeeze(1)


with torch.no_grad():
    pred = torch.cat([forward(Xte[i:i+2048]) for i in range(0, len(Xte), 2048)]).cpu().numpy()
pred = pred * ASD + AMU
gap = pred - age                                             # ECG age-gap
g, a, Y = gap[valid], age[valid], Yte[valid]

# control for actual age: residualize gap on a quadratic in actual age
Ade = np.stack([np.ones_like(a), a, a**2], 1)
beta = np.linalg.lstsq(Ade, g, rcond=None)[0]
gap_adj = g - Ade @ beta                                     # age-controlled gap

is_norm = (Y[:, 0] == 1) & (Y.sum(1) == 1)                   # pure normal
has_path = Y[:, 1:].max(1) > 0                                # any of MI/STTC/CD/HYP
res = {'n': int(valid.sum()), 'raw_gap_mean': round(float(g.mean()), 2),
       'agecontrolled_gap_pureNORM': round(float(gap_adj[is_norm].mean()), 3),
       'agecontrolled_gap_hasPathology': round(float(gap_adj[has_path].mean()), 3)}
# per superclass (age-controlled mean gap)
for k, name in enumerate(SUP):
    m = Y[:, k] == 1
    res[f'gap_{name}'] = round(float(gap_adj[m].mean()), 3)
# disease burden: correlation of age-controlled gap with number of pathology labels
nburden = Y[:, 1:].sum(1)
res['gap_vs_disease_burden_r'] = round(float(np.corrcoef(gap_adj, nburden)[0, 1]), 3)
# effect size NORM vs pathology (Cohen's d)
d = (gap_adj[has_path].mean() - gap_adj[is_norm].mean()) / gap_adj.std()
res['cohens_d_path_vs_norm'] = round(float(d), 3)
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/ecg_agegap.json', 'w'), indent=2)
print('ECG AGEGAP DONE', flush=True)
