"""ECG STAGE 2b: THREE-CONTINENT validation. Does the conduction-disturbance feature
the German (PTB-XL) model learned hold across the US (Georgia) AND China (Chapman)?
A feature surviving three continents/equipment sets is a strong biomarker candidate.
"""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
CD = 3
ck = torch.load(f'{QK}/ecg_model.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
norm = lambda x: (x - MU) / SD

Xp = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
Yp = (torch.from_numpy(np.load(f'{OUT}/ecg_Y_test.npy'))[:, CD] == 1).to(DEV)
Xg = torch.from_numpy(np.load(f'{OUT}/georgia_X.npy')).to(DEV)
Yg = (torch.from_numpy(np.load(f'{OUT}/georgia_yCD.npy')) == 1).to(DEV)
Xc = torch.from_numpy(np.load(f'{OUT}/chapman_X.npy')).to(DEV)
Yc = (torch.from_numpy(np.load(f'{OUT}/chapman_yCD.npy')) == 1).to(DEV)


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


res = {}
for name, X, Y in (('ptbxl_DE', Xp, Yp), ('georgia_US', Xg, Yg), ('chapman_CN', Xc, Yc)):
    with torch.no_grad():
        sc = torch.cat([forward(X[i:i+2048])[:, CD] for i in range(0, len(X), 2048)])
    res[f'model_CD_auc_{name}'] = round(auc(sc, Y), 4)
    res[f'{name}_n'] = int(len(X)); res[f'{name}_CD_prev'] = round(float(Y.float().mean()), 3)

Fp, Fg, Fc = feats(Xp), feats(Xg), feats(Xc)
sp = np.array([abs(auc(Fp[:, j], Yp)-0.5) for j in range(INNER)])
sg = np.array([abs(auc(Fg[:, j], Yg)-0.5) for j in range(INNER)])
sc = np.array([abs(auc(Fc[:, j], Yc)-0.5) for j in range(INNER)])
res['feature_corr_DE_US'] = round(float(np.corrcoef(sp, sg)[0, 1]), 3)
res['feature_corr_DE_CN'] = round(float(np.corrcoef(sp, sc)[0, 1]), 3)
res['feature_corr_US_CN'] = round(float(np.corrcoef(sg, sc)[0, 1]), 3)
# features surviving ALL THREE (top by min across cohorts)
tri = np.minimum(np.minimum(sp, sg), sc)
top3 = list(np.argsort(-tri)[:8])
res['three_continent_units'] = [int(u) for u in top3]
res['three_continent_strengths'] = [[round(float(sp[j]), 3), round(float(sg[j]), 3), round(float(sc[j]), 3)] for j in top3]
# Stage-1 V1/QRS units across all three
try:
    s1 = json.load(open(f'{QK}/ecg_analyze.json'))['fold_top_units']
    res['stage1_units_DE'] = [round(float(sp[j]), 3) for j in s1]
    res['stage1_units_US'] = [round(float(sg[j]), 3) for j in s1]
    res['stage1_units_CN'] = [round(float(sc[j]), 3) for j in s1]
except Exception as e:
    res['stage1_err'] = str(e)
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/ecg_crosscohort3.json', 'w'), indent=2)
print('ECG CROSSCOHORT3 DONE', flush=True)
