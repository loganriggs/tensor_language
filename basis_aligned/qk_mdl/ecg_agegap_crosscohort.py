"""Tier-2 cross-cohort validation: does the mortality-direction signal (pathology reads
prematurely OLD) generalize beyond PTB-XL? The age-gap student predicts the premature-aging gap
from the ECG alone (no true age needed at inference), so run it on the independent US (Georgia)
and China (Chapman) cohorts and compare predicted gap for conduction-disease (CD) vs NORMAL.
If pathology reads prematurely-older abroad too, the biomarker signal is corpus-general, not a
PTB-XL artifact (the program's cross-cohort truth filter).
"""
import json
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
ck = torch.load(f'{QK}/ecg_agegap_student_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; gmu, gsd = ck['gmu'], ck['gsd']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)


def patch(xn):
    B = xn.shape[0]; return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def gap(x):
    xn = (x - MU) / SD
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2'); v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        h = h + ((hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T
    return (F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']).squeeze(1) * gsd + gmu


def auc(score, y):
    y = y.astype(bool); p = y.sum(); n = (~y).sum()
    if p == 0 or n == 0: return 0.5
    o = np.argsort(np.argsort(score)) + 1
    return float((o[y].sum() - p*(p+1)/2) / (p*n))


res = {'reference_ptbxl_pathology_minus_normal_gap': 3.03}
for coh in ['georgia', 'chapman']:
    X = torch.from_numpy(np.load(f'{OUT}/{coh}_X.npy')).to(DEV)
    G = torch.cat([gap(X[i:i+2048]) for i in range(0, len(X), 2048)]).cpu().numpy()
    cd = np.load(f'{OUT}/{coh}_yCD.npy').astype(bool); nm = np.load(f'{OUT}/{coh}_yNORM.npy').astype(bool)
    gcd, gnm = G[cd].mean(), G[nm].mean()
    # discrimination: does the predicted gap separate CD from NORM?
    mask = cd | nm; y = cd[mask]
    res[coh] = {'n_CD': int(cd.sum()), 'n_NORM': int(nm.sum()),
                'mean_gap_CD': round(float(gcd), 2), 'mean_gap_NORM': round(float(gnm), 2),
                'CD_minus_NORM': round(float(gcd - gnm), 2),
                'gap_discriminates_CD_vs_NORM_auc': round(auc(G[mask], y), 3)}
    print(f"  {coh}: gap CD {gcd:.2f} vs NORM {gnm:.2f} (diff {gcd-gnm:+.2f}) | CD-vs-NORM AUC {res[coh]['gap_discriminates_CD_vs_NORM_auc']}", flush=True)

res['generalizes'] = bool(res['georgia']['CD_minus_NORM'] > 0.5 and res['chapman']['CD_minus_NORM'] > 0.5)
json.dump(res, open(f'{QK}/ecg_agegap_crosscohort.json', 'w'), indent=2)
print(json.dumps(res, indent=1), flush=True)
print('ECG AGEGAP CROSSCOHORT DONE', flush=True)
