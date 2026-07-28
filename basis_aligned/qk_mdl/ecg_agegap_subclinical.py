"""Tier-2 sharpening: is the ECG-age biomarker SUBCLINICAL (detects aging in normal-looking ECGs)
or does it just re-read overt disease? Within PURE-NORMAL ECGs (only the NORM label, no diagnosis):
(1) does predicted ECG-age still track TRUE age (aging read in healthy hearts)? (2) is there
meaningful within-normal age-gap spread (subclinical variation)? (3) what characterizes high-gap
normals (older? faster HR? subtle measures)? Honest: no mortality labels here, so we CHARACTERIZE
the subclinical signal, not validate it against outcomes.
"""
import ast, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# age-gap student (predicts teacher age-gap) AND raw-age student (predicts ECG-age) for the age-tracking test
ckg = torch.load(f'{QK}/ecg_agegap_student_model.pt', map_location=DEV, weights_only=False)
cka = torch.load(f'{QK}/ecg_age_student_model.pt', map_location=DEV, weights_only=False)
cfg = ckg['cfg']; D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD']
MU, SD = ckg['MU'].to(DEV), ckg['SD'].to(DEV)
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)


def patch(xn):
    B = xn.shape[0]; return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def run(x, W, scale, shift):
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
    return (F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']).squeeze(1) * scale + shift


GAP = torch.cat([run(Xte[i:i+2048], ckg['state'], ckg['gsd'], ckg['gmu']) for i in range(0, len(Xte), 2048)]).cpu().numpy()
AGE = torch.cat([run(Xte[i:i+2048], cka['state'], cka['asd'], cka['amu']) for i in range(0, len(Xte), 2048)]).cpu().numpy()

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
agg = pd.read_csv(f'{OUT}/scp_statements.csv', index_col=0)
diag_codes = set(agg[agg.diagnostic == 1].index)
fold = df.strat_fold.values; true_age = df['age'].values.astype(np.float32)[fold == 10]
te_codes = df.scp_codes.values[fold == 10]
# pure-normal: has NORM and NO diagnostic code other than NORM
pure_norm = np.array([('NORM' in cc) and (len([c for c in cc if c in diag_codes and c != 'NORM']) == 0) for cc in te_codes])
valid = (true_age >= 18) & (true_age <= 89)
pn = pure_norm & valid

# heart rate within normals (subtle measure)
Xn = ((Xte - MU) / SD).cpu().numpy()
HR = np.full(len(Xn), 60.0)
for i in np.where(pn)[0]:
    sig = Xn[i, 1]; pk, _ = find_peaks(np.abs(sig), distance=40, height=np.percentile(np.abs(sig), 90))
    if len(pk) >= 3: HR[i] = 60/(np.diff(pk)/100).mean()


def corr(a, b): return round(float(np.corrcoef(a, b)[0, 1]), 3)


gpn = GAP[pn]; apn = AGE[pn]; tpn = true_age[pn]
# high-gap normals: gap > +5y (prematurely old despite looking normal)
himask = gpn > 5.0
res = {'n_pure_normal': int(pn.sum()),
       'ECGage_tracks_true_age_within_normals_corr': corr(apn, tpn),
       'MAE_ECGage_within_normals': round(float(np.mean(np.abs(apn - tpn))), 2),
       'within_normal_gap_mean': round(float(gpn.mean()), 2), 'within_normal_gap_std': round(float(gpn.std()), 2),
       'frac_normals_gap_over_5y': round(float(himask.mean()), 3),
       'gap_vs_trueage_corr_within_normals': corr(gpn, tpn),
       'gap_vs_HR_corr_within_normals': corr(gpn, HR[pn]),
       'high_gap_normals_mean_trueage': round(float(tpn[himask].mean()), 1) if himask.sum() else None,
       'low_gap_normals_mean_trueage': round(float(tpn[~himask].mean()), 1),
       'high_gap_normals_mean_HR': round(float(HR[pn][himask].mean()), 1) if himask.sum() else None,
       'low_gap_normals_mean_HR': round(float(HR[pn][~himask].mean()), 1),
       'pathology_vs_normal_gap_diff_ref_§53': 3.03}
json.dump(res, open(f'{QK}/ecg_agegap_subclinical.json', 'w'), indent=2)
print(json.dumps(res, indent=1), flush=True)
print('ECG AGEGAP SUBCLINICAL DONE', flush=True)
