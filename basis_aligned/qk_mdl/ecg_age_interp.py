"""Tier-2 step 3: interpret the foldable ECG-age student — WHAT makes an ECG look older?
(1) Decompose student ECG-age into KNOWN clinical measures (heart rate, HRV, QRS width, QT proxy,
    R amplitude, ST, T, P) via regression -> R^2 = how much is 'known aging', residual = novel.
(2) Which single measures correlate most with predicted age.
(3) Lead occlusion -> which leads drive the age estimate.
(4) Age-GAP (student age - true age): does it track pathology (the mortality direction)?
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
L = {n: i for i, n in enumerate(LEADS)}
ck = torch.load(f'{QK}/ecg_age_student_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; amu, asd = ck['amu'], ck['asd']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
FS = 100


def patch(xn):
    B = xn.shape[0]; return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def age_pred(x, occ_lead=None):
    xn = (x - MU) / SD
    if occ_lead is not None:
        xn = xn.clone(); xn[:, occ_lead] = 0.0
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2'); v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        h = h + ((hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T
    return (F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']).squeeze(1) * asd + amu


PA = torch.cat([age_pred(Xte[i:i+2048]) for i in range(0, len(Xte), 2048)]).cpu().numpy()

# clinical features from full strip + median beat
Xn = ((Xte - MU) / SD).cpu().numpy()
FEATS = ['HR', 'HRV(RRCV)', 'QRSwidth', 'QT', 'R_precordial', 'ST', 'T_amp', 'P_amp', 'axisFront']
F_ = np.zeros((len(Xn), len(FEATS)), np.float32)
for i in range(len(Xn)):
    sig = Xn[i, 1]
    pk, _ = find_peaks(np.abs(sig), distance=40, height=np.percentile(np.abs(sig), 90))
    if len(pk) >= 3:
        rr = np.diff(pk)/FS; hr = 60/rr.mean(); rrcv = rr.std()/(rr.mean()+1e-9)
    else:
        hr, rrcv = 60.0, 0.0
    beats = [Xn[i, :, p-50:p+50] for p in pk if p-50 >= 0 and p+50 < 1000]
    b = np.median(np.stack(beats), 0) if beats else Xn[i, :, 450:550]   # (12,100)
    base = b[:, 25:40].mean(1)
    R = np.maximum(b[:, 42:58].max(1)-base, 0); Sd = np.maximum(base-b[:, 48:72].min(1), 0)
    dev = np.abs(b[:, 40:72]-base[:, None]); qrs = (dev > 0.25*dev.max(1, keepdims=True)).sum(1).mean()
    # QT proxy: R (sample 50) to T-end (last sample where |T-region| > 0.1*max in 60:95)
    treg = np.abs(b[L['II'], 60:95]-base[L['II']]); tend = 60 + (np.where(treg > 0.15*treg.max())[0].max() if treg.max() > 0 else 30)
    F_[i] = [hr, rrcv, qrs, (tend-50)/FS*1000, R[[L['V5'], L['V6']]].mean(),
             (b[:, 64:70].mean(1)-base).mean(), (b[:, 80:92].mean(1)-base).mean(),
             np.abs(b[L['II'], 30:44]-base[L['II']]).max(), (R[L['I']]-Sd[L['I']])]

# regress student ECG-age on clinical features
Xc = (F_ - F_.mean(0)) / (F_.std(0)+1e-6)
w, *_ = np.linalg.lstsq(np.c_[Xc, np.ones(len(Xc))], PA, rcond=None)
pred = np.c_[Xc, np.ones(len(Xc))] @ w
r2 = 1 - ((PA-pred)**2).sum()/((PA-PA.mean())**2).sum()
# single-feature correlations with predicted age
corrs = {FEATS[j]: round(float(np.corrcoef(F_[:, j], PA)[0, 1]), 3) for j in range(len(FEATS))}

# lead occlusion: mean |Δage| when each lead zeroed
base_age = PA
lead_imp = {}
for Li in range(NLEAD):
    pa = torch.cat([age_pred(Xte[i:i+2048], occ_lead=Li) for i in range(0, len(Xte), 2048)]).cpu().numpy()
    lead_imp[LEADS[Li]] = round(float(np.mean(np.abs(pa-base_age))), 2)

# age-gap vs pathology (mortality direction)
df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values; true_age = df['age'].values.astype(np.float32)[fold == 10]
valid = (true_age >= 18) & (true_age <= 89)
gap = PA[valid] - true_age[valid]
norm = np.array(['NORM' in cc for cc in df.scp_codes.values[fold == 10]])[valid]

res = {'clinical_R2_of_ECGage': round(float(r2), 3), 'novel_residual_frac': round(float(1-r2), 3),
       'single_feature_corrs': dict(sorted(corrs.items(), key=lambda kv: -abs(kv[1]))),
       'lead_importance_meanAbsDeltaYears': dict(sorted(lead_imp.items(), key=lambda kv: -kv[1])),
       'age_gap_normal': round(float(gap[norm].mean()), 2), 'age_gap_pathology': round(float(gap[~norm].mean()), 2),
       'age_gap_diff_path_minus_norm': round(float(gap[~norm].mean()-gap[norm].mean()), 2)}
json.dump(res, open(f'{QK}/ecg_age_interp.json', 'w'), indent=2)
print(f"ECG-age explained by known clinical measures: R2 {r2:.3f} (novel residual {1-r2:.3f})", flush=True)
print("top single-feature corrs:", json.dumps(res['single_feature_corrs'], indent=0), flush=True)
print("lead importance (mean |Δyears| when zeroed):", json.dumps(res['lead_importance_meanAbsDeltaYears'], indent=0), flush=True)
print(f"age-gap: pathology {res['age_gap_pathology']}y vs normal {res['age_gap_normal']}y (diff {res['age_gap_diff_path_minus_norm']})", flush=True)
print('ECG AGE INTERP DONE', flush=True)
