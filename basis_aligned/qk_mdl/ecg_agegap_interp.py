"""Tier-2 payoff: interpret the foldable AGE-GAP student — WHAT makes an ECG read prematurely old
(the mortality-linked signal). (1) Which conditions/diagnoses drive the highest predicted age-gap
(the mortality mechanism). (2) Which known clinical measures correlate with the gap. (3) Which leads.
This is a decomposable account of a mortality-direction biomarker.
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
Lx = {n: i for i, n in enumerate(LEADS)}
ck = torch.load(f'{QK}/ecg_agegap_student_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; gmu, gsd = ck['gmu'], ck['gsd']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
FS = 100


def patch(xn):
    B = xn.shape[0]; return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def gap_pred(x, occ_lead=None):
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
    return (F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']).squeeze(1) * gsd + gmu


GAP = torch.cat([gap_pred(Xte[i:i+2048]) for i in range(0, len(Xte), 2048)]).cpu().numpy()

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
agg = pd.read_csv(f'{OUT}/scp_statements.csv', index_col=0)
fold = df.strat_fold.values
te_codes = df.scp_codes.values[fold == 10]
true_age = df['age'].values.astype(np.float32)[fold == 10]
valid = (true_age >= 18) & (true_age <= 89)

# (1) per diagnostic SUPERCLASS mean predicted age-gap (which conditions age the ECG most)
diag2super = {c: agg.loc[c, 'diagnostic_class'] for c in agg.index if agg.loc[c, 'diagnostic'] == 1}
supers = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
super_gap = {}
for s in supers:
    mask = np.array([any(diag2super.get(c) == s for c in cc) for cc in te_codes]) & valid
    if mask.sum() >= 10:
        super_gap[s] = {'n': int(mask.sum()), 'mean_gap': round(float(GAP[mask].mean()), 2)}
# per specific capable code
codegap = {}
for c in set(x for cc in te_codes for x in cc):
    mask = np.array([c in cc for cc in te_codes]) & valid
    if mask.sum() >= 20:
        codegap[c] = round(float(GAP[mask].mean()), 2)
top_old = sorted(codegap.items(), key=lambda kv: -kv[1])[:8]
top_young = sorted(codegap.items(), key=lambda kv: kv[1])[:5]

# (2) clinical measures vs gap
Xn = ((Xte - MU) / SD).cpu().numpy()
FEATS = ['HR', 'HRV', 'QRSwidth', 'QT', 'R_precordial', 'ST_ant', 'T_amp']
Fc = np.zeros((len(Xn), len(FEATS)), np.float32)
for i in range(len(Xn)):
    sig = Xn[i, 1]; pk, _ = find_peaks(np.abs(sig), distance=40, height=np.percentile(np.abs(sig), 90))
    hr, rrcv = (60/(np.diff(pk)/FS).mean(), (np.diff(pk)/FS).std()/((np.diff(pk)/FS).mean()+1e-9)) if len(pk) >= 3 else (60., 0.)
    beats = [Xn[i, :, p-50:p+50] for p in pk if p-50 >= 0 and p+50 < 1000]
    b = np.median(np.stack(beats), 0) if beats else Xn[i, :, 450:550]
    base = b[:, 25:40].mean(1); R = np.maximum(b[:, 42:58].max(1)-base, 0)
    dev = np.abs(b[:, 40:72]-base[:, None]); qrs = (dev > 0.25*dev.max(1, keepdims=True)).sum(1).mean()
    treg = np.abs(b[Lx['II'], 60:95]-base[Lx['II']]); tend = 60 + (np.where(treg > 0.15*treg.max())[0].max() if treg.max() > 0 else 30)
    Fc[i] = [hr, rrcv, qrs, (tend-50)/FS*1000, R[[Lx['V5'], Lx['V6']]].mean(),
             (b[[Lx['V1'], Lx['V2'], Lx['V3']], 64:70].mean()-base[[Lx['V1'], Lx['V2'], Lx['V3']]].mean()),
             (b[:, 80:92].mean(1)-base).mean()]
corrs = {FEATS[j]: round(float(np.corrcoef(Fc[valid, j], GAP[valid])[0, 1]), 3) for j in range(len(FEATS))}

# (3) leads
lead_imp = {}
for Li in range(NLEAD):
    g = torch.cat([gap_pred(Xte[i:i+2048], occ_lead=Li) for i in range(0, len(Xte), 2048)]).cpu().numpy()
    lead_imp[LEADS[Li]] = round(float(np.mean(np.abs(g-GAP))), 2)

res = {'superclass_mean_agegap': dict(sorted(super_gap.items(), key=lambda kv: -kv[1]['mean_gap'])),
       'codes_read_most_prematurely_old': top_old, 'codes_read_youngest': top_young,
       'clinical_corrs_with_gap': dict(sorted(corrs.items(), key=lambda kv: -abs(kv[1]))),
       'lead_importance': dict(sorted(lead_imp.items(), key=lambda kv: -kv[1]))}
json.dump(res, open(f'{QK}/ecg_agegap_interp.json', 'w'), indent=2)
print('mean age-gap by superclass:', json.dumps(res['superclass_mean_agegap']), flush=True)
print('conditions read most prematurely OLD:', top_old, flush=True)
print('clinical corrs with gap:', json.dumps(res['clinical_corrs_with_gap']), flush=True)
print('top leads:', list(res['lead_importance'].items())[:4], flush=True)
print('ECG AGEGAP INTERP DONE', flush=True)
