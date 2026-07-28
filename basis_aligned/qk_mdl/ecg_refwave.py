"""ECG reference waveforms + cross-continent feature validation (Logan). ONE model (PTB-XL/
Germany), ONE feature basis; apply to US (Georgia) + China (Chapman) raw ECGs. Deliver:
 (1) THE DIAGNOSTIC WAVEFORM: R-peak-aligned MEDIAN BEAT of confirmed-positive cases in the
     external cohorts (real morphology cardiologists see) -- for LBBB, RBBB (Georgia, specific
     SNOMED) and conduction-disturbance (Chapman).
 (2) REFERENCE-WAVEFORM MATCH: cosine of our Germany feature templates to those external median
     beats (best time-shift) -> is the shape the same abroad?
 (3) FEATURE TRANSFER: does each Germany feature's activation discriminate the diagnosis on the
     foreign cohort (external AUC)?
Exports ecg_refwave_data.json for rendering.
"""
import ast, glob, json
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
SNOMED = {'CLBBB': '164909002', 'CRBBB': '713427006', 'IRBBB': '713426002', 'LAFB': '445118002', 'NORM': '426783006'}
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; CODES = ck['codes']; NCLS = len(CODES)
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
ib = torch.load(f'{QK}/ecg_interaction_basis.pt', map_location=DEV, weights_only=False)
A = ib['A'].to(DEV); R = ib['rank']; fc = ib['feat_code_auc']
Ahat = A / A.norm(dim=0, keepdim=True).clamp_min(1e-8)
fb = torch.load(f'{QK}/ecg_fold_block0.pt', map_location=DEV, weights_only=False)


def patch(xn):
    B = xn.shape[0]
    return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def feats_pooled(X):
    xn = (X - MU) / SD
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    aw = 'blocks.0.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
    def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
    q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
    v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
    pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
    h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W['blocks.0.proj.weight'].T)
    hn0 = F.rms_norm(h, (D,))
    return (hn0 @ Ahat).pow(2).mean(1)                   # (B,R) pooled feature activations


def feats_all(X):
    return torch.cat([feats_pooled(X[i:i+2048]) for i in range(0, len(X), 2048)])


def auc(vec, lab):
    lab = lab.bool(); p = lab.sum().float(); n = (~lab).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(vec)).float() + 1
    return float((r[lab].sum()-p*(p+1)/2)/(p*n))


# ---- our Germany feature templates (from PTB-XL test), reuse rank-1 rendering ----
Xpt = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV); Xpt_n = (Xpt - MU) / SD
@torch.no_grad()
def hn0_of(xn):
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    aw = 'blocks.0.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
    def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
    q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
    v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
    pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
    h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W['blocks.0.proj.weight'].T)
    return F.rms_norm(h, (D,))
HN = torch.cat([hn0_of(Xpt_n[i:i+2048]) for i in range(0, len(Xpt_n), 2048)])
ACT = (HN @ Ahat).pow(2)
def template(r, k=300):
    flat = ACT[:, :, r].reshape(-1); kk = min(k, int((flat > 0).sum()))
    topi = torch.topk(flat, kk).indices; ex = topi // NP; pos = topi % NP
    T = torch.zeros(NLEAD, PT, device=DEV)
    for e, p in zip(ex.tolist(), pos.tolist()): T += Xpt_n[e, :, p*PT:(p+1)*PT]
    return (T / kk).cpu().numpy()

# ---- Georgia: specific SNOMED labels (sorted-glob order matches georgia_X) ----
recs = sorted(glob.glob(f'{OUT}/georgia/*/*.hea'))
dx = []
for hp in recs:
    codes = ''
    for line in open(hp):
        if line.lower().replace(' ', '').startswith('#dx'):
            codes = line.split(':', 1)[1]
    dx.append(codes)
Xg = torch.from_numpy(np.load(f'{OUT}/georgia_X.npy')).to(DEV)
assert len(dx) == len(Xg), (len(dx), len(Xg))
def glabel(code): return torch.tensor([1.0 if SNOMED[code] in d else 0.0 for d in dx], device=DEV)
# sanity: NORM count vs precomputed
gNORM = glabel('NORM'); print(f'sanity georgia NORM parsed {int(gNORM.sum())} vs precomputed {int(np.load(f"{OUT}/georgia_yNORM.npy").sum())}', flush=True)
Xc = torch.from_numpy(np.load(f'{OUT}/chapman_X.npy')).to(DEV)
cCD = torch.from_numpy(np.load(f'{OUT}/chapman_yCD.npy')).to(DEV)
gCD = torch.from_numpy(np.load(f'{OUT}/georgia_yCD.npy')).to(DEV)

Fg = feats_all(Xg); Fc = feats_all(Xc)

# ---- (3) feature transfer: conduction features' external AUC ----
transfer = {}
COND = ['CLBBB', 'CRBBB', 'IRBBB', 'LAFB']
for code in COND:
    c = CODES.index(code); r = int(np.argmax(fc[:, c]))
    row = {'feature': r, 'germany_auc': round(float(fc[r, c]), 3)}
    if int(glabel(code).sum()) >= 10:
        row['georgia_specific_auc'] = round(auc(Fg[:, r], glabel(code)), 3)
        row['georgia_n_pos'] = int(glabel(code).sum())
    row['georgia_CD_auc'] = round(auc(Fg[:, r], gCD), 3)
    row['chapman_CD_auc'] = round(auc(Fc[:, r], cCD), 3)
    transfer[code] = row
    print(f'  {code} feat#{r}: DE {row["germany_auc"]} | GA-spec {row.get("georgia_specific_auc")} '
          f'| GA-CD {row["georgia_CD_auc"]} | CN-CD {row["chapman_CD_auc"]}', flush=True)

# ---- (1) reference median beats (the diagnostic waveform) ----
def median_beat(X, mask, win=100, detlead=1, maxrec=600):
    idx = torch.where(mask.bool())[0][:maxrec].tolist()
    beats = []
    Xn = ((X - MU) / SD).cpu().numpy()
    for e in idx:
        sig = Xn[e, detlead]
        pk, _ = find_peaks(np.abs(sig), distance=40, height=np.percentile(np.abs(sig), 90))
        for p in pk:
            if p - win//2 >= 0 and p + win//2 < 1000:
                beats.append(Xn[e, :, p-win//2:p+win//2])
    if len(beats) < 5: return None, 0
    B = np.stack(beats)                                  # (nbeats,12,win)
    return np.median(B, 0), len(beats)                   # (12,win)

refbeats = {}
jobs = [('LBBB_US', Xg, glabel('CLBBB'), 'CLBBB'), ('RBBB_US', Xg, glabel('CRBBB'), 'CRBBB'),
        ('CD_China', Xc, cCD, 'CLBBB'), ('NORM_US', Xg, glabel('NORM'), None)]
for name, X, mask, matchcode in jobs:
    beat, nb = median_beat(X, mask)
    if beat is None: continue
    entry = {'n_beats': nb, 'beat': [[round(float(v), 3) for v in beat[L]] for L in range(NLEAD)]}
    if matchcode:
        c = CODES.index(matchcode); r = int(np.argmax(fc[:, c])); T = template(r)   # (12,50) Germany feature
        # best-shift cosine over the beat, using top leads of the feature
        u = np.abs(T).max(1); topL = np.argsort(-u)[:4]
        Tf = T[topL].flatten()
        best = -1
        for off in range(0, beat.shape[1]-PT+1):
            wf = beat[topL, off:off+PT].flatten()
            a = Tf - Tf.mean(); b = wf - wf.mean()
            cs = float(abs((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)))
            best = max(best, cs)
        entry['match_feature'] = r; entry['match_code'] = matchcode
        entry['shape_cosine_to_germany_feature'] = round(best, 3)
        entry['germany_feature_template'] = [[round(float(v), 3) for v in T[L]] for L in range(NLEAD)]
        print(f'  refbeat {name}: {nb} beats, cosine to Germany {matchcode} feat#{r} = {best:.3f}', flush=True)
    refbeats[name] = entry

out = {'leads': LEADS, 'feature_transfer': transfer, 'reference_beats': refbeats}
json.dump(out, open(f'{QK}/ecg_refwave_data.json', 'w'))
summ = {'feature_transfer': transfer,
        'refbeat_cosines': {k: v.get('shape_cosine_to_germany_feature') for k, v in refbeats.items() if 'shape_cosine_to_germany_feature' in v}}
json.dump(summ, open(f'{QK}/ecg_refwave.json', 'w'), indent=2)
print(json.dumps(summ, indent=1), flush=True)
print('ECG REFWAVE DONE', flush=True)
