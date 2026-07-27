"""ECG STAGE 1b: does the fold recover KNOWN conduction-disturbance (bundle branch
block) morphology? Two validations against cardiology ground truth:
(A) Causal per-lead importance for CD: occlude each of the 12 leads, measure CD-AUC
    drop. BBB is diagnosed from specific leads (V1, V6, I, aVL, V5). Do the causally
    important leads match?
(B) Exact fold rendering: top CD-discriminative bilinear units, render their preferred
    12-lead x 50-sample waveform (top eigenvector of the signal-space quadratic form).
    Report per-lead energy concentration and save waveforms. The fold's unique job is
    rendering the exact discriminative waveform, which saliency cannot do.
Lead order (PTB-XL): I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6 (idx 0-11).
"""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
BBB_LEADS = {'V1', 'V6', 'I', 'aVL', 'V5'}     # cardiology: BBB diagnostic leads
SUP = ['NORM', 'MI', 'STTC', 'CD', 'HYP']; CD = 3
ck = torch.load(f'{QK}/ecg_model.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)

Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
Yte = torch.from_numpy(np.load(f'{OUT}/ecg_Y_test.npy')).to(DEV)
norm = lambda x: (x - MU) / SD


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
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


def auc(s, lab):
    o = torch.argsort(torch.argsort(s)); r = o.float()+1
    p = lab.sum().float(); n = (~lab).sum().float()
    return 0.5 if p == 0 or n == 0 else float((r[lab].sum()-p*(p+1)/2)/(p*n))


# (A) causal per-lead importance for CD (restrict to CD-vs-NORM records for a clean contrast)
sel = (Yte[:, CD] == 1) | ((Yte.sum(1) == 1) & (Yte[:, 0] == 1))    # CD or pure-NORM
Xs, ys = Xte[sel], (Yte[sel, CD] == 1)
with torch.no_grad():
    base = torch.cat([forward(Xs[i:i+2048])[:, CD] for i in range(0, len(Xs), 2048)])
base_auc = auc(base, ys)
lead_imp = {}
for L in range(NLEAD):
    with torch.no_grad():
        sc = torch.cat([forward(Xs[i:i+2048], occ_lead=L)[:, CD] for i in range(0, len(Xs), 2048)])
    lead_imp[LEADS[L]] = round(base_auc - auc(sc, ys), 4)
ranked = sorted(lead_imp, key=lambda k: -lead_imp[k])
top5 = ranked[:5]
bbb_hits = len(set(top5) & BBB_LEADS)

# (B) fold rendering: CD-discriminative units + preferred waveforms
A = W['embed.weight'].T @ W['blocks.1.L.weight'].T          # (PXD, INNER)
Bm = W['embed.weight'].T @ W['blocks.1.R.weight'].T
with torch.no_grad():
    feat = []
    for i in range(0, len(Xte), 2048):
        P = patch(norm(Xte[i:i+2048]))
        feat.append((torch.einsum('bnp,pj->bnj', P, A) * torch.einsum('bnp,pj->bnj', P, Bm)).mean(1))
    feat = torch.cat(feat)
cd_lab = (Yte[:, CD] == 1)
unit_auc = np.array([abs(auc(feat[:, j], cd_lab) - 0.5) for j in range(INNER)])
top_units = list(np.argsort(-unit_auc)[:8])
waves = {}; lead_energy = np.zeros(NLEAD)
for j in top_units:
    a_j, b_j = A[:, j], Bm[:, j]
    S = 0.5*(torch.outer(a_j, b_j)+torch.outer(b_j, a_j))
    ev, evec = torch.linalg.eigh(S)
    w = (evec[:, ev.abs().argmax()] * torch.sign(ev[ev.abs().argmax()])).reshape(NLEAD, PT).cpu()
    waves[int(j)] = w
    lead_energy += w.pow(2).sum(1).numpy()
lead_energy /= lead_energy.sum()
cd_lead_rank = [LEADS[i] for i in np.argsort(-lead_energy)[:5]]
fold_bbb_hits = len(set(cd_lead_rank) & BBB_LEADS)

torch.save(waves, f'{QK}/ecg_cd_waveforms.pt')
res = {'test_CD_auc': round(base_auc, 4), 'per_lead_causal_importance': lead_imp,
       'causal_top5_leads': top5, 'causal_BBB_lead_hits': f'{bbb_hits}/5',
       'fold_top5_leads_by_energy': cd_lead_rank, 'fold_BBB_lead_hits': f'{fold_bbb_hits}/5',
       'fold_top_units': [int(u) for u in top_units]}
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/ecg_analyze.json', 'w'), indent=2)
print('ECG ANALYZE DONE', flush=True)
