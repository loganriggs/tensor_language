"""Path 2 step 5: interpret how the distilled student detects each class, focus on AF. Two probes:
 (1) TEMPORAL CONTEXT: keep only the first k time-patches (zero the rest), sweep k -> how many
     beats each class needs. Morphology (LBBB/RBBB) should saturate at ~1 beat; rhythm (AF, and
     rate classes) should need many patches (irregularity/rate need multiple beats).
 (2) LEADS: per-class lead occlusion -> which leads (AF: II/V1 for P/fibrillatory waves).
This shows AF = distributed-temporal multi-lead, morphology = focal.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
ck = torch.load(f'{QK}/ecg_student_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; TC = ck['classes']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
Yhard = (torch.from_numpy(np.load(f'{QK}/teacher_soft_test.npy')) > 0.5).float().to(DEV)


def patch(xn):
    B = xn.shape[0]; return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def forward(x, keep_k=None, occ_lead=None):
    xn = (x - MU) / SD
    if occ_lead is not None:
        xn = xn.clone(); xn[:, occ_lead] = 0.0
    p = patch(xn)                                        # (B,NP,PXD)
    if keep_k is not None:
        p = p.clone(); p[:, keep_k:] = 0.0               # zero time-patches beyond first keep_k
    h = p @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2'); v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        h = h + ((hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


@torch.no_grad()
def all_auc(**kw):
    s = torch.cat([forward(Xte[i:i+2048], **kw) for i in range(0, len(Xte), 2048)]).float()
    R = torch.argsort(torch.argsort(s, 0), 0).float() + 1
    out = np.zeros(NCLS)
    for c in range(NCLS):
        lab = Yhard[:, c].bool(); p = lab.sum().float(); n = (~lab).sum().float()
        out[c] = 0.5 if p == 0 or n == 0 else float((R[lab, c].sum()-p*(p+1)/2)/(p*n))
    return out


base = all_auc()
# temporal context sweep
Ks = [1, 2, 3, 5, 8, 12, 20]
ctx = {K: all_auc(keep_k=K) for K in Ks}
# lead occlusion
lead_drop = np.stack([base - all_auc(occ_lead=L) for L in range(NLEAD)])   # (12,6)

res = {'base_auc': {TC[c]: round(float(base[c]), 3) for c in range(NCLS)}, 'per_class': {}}
for c in range(NCLS):
    cls = TC[c]
    curve = {str(K): round(float(ctx[K][c]), 3) for K in Ks}
    # min patches to reach 95% of full (base) AUC-above-chance
    tgt = 0.5 + 0.95*(base[c]-0.5)
    kmin = next((K for K in Ks if ctx[K][c] >= tgt), 20)
    topL = [LEADS[i] for i in np.argsort(-lead_drop[:, c])[:3]]
    res['per_class'][cls] = {'base': round(float(base[c]), 3), 'ctx_curve': curve,
                             'min_patches_95pct': kmin, 'auc_1patch': curve['1'], 'top_leads': topL}
    print(f"  {cls:6s}: 1-patch {curve['1']} -> full {round(float(base[c]),3)} | min-patches-95% {kmin} | leads {topL}", flush=True)

RHYTHM = ['SB', 'AF', 'ST']; MORPH = ['1dAVb', 'RBBB', 'LBBB']
res['rhythm_mean_min_patches'] = round(float(np.mean([res['per_class'][c]['min_patches_95pct'] for c in RHYTHM])), 1)
res['morph_mean_min_patches'] = round(float(np.mean([res['per_class'][c]['min_patches_95pct'] for c in MORPH])), 1)
res['rhythm_mean_1patch_auc'] = round(float(np.mean([res['per_class'][c]['auc_1patch'] for c in RHYTHM])), 3)
res['morph_mean_1patch_auc'] = round(float(np.mean([res['per_class'][c]['auc_1patch'] for c in MORPH])), 3)
json.dump(res, open(f'{QK}/ecg_af_interp.json', 'w'), indent=2)
print(f"RHYTHM needs {res['rhythm_mean_min_patches']} patches (1-patch AUC {res['rhythm_mean_1patch_auc']}) vs "
      f"MORPH {res['morph_mean_min_patches']} patches (1-patch AUC {res['morph_mean_1patch_auc']})", flush=True)
print('ECG AF INTERP DONE', flush=True)
