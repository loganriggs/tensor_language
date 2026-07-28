"""ECG feature SHAPE decomposition + datapoint breakdown (Logan Qs):
 (1) Each feature template is ~rank-1 = (signed lead-weights) x (one time-course). Extract
     that (rank-1 SVD) for ALL features -> the honest 'shape to look for' + spatial map.
 (2) SHAPE-cosine validation (not lead-based): cosine(feature template, empirical positive-
     minus-negative morphology of its diagnosis at the feature's active position). Data-
     grounded 'does the shape match the diagnostic signal', verifiable as a number.
 (3) Single-datapoint decomposition: break specific ECGs into feature activations; is it a
     sparse dictionary? report participation ratio + top features + per-feature logit contrib.
Exports ecg_atlas2_data.json for the updated all-features render.
"""
import ast, json
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
KNOWN = {'CRBBB': {'V1', 'V2'}, 'CLBBB': {'V1', 'V6', 'I'}, 'IRBBB': {'V1', 'V2'},
         'LAFB': {'I', 'aVL', 'III', 'aVF'}, 'LPFB': {'III', 'aVF', 'I'},
         'INJAS': {'V1', 'V2', 'V3'}, 'INJAL': {'I', 'aVL', 'V5', 'V6'},
         'IMI': {'II', 'III', 'aVF'}, 'AMI': {'V1', 'V2', 'V3', 'V4'},
         'ISCIN': {'II', 'III', 'aVF'}, 'LVH': {'V5', 'V6', 'I'}}
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; CODES = ck['codes']; NCLS = len(CODES)
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
ib = torch.load(f'{QK}/ecg_interaction_basis.pt', map_location=DEV, weights_only=False)
A = ib['A'].to(DEV); R = ib['rank']; fc = ib['feat_code_auc']
Ahat = A / A.norm(dim=0, keepdim=True).clamp_min(1e-8)
fb = torch.load(f'{QK}/ecg_fold_block0.pt', map_location=DEV, weights_only=False)
Gsqrt_inv = torch.linalg.inv(fb['Gsqrt'].to(DEV))

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV); Xte_n = (Xte - MU) / SD
Yte = np.zeros((int((fold == 10).sum()), NCLS), dtype=np.float32)
for i, cc in enumerate(df.scp_codes.values[fold == 10]):
    for j, c in enumerate(CODES):
        if c in cc: Yte[i, j] = 1.0
Yte = torch.from_numpy(Yte).to(DEV); NTE = Yte.shape[0]
ecg_ids = df.index.values[fold == 10]


def patch(xn):
    B = xn.shape[0]
    return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


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


@torch.no_grad()
def model_logits(xn):
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        h = h + ((hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


HN = torch.cat([hn0_of(Xte_n[i:i+2048]) for i in range(0, len(Xte_n), 2048)])   # (NTE,NP,D)
ACT = (HN @ Ahat).pow(2)                                                          # (NTE,NP,R)
Lte = torch.cat([model_logits(Xte_n[i:i+2048]) for i in range(0, len(Xte_n), 2048)]).float()


def peak_pos(r):
    return ACT[:, :, r].argmax(1)                        # (NTE,) best patch per ecg for feature r


def template(r, k=300):
    flat = ACT[:, :, r].reshape(-1); kk = min(k, int((flat > 0).sum()))
    topi = torch.topk(flat, kk).indices; ex = topi // NP; pos = topi % NP
    T = torch.zeros(NLEAD, PT, device=DEV)
    for e, p in zip(ex.tolist(), pos.tolist()):
        T += Xte_n[e, :, p*PT:(p+1)*PT]
    return T / kk


def rank1(T):
    U, S, Vh = torch.linalg.svd(T, full_matrices=False)
    u = U[:, 0] * S[0].sqrt(); v = Vh[0] * S[0].sqrt()
    # canonical sign: make max-|lead-weight| positive
    if u[u.abs().argmax()] < 0:
        u, v = -u, -v
    frac = float((S[0]**2 / (S**2).sum()))
    return u.cpu().numpy(), v.cpu().numpy(), frac


# ---- all features: shape decomposition + shape-cosine vs empirical diagnostic morphology ----
features = {}
for r in range(R):
    served = sorted([(CODES[c], round(float(fc[r, c]), 3)) for c in range(NCLS) if fc[r, c] >= 0.72],
                    key=lambda t: -t[1])
    T = template(r)
    u, v, frac = rank1(T)
    bestc = int(np.argmax(fc[r]))
    # shape-cosine: empirical positive-minus-negative morphology of bestc at feature's peak position
    pos_idx = peak_pos(r)                                 # per-ecg peak patch
    patches = torch.stack([Xte_n[e, :, p*PT:(p+1)*PT] for e, p in enumerate(pos_idx.tolist())])  # (NTE,12,50)
    m = Yte[:, bestc].bool()
    if m.sum() >= 5 and (~m).sum() >= 5:
        Dmorph = (patches[m].mean(0) - patches[~m].mean(0))
        sc = float(F.cosine_similarity(T.flatten().unsqueeze(0), Dmorph.flatten().unsqueeze(0))[0])
    else:
        sc = None
    features[r] = {'serves': served, 'best_code': CODES[bestc], 'best_auc': round(float(fc[r, bestc]), 3),
                   'lead_weights': {LEADS[L]: round(float(u[L]), 3) for L in range(NLEAD)},
                   'time_course': [round(float(x), 4) for x in v],
                   'rank1_frac': round(frac, 3), 'shape_cosine_posneg': None if sc is None else round(abs(sc), 3)}

# ---- explicit readout for datapoint decomposition (mean-pool features -> 28 codes) ----
def poolfeats(xn_batch):
    hn = torch.cat([hn0_of(xn_batch[i:i+2048]) for i in range(0, len(xn_batch), 2048)])
    return (hn @ Ahat).pow(2).mean(1)                    # (N,R) mean over patches
Fte = poolfeats(Xte_n)
Xtr = torch.from_numpy(np.load(f'{OUT}/ecg_X_train.npy')).to(DEV); Xtr_n = (Xtr - MU) / SD
Ftr = poolfeats(Xtr_n)
Ytr = np.zeros((int((fold <= 8).sum()), NCLS), dtype=np.float32)
for i, cc in enumerate(df.scp_codes.values[fold <= 8]):
    for j, c in enumerate(CODES):
        if c in cc: Ytr[i, j] = 1.0
Ytr = torch.from_numpy(Ytr).to(DEV)
mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True).clamp_min(1e-6)
Ftr_s, Fte_s = (Ftr - mu) / sd, (Fte - mu) / sd
lin = nn.Linear(R, NCLS).to(DEV)
opt = torch.optim.Adam(lin.parameters(), lr=5e-3, weight_decay=1e-3)
for _ in range(4000):
    bi = torch.randint(0, len(Ftr_s), (1024,), device=DEV)
    loss = F.binary_cross_entropy_with_logits(lin(Ftr_s[bi]), Ytr[bi])
    opt.zero_grad(); loss.backward(); opt.step()
Wc = lin.weight.detach()                                  # (NCLS,R)

# pick example ECGs: confident correct positives for LBBB, AMI, and a NORM
def pick(code):
    c = CODES.index(code); m = Yte[:, c].bool()
    if m.sum() == 0: return None
    idx = torch.where(m)[0]
    best = idx[Lte[idx, c].argmax()]
    return int(best)
examples = {}
for code in ['CLBBB', 'AMI', 'IMI', 'NORM']:
    ei = pick(code)
    if ei is None: continue
    c = CODES.index(code)
    act = Fte[ei]                                         # raw mean-pooled activations (R,)
    pr = float((act / act.sum()).pow(2).sum())            # not PR; compute PR next
    PR = float(act.sum()**2 / (act.pow(2).sum() + 1e-9))  # participation ratio (eff # active features)
    contrib = (Fte_s[ei] * Wc[c])                         # per-feature contribution to code logit
    topc = torch.argsort(-contrib)[:6]
    topact = torch.argsort(-act)[:6]
    examples[code] = {'ecg_id': int(ecg_ids[ei]), 'model_logit': round(float(Lte[ei, c]), 2),
                      'participation_ratio': round(PR, 1), 'n_features': R,
                      'top_activated_features': [(int(f), round(float(act[f]), 2)) for f in topact],
                      'top_logit_contributors': [(int(f), CODES[c], round(float(contrib[f]), 2),
                                                  features[int(f)]['best_code']) for f in topc]}

# summary stats
scs = [features[r]['shape_cosine_posneg'] for r in features if features[r]['serves'] and features[r]['shape_cosine_posneg'] is not None]
r1 = [features[r]['rank1_frac'] for r in features]
summary = {'n_features': R, 'mean_rank1_frac': round(float(np.mean(r1)), 3),
           'features_serving_a_code': sum(1 for r in features if features[r]['serves']),
           'mean_shape_cosine_posneg_servingfeats': round(float(np.mean(scs)), 3) if scs else None,
           'shape_cosine_by_feature': {str(r): features[r]['shape_cosine_posneg'] for r in features if features[r]['serves']},
           'examples': examples}
out = {'leads': LEADS, 'features': {str(r): features[r] for r in features}, 'summary': summary, 'examples': examples}
json.dump(out, open(f'{QK}/ecg_atlas2_data.json', 'w'))
json.dump(summary, open(f'{QK}/ecg_shape_decomp.json', 'w'), indent=2)
print(json.dumps(summary, indent=1)[:1500], flush=True)
print('ECG SHAPE DECOMP DONE', flush=True)
