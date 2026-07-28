"""ECG causal steering in the MINIMAL INTERACTION BASIS (Logan: insert/remove features ->
predictable diagnosis changes on the test set; correlated diagnoses share features).
Unlike the neuron gauge (§28, buffered), here we steer along the behavioral feature dirs
a_r (from §32). Intervene on hn at the block-0 MLP input:
  REMOVE feature r:  hn' = hn - (hn·â_r) â_r     (project the feature out)
  INSERT feature r:  hn' = hn + α·â_r            (add it to true negatives)
Measure on the held-out TEST set:
  (1) per code, remove its TOP feature -> AUC drop; insert -> negative-prob rise;
  (2) SHARED features: remove one feature -> do ALL its served (correlated) codes fall
      together? (the shared-feature-causes-correlation test).
"""
import ast, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; CODES = ck['codes']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
ib = torch.load(f'{QK}/ecg_interaction_basis.pt', map_location=DEV, weights_only=False)
A = ib['A'].to(DEV); R = ib['rank']; fc = ib['feat_code_auc']       # A: (D,R) feature dirs in hn space
Ahat = A / A.norm(dim=0, keepdim=True).clamp_min(1e-8)

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
Yte = np.zeros((int((fold == 10).sum()), NCLS), dtype=np.float32)
for i, cc in enumerate(df.scp_codes.values[fold == 10]):
    for j, c in enumerate(CODES):
        if c in cc:
            Yte[i, j] = 1.0
Yte = torch.from_numpy(Yte).to(DEV)


def patch(x):
    B = x.shape[0]
    return x.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def forward(x, remove_r=None, insert_r=None, alpha=0.0):
    xn = (x - MU) / SD
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        if li == 0 and remove_r is not None:
            u = Ahat[:, remove_r]; hn2 = hn2 - (hn2 @ u).unsqueeze(-1) * u
        if li == 0 and insert_r is not None:
            hn2 = hn2 + alpha * Ahat[:, insert_r]
        inner = (hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)
        h = h + (inner @ W[mw+'Dn.weight'].T)
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


@torch.no_grad()
def scores(**kw):
    return torch.cat([forward(Xte[i:i+2048], **kw) for i in range(0, len(Xte), 2048)]).float()


def auc_col(s, c):
    lab = Yte[:, c].bool(); p = lab.sum().float(); n = (~lab).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(s[:, c])).float() + 1
    return float((r[lab].sum()-p*(p+1)/2)/(p*n))


base_s = scores()
base = np.array([auc_col(base_s, c) for c in range(NCLS)])
capable = [c for c in range(NCLS) if base[c] >= 0.75 and int(Yte[:, c].sum()) >= 10]
# typical feature activation magnitude for insert scaling
with torch.no_grad():
    hn0 = []
    for i in range(0, len(Xte), 2048):
        xn = (Xte[i:i+2048] - MU) / SD
        h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
        aw = 'blocks.0.'; hnn = F.rms_norm(h, (D,)); B, T, _ = hnn.shape
        def hd(nm): return F.rms_norm((hnn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hnn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W['blocks.0.proj.weight'].T)
        hn0.append(F.rms_norm(h, (D,)))
    hn0 = torch.cat(hn0)
    proj = (hn0.reshape(-1, D) @ Ahat)                          # (Npatch, R)
    act_std = proj.std(0)                                       # per-feature activation scale

# ---- (1) per-code: remove top feature, insert top feature ----
percode = {}
for c in capable:
    r = int(np.argmax(fc[:, c]))
    s_rm = scores(remove_r=r)
    a_rm = auc_col(s_rm, c)
    alpha = 3.0 * float(act_std[r])
    s_in = scores(insert_r=r, alpha=alpha)
    mneg_b = float(torch.sigmoid(base_s[:, c])[~Yte[:, c].bool()].mean())
    mneg_i = float(torch.sigmoid(s_in[:, c])[~Yte[:, c].bool()].mean())
    percode[CODES[c]] = {'top_feature': r, 'auc': round(float(base[c]), 3),
                         'auc_after_remove': round(a_rm, 3), 'auc_drop': round(float(base[c]-a_rm), 3),
                         'neg_prob_base': round(mneg_b, 3), 'neg_prob_after_insert': round(mneg_i, 3),
                         'insert_rise': round(mneg_i-mneg_b, 3)}

# ---- (2) shared features: remove -> do all served codes fall together? ----
shared = {}
for r in range(R):
    served = [c for c in capable if fc[r, c] >= 0.72]
    if len(served) >= 2:
        s_rm = scores(remove_r=r)
        drops = {CODES[c]: round(float(base[c] - auc_col(s_rm, c)), 3) for c in served}
        shared[r] = {'served_codes': [CODES[c] for c in served],
                     'auc_drops_on_remove': drops,
                     'mean_drop': round(float(np.mean(list(drops.values()))), 3),
                     'all_fell': all(v > 0.005 for v in drops.values())}

res = {'n_capable': len(capable), 'chosen_rank': R,
       'mean_auc_drop_remove_top': round(float(np.mean([percode[CODES[c]]['auc_drop'] for c in capable])), 3),
       'codes_remove_collapses>=0.05': [CODES[c] for c in capable if percode[CODES[c]]['auc_drop'] >= 0.05],
       'mean_insert_rise': round(float(np.mean([percode[CODES[c]]['insert_rise'] for c in capable])), 3),
       'n_shared_features': len(shared),
       'shared_features_all_served_fell': int(sum(1 for v in shared.values() if v['all_fell'])),
       'per_code': percode, 'shared': shared}
json.dump(res, open(f'{QK}/ecg_feature_steer.json', 'w'), indent=2)
print(json.dumps({k: res[k] for k in ('mean_auc_drop_remove_top', 'codes_remove_collapses>=0.05',
      'mean_insert_rise', 'n_shared_features', 'shared_features_all_served_fell')}, indent=1), flush=True)
print('example shared-feature removals:', flush=True)
for r in list(shared)[:6]:
    print(f'  feat#{r}: {shared[r]["auc_drops_on_remove"]} (all fell: {shared[r]["all_fell"]})', flush=True)
print('ECG FEATURE STEER DONE', flush=True)
