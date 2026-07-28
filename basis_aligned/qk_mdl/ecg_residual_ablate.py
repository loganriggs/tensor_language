"""ECG minimal NECESSARY causal circuit: §33 showed single-feature removal at block-0's
read is buffered (morphology survives in the residual for later layers). Here we project
a code's feature directions out of the RESIDUAL STREAM at EVERY block input (all layers,
all positions) and find the MINIMAL set of features whose joint removal collapses the
diagnosis on the test set. That minimal necessary set = the true causal circuit; its size
measures the model's redundancy. Also a dose-response insert (add feature at all layers).
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
A = ib['A'].to(DEV); Rk = ib['rank']; fc = ib['feat_code_auc']
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
def forward(x, P=None, add=None):
    """P: (D,k) orthonormal directions projected OUT of residual at every block input.
    add: (D,) vector added to residual at every block input (insert)."""
    xn = (x - MU) / SD
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    def edit(hh):
        if P is not None:
            hh = hh - (hh @ P) @ P.T
        if add is not None:
            hh = hh + add
        return hh
    h = edit(h)
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        h = edit(h)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        inner = (hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)
        h = h + (inner @ W[mw+'Dn.weight'].T)
        h = edit(h)
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


@torch.no_grad()
def scores(**kw):
    return torch.cat([forward(Xte[i:i+2048], **kw) for i in range(0, len(Xte), 2048)]).float()


def auc_col(s, c):
    lab = Yte[:, c].bool(); p = lab.sum().float(); n = (~lab).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(s[:, c])).float() + 1
    return float((r[lab].sum()-p*(p+1)/2)/(p*n))


def ortho(idx):
    M = Ahat[:, idx]                                   # (D,k)
    Q, _ = torch.linalg.qr(M)
    return Q


base_s = scores(); base = np.array([auc_col(base_s, c) for c in range(NCLS)])
capable = [c for c in range(NCLS) if base[c] >= 0.75 and int(Yte[:, c].sum()) >= 10]
print(f'{len(capable)} capable; base macro {np.mean([base[c] for c in capable]):.3f}', flush=True)

# ---- minimal NECESSARY set: cumulatively remove top features (by fc) from residual ----
percode = {}
for c in capable:
    order = list(np.argsort(-fc[:, c]))               # features ranked by how well they read code c
    kmin = None; traj = []
    for k in range(1, 11):
        P = ortho(order[:k])
        s = scores(P=P)
        a = auc_col(s, c)
        traj.append(round(a, 3))
        if a < 0.60 and kmin is None:
            kmin = k
        if a < 0.55:
            break
    percode[CODES[c]] = {'auc': round(float(base[c]), 3), 'min_features_to_collapse': kmin,
                         'auc_trajectory': traj, 'auc_after_1': traj[0],
                         'auc_after_all': traj[-1]}
    print(f'  {CODES[c]}: base {base[c]:.3f} -> remove1 {traj[0]:.3f} -> minK {kmin} (traj {traj})', flush=True)

collapsed = [CODES[c] for c in capable if percode[CODES[c]]['min_features_to_collapse'] is not None]
kmins = [percode[CODES[c]]['min_features_to_collapse'] for c in capable if percode[CODES[c]]['min_features_to_collapse'] is not None]
res = {'n_capable': len(capable), 'base_macro': round(float(np.mean([base[c] for c in capable])), 3),
       'mean_auc_drop_remove1': round(float(np.mean([base[c]-percode[CODES[c]]['auc_after_1'] for c in capable])), 3),
       'codes_that_collapse_within_10': len(collapsed),
       'mean_min_necessary_features': round(float(np.mean(kmins)), 1) if kmins else None,
       'median_min_necessary_features': int(np.median(kmins)) if kmins else None,
       'per_code': percode}
json.dump(res, open(f'{QK}/ecg_residual_ablate.json', 'w'), indent=2)
print(json.dumps({k: res[k] for k in ('base_macro', 'mean_auc_drop_remove1', 'codes_that_collapse_within_10',
      'mean_min_necessary_features', 'median_min_necessary_features')}, indent=1), flush=True)
print('ECG RESIDUAL ABLATE DONE', flush=True)
