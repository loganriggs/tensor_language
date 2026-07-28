"""ECG FULL-circuit sufficiency frontier (Logan Q follow-up): the block-0 interaction
features top out at ~0.84 (0.908 retention) because attention-layer-2 is a co-equal part
of the circuit (§29) NOT in that basis. Bring Attn-2 in: its pooled output contributes
LINEARLY to the diagnosis logit (head reads the final residual linearly, Attn-2 adds to it),
so its interpretable 'features' are directions in its pooled output a2. Fit the explicit
readout on [block-0 interaction features + Attn-2 directions], sweep counts, and ask:
  (1) does adding Attn-2 close the gap to the model's 0.925?
  (2) what is the minimal Attn-2 feature count (its description length)?
  (3) random-direction control on Attn-2 (is its basis privileged?).
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
MODEL_MACRO = 0.925

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
def labels(mask):
    Y = np.zeros((mask.sum(), NCLS), dtype=np.float32)
    for i, cc in enumerate(df.scp_codes.values[mask]):
        for j, c in enumerate(CODES):
            if c in cc: Y[i, j] = 1.0
    return torch.from_numpy(Y).to(DEV)
Ytr, Yte = labels(fold <= 8), labels(fold == 10)
Xtr = torch.from_numpy(np.load(f'{OUT}/ecg_X_train.npy')).to(DEV)
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)


def patch(xn):
    B = xn.shape[0]
    return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def attn2_pooled(x):
    xn = (x - MU) / SD
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    a2 = None
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        ac = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T
        h = h + ac
        if li == 2:
            a2 = ac.mean(1)                                     # (B,D) pooled Attn-2 output
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        h = h + ((hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T
    return a2


def block0_feats(tag, n):
    rs = torch.from_numpy(np.load(f'{QK}/ecg_readspace_{tag}.npy')).to(DEV)
    hn = rs @ Gsqrt_inv.T
    f = (hn @ Ahat).pow(2).reshape(n, NP, R)
    return torch.cat([f.mean(1), f.amax(1)], 1)                  # (n,2R)

B0tr, B0te = block0_feats('train', Ytr.shape[0]), block0_feats('test', Yte.shape[0])
A2tr = torch.cat([attn2_pooled(Xtr[i:i+2048]) for i in range(0, len(Xtr), 2048)])
A2te = torch.cat([attn2_pooled(Xte[i:i+2048]) for i in range(0, len(Xte), 2048)])
# standardize both feature banks
def std(Xtr, Xte):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    return (Xtr-mu)/sd, (Xte-mu)/sd
B0tr, B0te = std(B0tr, B0te); A2tr, A2te = std(A2tr, A2te)

def auc_col(s, y, c):
    lab = y[:, c].bool(); p = lab.sum().float(); n = (~lab).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(s[:, c])).float() + 1
    return float((r[lab].sum()-p*(p+1)/2)/(p*n))
capable = [c for c in range(NCLS) if int(Yte[:, c].sum()) >= 10 and
           ({d['code']: d['auc'] for d in json.load(open(f'{QK}/ecg_codes_train.json'))['per_code']}.get(CODES[c], 0) >= 0.75)]

def fit(Xtr, Xte):
    lin = nn.Linear(Xtr.shape[1], NCLS).to(DEV)
    opt = torch.optim.Adam(lin.parameters(), lr=5e-3, weight_decay=1e-3)
    for _ in range(3500):
        bi = torch.randint(0, len(Xtr), (1024,), device=DEV)
        loss = F.binary_cross_entropy_with_logits(lin(Xtr[bi]), Ytr[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s = lin(Xte)
    return float(np.mean([auc_col(s, Yte, c) for c in capable]))

# rank block-0 features by discriminativeness; rank Attn-2 dirs by per-dim discriminativeness
disc0 = np.abs(fc - 0.5).max(1); ord0 = list(np.argsort(-disc0))
def auc1(vec, yc):
    lab = yc.bool(); p = lab.sum().float(); n = (~lab).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(vec)).float() + 1
    return float((r[lab].sum()-p*(p+1)/2)/(p*n))
dA = np.zeros(D)
for j in range(D):
    dA[j] = max(abs(auc1(A2te[:, j], Yte[:, c]) - 0.5) for c in capable)
ordA = list(np.argsort(-dA))

def cols0(K): return ord0[:K] + [R + i for i in ord0[:K]]
K1 = 64
b0_only = fit(B0tr[:, cols0(K1)], B0te[:, cols0(K1)])
res = {'model_macro': MODEL_MACRO, 'block0_only_K64': round(b0_only, 3),
       'block0_only_retention': round(b0_only/MODEL_MACRO, 3), 'full_frontier': {}}
for K2 in [0, 2, 4, 8, 16, 32, 64, 96]:
    if K2 == 0:
        macro = b0_only
    else:
        Xtr = torch.cat([B0tr[:, cols0(K1)], A2tr[:, ordA[:K2]]], 1)
        Xte = torch.cat([B0te[:, cols0(K1)], A2te[:, ordA[:K2]]], 1)
        macro = fit(Xtr, Xte)
    res['full_frontier'][K2] = {'macro': round(macro, 3), 'retention': round(macro/MODEL_MACRO, 3)}
    print(f'  block0(64)+attn2({K2}): macro {macro:.3f} retention {macro/MODEL_MACRO:.3f}', flush=True)
# attn2 alone + random-attn2 control at K2=16
a2_alone = fit(A2tr[:, ordA[:32]], A2te[:, ordA[:32]])
rng = np.random.default_rng(0)
randA = list(rng.choice(D, 16, replace=False))
rand_full = fit(torch.cat([B0tr[:, cols0(K1)], A2tr[:, randA]], 1), torch.cat([B0te[:, cols0(K1)], A2te[:, randA]], 1))
res['attn2_alone_K32'] = round(a2_alone, 3)
res['full_block0_64_plus_random16attn2'] = round(rand_full, 3)
res['full_block0_64_plus_ranked16attn2'] = res['full_frontier'][16]['macro']
best = max(v['macro'] for v in res['full_frontier'].values())
res['best_full_macro'] = round(best, 3); res['best_full_retention'] = round(best/MODEL_MACRO, 3)
json.dump(res, open(f'{QK}/ecg_fullcircuit_metrics.json', 'w'), indent=2)
print(json.dumps({k: res[k] for k in ('block0_only_K64', 'attn2_alone_K32', 'best_full_macro', 'best_full_retention',
      'full_block0_64_plus_random16attn2', 'full_block0_64_plus_ranked16attn2')}, indent=1), flush=True)
print('ECG FULLCIRCUIT METRICS DONE', flush=True)
