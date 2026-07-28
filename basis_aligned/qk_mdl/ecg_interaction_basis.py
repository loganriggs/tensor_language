"""ECG minimal interaction basis (Logan: minimal, simple, interpretable circuit; basis
sparse rel. input AND output). The block-0 MLP folds to symmetric T0s[o,i,j]; the 192
neurons are a redundant CP factorization. Refit a MINIMAL symmetric CP to the layer's
ACTUAL behavior on data (tensor-action fidelity, the arbiter):
    out_o ≈ sum_r U[o,r] (a_r · hn)^2 ,   a_r input feature dir, U output map
Sweep rank R -> behavioral R2 frontier = how few features the layer really uses.
Render top features as ECG waveforms; map each to codes. Minimal circuit = the smallest
R that keeps downstream code AUC. Includes a downstream-faithful check: splice the
rank-R MLP-0 back into the FULL model and measure macro-AUC (does the minimal circuit
preserve the model's behavior end-to-end?).
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
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']; CODES = ck['codes']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
fb = torch.load(f'{QK}/ecg_fold_block0.pt', map_location=DEV)
Gsqrt_inv = torch.linalg.inv(fb['Gsqrt'].to(DEV)); T0s = fb['T0_sym'].to(DEV)

Ytr_rs = torch.from_numpy(np.load(f'{QK}/ecg_readspace_train.npy')).to(DEV)
Yte_rs = torch.from_numpy(np.load(f'{QK}/ecg_readspace_test.npy')).to(DEV)
HNtr = Ytr_rs @ Gsqrt_inv.T                          # unwhiten -> hn (Ntr*NP, D)
HNte = Yte_rs @ Gsqrt_inv.T
df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV); Xte_n = (Xte - MU) / SD
Yte = np.zeros((int((fold == 10).sum()), NCLS), dtype=np.float32)
for i, cc in enumerate(df.scp_codes.values[fold == 10]):
    for j, c in enumerate(CODES):
        if c in cc:
            Yte[i, j] = 1.0
Yte = torch.from_numpy(Yte).to(DEV); NTE = Yte.shape[0]


def tact(hn):
    return torch.einsum('oij,ni,nj->no', T0s, hn, hn)


# precompute target out on a fixed eval set
with torch.no_grad():
    tgt_te = tact(HNte[:20000]); tvar = (tgt_te - tgt_te.mean(0)).pow(2).sum(-1).mean()


def fit_cp(R, steps=4000):
    A = nn.Parameter(torch.randn(D, R, device=DEV) / np.sqrt(D))
    U = nn.Parameter(torch.randn(D, R, device=DEV) / np.sqrt(R))
    opt = torch.optim.Adam([A, U], lr=3e-3)
    for st in range(steps):
        bi = torch.randint(0, HNtr.shape[0], (8192,), device=DEV)
        hn = HNtr[bi]
        with torch.no_grad():
            tgt = tact(hn)
        act = (hn @ A).pow(2)                          # (B,R) feature activations
        pred = act @ U.T
        loss = (pred - tgt).pow(2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred_te = (HNte[:20000] @ A).pow(2) @ U.T
        r2 = 1 - (pred_te - tgt_te).pow(2).sum(-1).mean().item() / tvar.item()
    return A.detach(), U.detach(), r2


# ---- rank sweep: behavioral tensor-action R2 ----
sweep = {}
best = {}
for R in [8, 16, 32, 64, 96, 192]:
    A, U, r2 = fit_cp(R, steps=4000)
    sweep[R] = round(r2, 3)
    best[R] = (A, U)
    print(f'  rank {R}: tensor-action R2 = {r2:.3f}', flush=True)

# ---- downstream-faithful splice: replace block-0 MLP with rank-R CP, measure macro-AUC ----
@torch.no_grad()
def forward_cp(x, A=None, U=None):
    xn = (x - MU) / SD
    B = xn.shape[0]
    h = xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)
    h = h @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); Bb, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(Bb, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(Bb, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(Bb, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        if li == 0 and A is not None:
            mout = ((hn2 @ A).pow(2)) @ U.T            # rank-R CP replaces block-0 MLP
        else:
            mout = ((hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T
        h = h + mout
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


def macro_auc(A=None, U=None):
    s = torch.cat([forward_cp(Xte[i:i+2048], A, U) for i in range(0, len(Xte), 2048)]).float()
    Rk = torch.argsort(torch.argsort(s, 0), 0).float() + 1
    aucs = []
    cap = []
    for c in range(NCLS):
        lab = Yte[:, c].bool(); p = lab.sum().float(); n = (~lab).sum().float()
        if p >= 10 and n > 0:
            aucs.append(float((Rk[lab, c].sum()-p*(p+1)/2)/(p*n))); cap.append(c)
    return float(np.mean(aucs)), aucs, cap


base_macro, base_aucs, cap = macro_auc()
splice = {}
for R in [8, 16, 32, 64]:
    A, U = best[R]
    m, _, _ = macro_auc(A, U)
    splice[R] = round(m, 3)
    print(f'  splice rank {R}: full-model macro-AUC = {m:.3f} (base {base_macro:.3f})', flush=True)

# ---- render + code-map for a chosen rank (smallest R within 0.01 macro of base) ----
Rpick = next((R for R in [8, 16, 32, 64] if splice[R] >= base_macro - 0.01), 64)
A, U = best[Rpick]
with torch.no_grad():
    act_te = (HNte @ A).pow(2).reshape(NTE, NP, Rpick).amax(1)     # (NTE,R) per-ecg feature activation
def auc(sc, lab):
    lab = lab.bool(); p = lab.sum().float(); n = (~lab).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(sc)).float() + 1
    return float((r[lab].sum()-p*(p+1)/2)/(p*n))
feat_code = np.zeros((Rpick, NCLS))
for r in range(Rpick):
    for c in cap:
        feat_code[r, c] = auc(act_te[:, r], Yte[:, c])
# render each feature waveform (top-activating test patches) + serves-codes
act_patch = (HNte @ A).pow(2)                                       # (NTE*NP, R)
render = {}
for r in range(Rpick):
    a = act_patch[:, r]; k = min(300, int((a > 0).sum()))
    topi = torch.topk(a, k).indices; ex = topi // NP; pos = topi % NP
    tmpl = torch.zeros(NLEAD, PT, device=DEV)
    for e, p in zip(ex.tolist(), pos.tolist()):
        tmpl += Xte_n[e, :, p*PT:(p+1)*PT]
    tmpl /= k
    peak = {LEADS[L]: round(float(tmpl[L].abs().max()), 2) for L in range(NLEAD)}
    served = sorted([CODES[c] for c in cap if feat_code[r, c] >= 0.75], key=lambda cc: -feat_code[r, CODES.index(cc)])
    render[r] = {'top_leads': sorted(peak, key=peak.get, reverse=True)[:3],
                 'best_code': (lambda i: {'code': CODES[i], 'auc': round(float(feat_code[r, i]), 3)})(int(np.argmax(feat_code[r]))),
                 'serves_codes': served[:6]}
# per-code: how many features (minimal circuit size in the interaction basis)
per_code = {CODES[c]: {'n_features_auc>=0.75': int((feat_code[:, c] >= 0.75).sum()),
                       'top_feature': int(np.argmax(feat_code[:, c])),
                       'top_feature_auc': round(float(feat_code[:, c].max()), 3)} for c in cap}

res = {'rank_sweep_tensor_action_R2': sweep, 'splice_full_macro_auc': splice, 'base_macro_auc': round(base_macro, 3),
       'chosen_rank': Rpick, 'mean_features_per_code': round(float(np.mean([per_code[CODES[c]]['n_features_auc>=0.75'] for c in cap])), 1),
       'per_code': per_code, 'feature_render': render}
torch.save({'A': A.cpu(), 'U': U.cpu(), 'rank': Rpick, 'feat_code_auc': feat_code}, f'{QK}/ecg_interaction_basis.pt')
json.dump(res, open(f'{QK}/ecg_interaction_basis.json', 'w'), indent=2)
print(json.dumps({k: res[k] for k in ('rank_sweep_tensor_action_R2', 'splice_full_macro_auc', 'base_macro_auc',
      'chosen_rank', 'mean_features_per_code')}, indent=1), flush=True)
print('sample features:', json.dumps({r: render[r] for r in list(render)[:8]}, indent=1), flush=True)
print('ECG INTERACTION BASIS DONE', flush=True)
