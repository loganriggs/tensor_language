"""ECG metric-aligned dictionary (Logan: basis sparse rel. input & output, NOT neurons).
Fit an overcomplete TopK SAE on the G^{1/2}-whitened block-0 read-space (correct geometry
from the exact fold). Then:
  (1) behavioral/moment recon check (arbiter, not FVU): does the dictionary preserve the
      folded tensor's action T(hn,hn) and the downstream code logits?
  (2) render each atom as an ECG waveform (top-activating real test patches);
  (3) atom -> code map (which diagnoses each atom serves) -> sparse, interpretable;
  (4) atom-basis sparsity of the symmetric folded tensor (few atom pairs carry the mass).
Carry-over: TopK (not L1; shrinkage biases the cubed moment). Fit on train, eval on test.
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
cfg = ck['cfg']; CODES = ck['codes']
D, NP, PT, NLEAD, NCLS = cfg['D'], cfg['NP'], cfg['PT'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
fb = torch.load(f'{QK}/ecg_fold_block0.pt', map_location=DEV)
Gsqrt, T0s = fb['Gsqrt'].to(DEV), fb['T0_sym'].to(DEV)
Gsqrt_inv = torch.linalg.inv(Gsqrt)

Ytr_rs = torch.from_numpy(np.load(f'{QK}/ecg_readspace_train.npy')).to(DEV)   # (Ntr*NP, D) whitened
Yte_rs = torch.from_numpy(np.load(f'{QK}/ecg_readspace_test.npy')).to(DEV)
df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV); Xte_n = (Xte - MU) / SD
Yte = np.zeros((int((fold == 10).sum()), NCLS), dtype=np.float32)
for i, cc in enumerate(df.scp_codes.values[fold == 10]):
    for j, c in enumerate(CODES):
        if c in cc:
            Yte[i, j] = 1.0
Yte = torch.from_numpy(Yte).to(DEV)
NTE = Yte.shape[0]
assert Yte_rs.shape[0] == NTE * NP, (Yte_rs.shape, NTE, NP)

M, K = 256, 8                                       # overcomplete atoms, active per patch


class SAE(nn.Module):
    def __init__(s):
        super().__init__()
        s.b_pre = nn.Parameter(torch.zeros(D))
        s.W_enc = nn.Parameter(torch.randn(M, D) * (1/np.sqrt(D)))
        s.b_enc = nn.Parameter(torch.zeros(M))
        s.W_dec = nn.Parameter(torch.randn(D, M) * (1/np.sqrt(M)))
    def norm_dec(s):
        with torch.no_grad():
            s.W_dec.data /= s.W_dec.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
    def encode(s, y):
        z = (y - s.b_pre) @ s.W_enc.T + s.b_enc
        val, idx = z.topk(K, dim=-1)
        val = F.relu(val)
        out = torch.zeros_like(z); out.scatter_(-1, idx, val)
        return out
    def forward(s, y):
        z = s.encode(y)
        return z @ s.W_dec.T + s.b_pre, z


sae = SAE().to(DEV); sae.norm_dec()
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
Ntr = Ytr_rs.shape[0]
for step in range(20000):
    bi = torch.randint(0, Ntr, (4096,), device=DEV)
    y = Ytr_rs[bi]
    yhat, z = sae(y)
    loss = (yhat - y).pow(2).sum(-1).mean()
    opt.zero_grad(); loss.backward(); sae.norm_dec(); opt.step(); sae.norm_dec()
    if step % 4000 == 0:
        with torch.no_grad():
            var = (y - y.mean(0)).pow(2).sum(-1).mean()
            print(f'  step {step} recon R2={1-loss.item()/var.item():.3f}', flush=True)

# ---- (1) recon quality on test read-space + behavioral (tensor-action) check ----
with torch.no_grad():
    yhat_te, z_te = sae(Yte_rs)
    var = (Yte_rs - Yte_rs.mean(0)).pow(2).sum(-1).mean()
    r2 = 1 - (yhat_te - Yte_rs).pow(2).sum(-1).mean().item() / var.item()
    dead = int((z_te.abs().sum(0) == 0).sum())
    l0 = float((z_te != 0).float().sum(-1).mean())
    # behavioral: T0s acts on hn = Gsqrt_inv @ y. Compare T(hn,hn) for true vs recon.
    hn_true = Yte_rs @ Gsqrt_inv.T
    hn_rec = yhat_te @ Gsqrt_inv.T
    def tact(hn):
        return torch.einsum('oij,ni,nj->no', T0s, hn, hn)
    a_true, a_rec = tact(hn_true[:20000]), tact(hn_rec[:20000])
    tensor_r2 = 1 - (a_true - a_rec).pow(2).sum(-1).mean().item() / (a_true - a_true.mean(0)).pow(2).sum(-1).mean().item()
print(f'test read-space R2={r2:.3f}, tensor-action R2={tensor_r2:.3f}, L0={l0:.1f}, dead={dead}/{M}', flush=True)

# ---- (3) atom -> code map: pool atom activation per ECG (max over patches), AUC vs labels
z_ecg = z_te.reshape(NTE, NP, M).amax(1)             # (NTE, M) max-pooled activation per ECG
def auc(sc, lab):
    lab = lab.bool(); p = lab.sum().float(); n = (~lab).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(sc)).float() + 1
    return float((r[lab].sum()-p*(p+1)/2)/(p*n))
capable = [c for c in range(NCLS) if int(Yte[:, c].sum()) >= 10]
atom_code_auc = np.zeros((M, NCLS))
for a in range(M):
    if z_ecg[:, a].sum() == 0: continue
    for c in capable:
        atom_code_auc[a, c] = auc(z_ecg[:, a], Yte[:, c])
# for each capable code: top atoms (by AUC), how many atoms exceed 0.70 (sparse code repr)
per_code = {}
for c in capable:
    aucs = atom_code_auc[:, c]
    top = np.argsort(-aucs)[:5]
    per_code[CODES[c]] = {'top_atoms': [int(a) for a in top], 'top_atom_aucs': [round(float(aucs[a]), 3) for a in top],
                          'n_atoms_auc>=0.70': int((aucs >= 0.70).sum())}

# ---- (2) render top atoms as waveforms: top-activating test patches -> avg raw waveform
render = {}
key_atoms = sorted(set(int(a) for c in capable for a in np.argsort(-atom_code_auc[:, c])[:2]))
for a in key_atoms:
    act = z_te[:, a]                                  # (NTE*NP,)
    k = min(300, int((act > 0).sum()))
    if k < 5: continue
    topi = torch.topk(act, k).indices
    ex = topi // NP; pos = topi % NP
    tmpl = torch.zeros(NLEAD, PT, device=DEV)
    for e, p in zip(ex.tolist(), pos.tolist()):
        tmpl += Xte_n[e, :, p*PT:(p+1)*PT]
    tmpl /= k
    peak = {LEADS[L]: round(float(tmpl[L].abs().max()), 2) for L in range(NLEAD)}
    topleads = sorted(peak, key=peak.get, reverse=True)[:3]
    served = [CODES[c] for c in capable if atom_code_auc[a, c] >= 0.70]
    render[int(a)] = {'top_leads': topleads, 'n_active_patches': int((act > 0).sum()), 'serves_codes': served}

# ---- (4) atom-basis sparsity of the symmetric folded tensor ----
# atom input direction in hn space = Gsqrt_inv @ W_dec[:,a]; project T0s onto atom pairs.
with torch.no_grad():
    Dhn = (Gsqrt_inv @ sae.W_dec)                     # (D, M) atom dirs in hn space
    Dhn = Dhn / Dhn.norm(dim=0, keepdim=True).clamp_min(1e-8)
    # T_atom[o,a,b] = sum_ij T0s[o,i,j] Dhn[i,a] Dhn[j,b]
    T_atom = torch.einsum('oij,ia,jb->oab', T0s, Dhn, Dhn)
    mass = T_atom.pow(2).sum(0)                        # (M,M) energy per atom pair
    total = mass.sum()
    flat = mass.reshape(-1).sort(descending=True).values
    frac_top1pct = float(flat[:max(1, (M*M)//100)].sum() / total)
    diag_frac = float(mass.diag().sum() / total)

res = {'M': M, 'K': K, 'test_readspace_R2': round(r2, 3), 'tensor_action_R2': round(tensor_r2, 3),
       'L0': round(l0, 1), 'dead_atoms': dead,
       'mean_atoms_per_code_auc>=0.70': round(float(np.mean([per_code[CODES[c]]['n_atoms_auc>=0.70'] for c in capable])), 1),
       'tensor_mass_top1pct_atompairs': round(frac_top1pct, 3), 'tensor_mass_diagonal': round(diag_frac, 3),
       'per_code': per_code, 'atom_render': render}
torch.save({'W_enc': sae.W_enc.detach().cpu(), 'W_dec': sae.W_dec.detach().cpu(),
            'b_pre': sae.b_pre.detach().cpu(), 'b_enc': sae.b_enc.detach().cpu(),
            'atom_code_auc': atom_code_auc}, f'{QK}/ecg_metric_dict.pt')
json.dump(res, open(f'{QK}/ecg_metric_dict.json', 'w'), indent=2)
print(json.dumps({k: res[k] for k in ('test_readspace_R2', 'tensor_action_R2', 'L0', 'dead_atoms',
      'mean_atoms_per_code_auc>=0.70', 'tensor_mass_top1pct_atompairs', 'tensor_mass_diagonal')}, indent=1), flush=True)
print('sample codes:', json.dumps({CODES[c]: per_code[CODES[c]] for c in capable[:6]}, indent=1), flush=True)
print('ECG METRIC DICT DONE', flush=True)
