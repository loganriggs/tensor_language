"""Step 1: fit the 6-way next-token-category ridge probe on the residual AFTER block 3
(exploration data: cooc rows 0-2400). Save probe weights, decision axes d_k, geometry stats.

NAME under test: 'd_k carries evidence that the NEXT token is category k.'
"""
import json, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_category')
from common import (m, D, V, CAT, CATNAMES, COOC, forward, batches, oom_retry,
                    T_CTX, OUT, DEV)

TRAIN_ROWS = range(0, 1800, 4)      # 450 rows
VAL_ROWS = range(1800, 2400, 4)     # 150 rows (exploration val, not the held audit set)

def gather(rows):
    X, Y = [], []
    for idx in batches(COOC, rows):
        _, got = oom_retry(forward, idx[:, :-1], collect=('blk3',))
        X.append(got['blk3'].cpu())
        Y.append(CAT[idx[:, 1:].reshape(-1)].cpu())
    return torch.cat(X), torch.cat(Y)

print("gathering blk3 residuals (train)...", flush=True)
Xtr, Ytr = gather(TRAIN_ROWS)
print("gathering blk3 residuals (val)...", flush=True)
Xva, Yva = gather(VAL_ROWS)
print(f"train {Xtr.shape} val {Xva.shape}", flush=True)

# ridge to one-hot, with bias, on GPU in double
def fit(X, Y, lam=50.0):
    Xd = torch.cat([X, torch.ones(X.shape[0], 1)], 1).double().to(DEV)
    Yoh = F.one_hot(Y, 6).double().to(DEV)
    W = torch.linalg.solve(Xd.T @ Xd + lam*torch.eye(Xd.shape[1], device=DEV, dtype=torch.double),
                           Xd.T @ Yoh)
    return W  # (D+1, 6)

W = fit(Xtr, Ytr)

def acc(X, Y, W):
    Xd = torch.cat([X, torch.ones(X.shape[0], 1)], 1).double().to(DEV)
    pred = (Xd @ W).argmax(1).cpu()
    return float((pred == Y).float().mean())

acc_tr, acc_va = acc(Xtr, Ytr, W), acc(Xva, Yva, W)
maj = torch.bincount(Yva, minlength=6).max().item() / Yva.numel()
per_cat_recall = {}
Xd = torch.cat([Xva, torch.ones(Xva.shape[0], 1)], 1).double().to(DEV)
pred = (Xd @ W).argmax(1).cpu()
for c in range(6):
    msk = Yva == c
    per_cat_recall[CATNAMES[c]] = round(float((pred[msk] == c).float().mean()), 4) if msk.any() else None
print(f"probe blk3: train acc {acc_tr:.4f} val acc {acc_va:.4f} (majority {maj:.4f})", flush=True)
print("per-cat recall:", per_cat_recall, flush=True)

# directions
Wd = W[:D].float()                       # (D,6) raw columns
d_cent = Wd - Wd.mean(1, keepdim=True)   # decision axes (evidence for k vs mean)
d_unit = d_cent / d_cent.norm(dim=0, keepdim=True)
raw_unit = Wd / Wd.norm(dim=0, keepdim=True)

# geometry
cos_cent = (d_unit.T @ d_unit).cpu()
sv_raw = torch.linalg.svdvals(Wd.double()).cpu()
sv_cent = torch.linalg.svdvals(d_cent.double()).cpu()

# residual norms at blk3 (for steering alpha scale)
rn = Xtr.norm(dim=1)
r_med = rn.median().item()

torch.save({'W': W.cpu(), 'Wd': Wd.cpu(), 'd_cent': d_cent.cpu(), 'd_unit': d_unit.cpu(),
            'raw_unit': raw_unit.cpu(), 'r_med': r_med, 'catnames': CATNAMES},
           f'{OUT}/probe_blk3.pt')

res = {'train_rows': [0, 1800, 4], 'val_rows': [1800, 2400, 4],
       'n_train': Xtr.shape[0], 'n_val': Xva.shape[0],
       'acc_train': round(acc_tr, 4), 'acc_val': round(acc_va, 4), 'majority_val': round(maj, 4),
       'per_cat_recall_val': per_cat_recall,
       'class_dist_val': [round(c/Yva.numel(), 4) for c in torch.bincount(Yva, minlength=6).tolist()],
       'residual_norm_blk3': {'median': round(r_med, 2),
                              'p10': round(rn.quantile(0.1).item(), 2),
                              'p90': round(rn.quantile(0.9).item(), 2)},
       'dir_norms_raw': [round(x, 5) for x in Wd.norm(dim=0).tolist()],
       'dir_norms_cent': [round(x, 5) for x in d_cent.norm(dim=0).tolist()],
       'cos_centered_dirs': [[round(cos_cent[i, j].item(), 3) for j in range(6)] for i in range(6)],
       'svals_raw_W': [round(x, 5) for x in sv_raw.tolist()],
       'svals_centered': [round(x, 5) for x in sv_cent.tolist()]}
json.dump(res, open(f'{OUT}/s1_probe.json', 'w'), indent=2)
print(json.dumps(res, indent=2), flush=True)
print("S1 DONE", flush=True)
