"""Step 5: PERSISTENCE. Does the block-3 category code survive to later layers?
- Transfer: blk3-fit probe applied to blk4/blk8/blk12/blk16 residuals (held FineWeb),
  raw and RMS-matched (residual rescaled to blk3 median norm).
- Ceiling: fresh probes fit at each depth (cooc exploration) evaluated on held FineWeb.
- Geometry: principal angles between the blk3 probe subspace and each fresh-probe subspace.
Consumed/overwritten = transfer acc collapses while fresh acc stays high.
"""
import json, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_category')
from common import (m, D, CAT, CATNAMES, COOC, FINEWEB, HELD_ROWS, forward, batches,
                    oom_retry, load_probe, OUT, DEV)

DEPTHS = ['blk3', 'blk4', 'blk8', 'blk12', 'blk16']
TRAIN_ROWS = range(0, 1800, 4)   # same exploration rows as s1

def gather(data, rows):
    X = {d: [] for d in DEPTHS}; Y = []
    for idx in batches(data, rows):
        _, got = oom_retry(forward, idx[:, :-1], collect=DEPTHS)
        for d in DEPTHS: X[d].append(got[d].cpu())
        Y.append(CAT[idx[:, 1:].reshape(-1)].cpu())
    return {d: torch.cat(X[d]) for d in DEPTHS}, torch.cat(Y)

print("gather cooc train...", flush=True)
Xtr, Ytr = gather(COOC, TRAIN_ROWS)
print("gather held FineWeb...", flush=True)
Xte, Yte = gather(FINEWEB, HELD_ROWS)
maj = torch.bincount(Yte, minlength=6).max().item() / Yte.numel()

def fit(X, Y, lam=50.0):
    Xd = torch.cat([X, torch.ones(X.shape[0], 1)], 1).double().to(DEV)
    Yoh = F.one_hot(Y, 6).double().to(DEV)
    W = torch.linalg.solve(Xd.T @ Xd + lam*torch.eye(Xd.shape[1], device=DEV, dtype=torch.double), Xd.T @ Yoh)
    del Xd; torch.cuda.empty_cache()
    return W

def acc(X, Y, W):
    Xd = torch.cat([X, torch.ones(X.shape[0], 1)], 1).double().to(DEV)
    a = float(((Xd @ W).argmax(1).cpu() == Y).float().mean())
    del Xd; torch.cuda.empty_cache()
    return a

P = load_probe()
W3 = P['W'].to(DEV)          # blk3-fit (on same cooc rows)
r_med3 = P['r_med']

res = {'majority_held': round(maj, 4), 'depths': {}}
fresh_W = {}
for d in DEPTHS:
    fresh_W[d] = fit(Xtr[d], Ytr)
    rn = Xte[d].norm(dim=1); scale = r_med3 / rn.median().item()
    a_transfer = acc(Xte[d], Yte, W3)
    a_transfer_rms = acc(Xte[d]*scale, Yte, W3)
    a_fresh = acc(Xte[d], Yte, fresh_W[d])
    res['depths'][d] = {'transfer_raw': round(a_transfer, 4),
                        'transfer_rms_matched': round(a_transfer_rms, 4),
                        'fresh_fit': round(a_fresh, 4),
                        'median_norm': round(rn.median().item(), 1),
                        'rms_scale_applied': round(scale, 3)}
    print(f"{d}: transfer {a_transfer:.4f} | rms-matched {a_transfer_rms:.4f} | fresh {a_fresh:.4f} "
          f"(norm {rn.median().item():.0f})", flush=True)

# principal angles between blk3 centered probe subspace and fresh centered subspaces
def cent_basis(W):
    Wd = W[:D].float().cpu(); c = Wd - Wd.mean(1, keepdim=True)
    Q, _ = torch.linalg.qr(c.double()); return Q[:, :5]
B3 = cent_basis(P['W'])
res['principal_cosines_vs_blk3'] = {}
for d in DEPTHS:
    Bd = cent_basis(fresh_W[d].cpu())
    sv = torch.linalg.svdvals(B3.T @ Bd)
    res['principal_cosines_vs_blk3'][d] = [round(x, 3) for x in sv.tolist()]
    print(f"principal cosines blk3~{d}: {[round(x,3) for x in sv.tolist()]}", flush=True)

# how much of the natural blk3 residual lies along the code axes? (context for s3's near-zero
# ablation cost): norm of projection onto centered 5-dim basis vs full residual norm
proj = Xte['blk3'].double() @ B3          # (N,5)
frac = (proj.norm(dim=1) / Xte['blk3'].double().norm(dim=1))
res['blk3_residual_fraction_in_cent5_subspace'] = {
    'median': round(frac.median().item(), 5), 'p90': round(frac.quantile(0.9).item(), 5),
    'expected_random_5dim': round((5/D)**0.5, 5)}
print("fraction of residual norm in cent5 subspace:", res['blk3_residual_fraction_in_cent5_subspace'], flush=True)

json.dump(res, open(f'{OUT}/s5_persistence.json', 'w'), indent=2)
print("S5 DONE", flush=True)
