"""e11: Logan's positive-control ladder + LEARNED ordering for the tensor nets.

Ladder (each rung has a known answer; optimizer must hit it before we blame the
representation):
  L0  single core (65536, 1024), gradient fit          -> FVU ~ 0
  L1  base (2, 32768), FULL ranks (r1=2, r2=1024)      -> FVU = 0 exactly
      (the first cut has rank <= 2 trivially: factorizing costs NOTHING)
  L1' base (2, 32768), r2 capped at 256, TT-SVD + grad -> must equal SVD-256
      (the last bond caps the GLOBAL rank: every recon row lies in the
       r_last-dim rowspace of the final core, so FVU(TT, rmax) >= FVU(SVD, rmax)
       for ANY ordering — this is the floor the 16^4 TT is fighting.)

Learned ordering (16^4, rmax=256): alternate
  [TT-SVD refit under current permutation] <-> [batched improving-swap search:
  sample disjoint position pairs, swap the tokens if it lowers total error]
from (a) the semantic k-means init and (b) random init. The gap to the SVD-256
floor (0.556) that ordering can recover is the measurement.
"""

import json
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
import e6_embedding_objects as e6
from e8_tensor_train import E_C, VPAD, make_orderings

DEV = 'cuda'
BASE = '/workspace/tensor_language/basis_aligned'
V, D_MODEL = e6.E.shape
torch.manual_seed(0)
results = {}


def padded_fvu(R, T):
    return float(((R - T) ** 2).sum() / (T ** 2).sum())


def tt_svd_generic(T_mat, dims, rmaxes):
    """TT-SVD for arbitrary digit dims; rmaxes[i] caps bond i."""
    cores = []
    W = T_mat.reshape(dims[0], -1)
    r_prev = 1
    for i, m in enumerate(dims):
        M, N = W.shape
        r = min(rmaxes[i], M, N)
        if M <= N:
            G = W @ W.T
            evals, evecs = torch.linalg.eigh(G.double())
            U = evecs[:, evals.argsort(descending=True)[:r]].float()
        else:  # tall unfolding: use the small-side Gram
            G = W.T @ W
            evals, evecs = torch.linalg.eigh(G.double())
            order = evals.argsort(descending=True)[:r]
            Vr = evecs[:, order].float()
            sig = evals[order].clamp(min=1e-30).sqrt().float()
            U = (W @ Vr) / sig[None, :]
        cores.append(U.reshape(r_prev, m, r))
        if i < len(dims) - 1:
            W = (U.T @ W).reshape(r * dims[i + 1], -1)
        else:
            W = U.T @ W
        r_prev = r
    cores.append(W)
    R = cores[0].reshape(-1, cores[0].shape[-1])
    for c in cores[1:len(dims)]:
        R = torch.einsum('ab,bcd->acd', R, c).reshape(R.shape[0] * c.shape[1], -1)
    return R @ cores[len(dims)], sum(c.numel() for c in cores)


# ---------------- L0: single core, gradient
print('L0: single full-size core, gradient fit (must reach ~0)')
T = E_C
X = torch.randn(VPAD, D_MODEL, device=DEV, requires_grad=True)
with torch.no_grad():
    X.mul_(0.01)
opt = torch.optim.Adam([X], lr=1e-2)
for step in range(1500):
    loss = ((X - T) ** 2).sum() / (T ** 2).sum()
    opt.zero_grad(); loss.backward(); opt.step()
results['L0_grad_fvu'] = loss.item()
print(f'  L0 gradient FVU: {loss.item():.2e}')
del X, opt
torch.cuda.empty_cache()

# ---------------- L1: (2, 32768) full ranks — exact
R, _ = tt_svd_generic(T, [2, 32768], [2, 1024])
results['L1_fullrank_ttsvd_fvu'] = padded_fvu(R, T)
print(f"L1 (2,32768) full-rank TT-SVD FVU: {results['L1_fullrank_ttsvd_fvu']:.2e} (must be ~0)")
del R; torch.cuda.empty_cache()

# ---------------- L1': (2, 32768) with r2=256 vs SVD-256 (the global-rank floor)
G = T.T @ T
evals, evecs = torch.linalg.eigh(G.double())
U = evecs[:, evals.argsort(descending=True)[:256]].float()
svd256_fvu = padded_fvu(T @ U @ U.T, T)
R, _ = tt_svd_generic(T, [2, 32768], [2, 256])
results['svd256_floor_fvu'] = svd256_fvu
results['L1_r256_ttsvd_fvu'] = padded_fvu(R, T)
print(f"SVD-256 floor FVU: {svd256_fvu:.4f}   L1' (2,32768,r2=256) TT-SVD: "
      f"{results['L1_r256_ttsvd_fvu']:.4f}  (must match)")
del R, G, evecs; torch.cuda.empty_cache()

# gradient version of L1'
c1 = torch.randn(2, 2, device=DEV, requires_grad=True)
c2 = (torch.randn(2, 32768, 256, device=DEV) * 0.02).requires_grad_(True)
c3 = (torch.randn(256, D_MODEL, device=DEV) * 0.02).requires_grad_(True)
opt = torch.optim.Adam([c1, c2, c3], lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000)
for step in range(4000):
    R = torch.einsum('ab,bcd->acd', c1, c2).reshape(VPAD, 256) @ c3
    loss = ((R - T) ** 2).sum() / (T ** 2).sum()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 1000 == 0:
        print(f"  L1' grad step {step} fvu {loss.item():.4f}", flush=True)
results['L1_r256_grad_fvu'] = loss.item()
print(f"L1' gradient FVU: {loss.item():.4f} (vs floor {svd256_fvu:.4f})")
del c1, c2, c3, R, opt; torch.cuda.empty_cache()

# ---------------- learned ordering at 16^4, rmax=256
def swap_round(perm, R, T_full, n_rounds=15):
    """Batched improving swaps. perm: positions->token rows of T_full (E_C)."""
    n_acc = 0
    for _ in range(n_rounds):
        err = ((T_full[perm] - R) ** 2).sum(1)          # cost at own position
        pair = torch.randperm(VPAD, device=DEV)
        p, q = pair[:VPAD // 2], pair[VPAD // 2:]
        cross_pq = ((T_full[perm[p]] - R[q]) ** 2).sum(1)
        cross_qp = ((T_full[perm[q]] - R[p]) ** 2).sum(1)
        delta = cross_pq + cross_qp - err[p] - err[q]
        acc = delta < 0
        pa, qa = p[acc], q[acc]
        perm[pa], perm[qa] = perm[qa].clone(), perm[pa].clone()
        n_acc += int(acc.sum())
    return perm, n_acc


orderings = {k: v.to(DEV) for k, v in
             torch.load(f'{BASE}/e8_orderings.pt').items()}
results['learned_ordering'] = {}
for init in ['semantic', 'random']:
    perm = orderings[init].clone()
    traj = []
    for it in range(10):
        R, params = tt_svd_generic(E_C[perm], [16, 16, 16, 16], [16, 256, 256, 256])
        fvu = padded_fvu(R, E_C[perm])
        traj.append(fvu)
        perm, n_acc = swap_round(perm, R, E_C)
        print(f'  learned({init}) iter {it}: fvu {fvu:.4f}  swaps {n_acc}', flush=True)
        del R; torch.cuda.empty_cache()
    R, _ = tt_svd_generic(E_C[perm], [16, 16, 16, 16], [16, 256, 256, 256])
    traj.append(padded_fvu(R, E_C[perm]))
    results['learned_ordering'][init] = {'trajectory': traj, 'final_fvu': traj[-1]}
    torch.save(perm.cpu(), f'{BASE}/e11_learned_perm_{init}.pt')
    print(f'learned ordering from {init}: {traj[0]:.4f} -> {traj[-1]:.4f} '
          f'(floor {svd256_fvu:.4f})')
    del R; torch.cuda.empty_cache()

with open(f'{BASE}/e11_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print('e11 done')
