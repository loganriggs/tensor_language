"""PHASE 0 positive control for the query/key sparse-dictionary program (Logan 2026-07-21).

Before fitting any dictionary to the real layer-1 query/key factor tables, verify on PLANTED
structure with known ground truth that the solver family is selective:

  plant S (sparse):   X = b + sum of k_true=8 atoms drawn from a known dictionary of n_true=512
                      unit atoms in d=256, plus a small noise floor.
                      -> the SPARSE arms must WIN and must RECOVER the planted atoms.
  plant L (low-rank): X = b + U V^T of rank r_true=16, plus the same noise floor.
                      -> the DENSE LOW-RANK arm must WIN and the sparse arms must LOSE.

Every comparison is at MATCHED DESCRIPTION-LENGTH BITS (the program's hard rule): the dictionary
arms share (n=512, k=8), and the singular-value-decomposition rank is chosen so its bit count
matches. Selectivity must be 2/2 or nothing downstream is reported.

Arms: singular-value-decomposition (dense low-rank) | per-token top-k with a linear encoder |
per-token top-k with orthogonal-matching-pursuit and least-squares coefficients | batch-top-k |
matryoshka (nested prefixes).

No model is loaded; this is pure synthetic ground truth. Writes qk_sae_control.json.
"""
import json
import math
import sys
import torch

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_sae_control.json'

N, DIM = 4000, 256           # rows (a stand-in for the merged vocabulary), row dim = 2*head_dim
N_ATOMS, K = 512, 8          # dictionary budget shared by every sparse arm
N_TRUE, K_TRUE = 512, 8      # planted sparse structure
R_TRUE = 16                  # planted low-rank structure
NOISE = 0.02                 # noise floor, relative to signal scale


# ---------------------------------------------------------------- description length

def bits_dict(n_atoms, dim, nnz):
    """atoms + bias floats at 32 bits, then per nonzero: 32-bit coefficient + log2(n) index bits."""
    return 32 * (n_atoms * dim + dim) + nnz * (32 + math.log2(max(n_atoms, 2)))


def bits_svd(r, n_rows, n_cols):
    """frozen convention (mdl_accounting.dl_svd): 32 * r * (rows + cols + 1)."""
    return 32 * r * (n_rows + n_cols + 1)


def rank_for_bits(target, n_rows, n_cols):
    return max(1, int(target / (32 * (n_rows + n_cols + 1))))


# ---------------------------------------------------------------- plants

def make_sparse_plant(seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dtrue = torch.randn(N_TRUE, DIM, generator=g)
    Dtrue = Dtrue / Dtrue.norm(dim=1, keepdim=True)
    idx = torch.stack([torch.randperm(N_TRUE, generator=g)[:K_TRUE] for _ in range(N)])
    coef = torch.randn(N, K_TRUE, generator=g)
    X = (coef.unsqueeze(-1) * Dtrue[idx]).sum(1)
    b = torch.randn(DIM, generator=g) * 0.1
    X = X + b
    X = X + NOISE * X.std() * torch.randn(N, DIM, generator=g)
    return X.to(DEV), Dtrue.to(DEV)


def make_lowrank_plant(seed=1):
    g = torch.Generator(device='cpu').manual_seed(seed)
    U = torch.randn(N, R_TRUE, generator=g)
    Vt = torch.randn(R_TRUE, DIM, generator=g)
    X = (U @ Vt) / math.sqrt(R_TRUE)
    b = torch.randn(DIM, generator=g) * 0.1
    X = X + b
    X = X + NOISE * X.std() * torch.randn(N, DIM, generator=g)
    return X.to(DEV), None


# ---------------------------------------------------------------- fits

def fvu(Xhat, X):
    return ((Xhat - X) ** 2).sum().item() / ((X - X.mean(0)) ** 2).sum().item()


@torch.no_grad()
def arm_svd(X, r):
    b = X.mean(0)
    U, S, Vh = torch.linalg.svd(X - b, full_matrices=False)
    return b + (U[:, :r] * S[:r]) @ Vh[:r]


def train_dict(X, n, k, mode='token', steps=3000, batch=2048, lr=3e-3, seed=0, nested=None):
    """Signed magnitude top-k dictionary (ov_sparse/e9 recipe) with dead-atom reinit.
    mode in {'token','batch'}; nested = list of prefixes for the matryoshka loss."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = X[torch.randperm(len(X), generator=g)[:n].to(X.device)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = X.mean(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=lr)
    fired = torch.zeros(n, device=X.device)
    losses = []
    for step in range(steps):
        x = X[torch.randint(0, len(X), (min(batch, len(X)),), device=X.device)]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (x - b) @ We.T
        if nested is not None:                       # matryoshka: sum reconstruction over prefixes
            loss = 0.0
            for P in nested:
                kp = max(1, int(round(k * P / n)))
                zp = z[:, :P]
                vals, idx = zp.abs().topk(min(kp, P), dim=1)
                coeff = torch.gather(zp, 1, idx)
                xhat = b + (coeff.unsqueeze(-1) * Dn[:P][idx]).sum(1)
                loss = loss + ((xhat - x) ** 2).mean()
        elif mode == 'token':
            vals, idx = z.abs().topk(k, dim=1)
            coeff = torch.gather(z, 1, idx)
            xhat = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
            fired.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=X.device))
            loss = ((xhat - x) ** 2).mean()
        else:                                        # batch-top-k inside the minibatch
            flat = z.abs().reshape(-1)
            thresh = flat.topk(k * len(x)).values.min()
            zc = z * (z.abs() >= thresh)
            fired.index_add_(0, (zc != 0).nonzero()[:, 1], torch.ones((zc != 0).sum(), device=X.device))
            loss = ((b + zc @ Dn - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if (step + 1) % 500 == 0 and nested is None:  # dead-atom reinit (e7/e9 recipe)
            dead = (fired == 0).nonzero().squeeze(1)
            if len(dead):
                with torch.no_grad():
                    Dn_ = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    zc_ = (X - b) @ We.T
                    v_, i_ = zc_.abs().topk(k, dim=1)
                    rec = b + (torch.gather(zc_, 1, i_).unsqueeze(-1) * Dn_[i_]).sum(1)
                    worst = ((rec - X) ** 2).sum(1).topk(len(dead)).indices
                    Dm[dead] = X[worst] / X[worst].norm(dim=1, keepdim=True).clamp(min=1e-8)
                    We[dead] = Dm[dead]
            fired.zero_()
    Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
    return Dn, b.detach(), We.detach(), losses


@torch.no_grad()
def encode_token(X, Dn, b, We, k):
    z = (X - b) @ We.T
    vals, idx = z.abs().topk(k, dim=1)
    coeff = torch.gather(z, 1, idx)
    return b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1), k * len(X)


@torch.no_grad()
def encode_batch(X, Dn, b, We, kavg):
    z = (X - b) @ We.T
    thresh = z.abs().reshape(-1).topk(kavg * len(X)).values.min()
    zc = z * (z.abs() >= thresh)
    return b + zc @ Dn, int((zc != 0).sum())


@torch.no_grad()
def encode_omp(X, Dn, b, k):
    """Greedy orthogonal matching pursuit with a least-squares refit at every step (ov_omp_batch)."""
    Y = X - b
    n = Dn.shape[0]
    r = Y.clone()
    sup = torch.full((len(X), k), -1, device=X.device, dtype=torch.long)
    chosen = torch.zeros(len(X), n, dtype=torch.bool, device=X.device)
    recon = torch.zeros_like(Y)
    for s in range(k):
        corr = (r @ Dn.T).abs()
        corr[chosen] = -1
        a = corr.argmax(1)
        sup[:, s] = a
        chosen[torch.arange(len(X), device=X.device), a] = True
        Ds = Dn[sup[:, :s + 1]]
        G = torch.bmm(Ds, Ds.transpose(1, 2))
        rhs = torch.bmm(Ds, Y.unsqueeze(-1)).squeeze(-1)
        c = torch.linalg.solve(G + 1e-6 * torch.eye(s + 1, device=X.device), rhs)
        recon = torch.bmm(c.unsqueeze(1), Ds).squeeze(1)
        r = Y - recon
    return b + recon, k * len(X)


@torch.no_grad()
def mmcs(Dn, Dtrue):
    """mean max cosine similarity of planted atoms to their best learned atom (recovery rate)."""
    sim = (Dtrue @ Dn.T).abs()
    return float(sim.max(1).values.mean())


# ---------------------------------------------------------------- run

TARGET_BITS = bits_dict(N_ATOMS, DIM, K * N)
R_MATCH = rank_for_bits(TARGET_BITS, N, DIM)
print(f'matched budget: dict(n={N_ATOMS},k={K}) = {TARGET_BITS/1e6:.2f} Mbit '
      f'-> svd rank {R_MATCH} = {bits_svd(R_MATCH, N, DIM)/1e6:.2f} Mbit', flush=True)

res = {'config': {'N': N, 'dim': DIM, 'n_atoms': N_ATOMS, 'k': K, 'n_true': N_TRUE,
                  'k_true': K_TRUE, 'r_true': R_TRUE, 'noise': NOISE,
                  'target_Mbits': round(TARGET_BITS / 1e6, 3), 'svd_rank_matched': R_MATCH},
       'plants': {}}

for pname, maker in [('sparse', make_sparse_plant), ('lowrank', make_lowrank_plant)]:
    X, Dtrue = maker()
    row = {}
    row['svd'] = round(fvu(arm_svd(X, R_MATCH), X), 5)

    Dn, b, We, losses = train_dict(X, N_ATOMS, K, mode='token', seed=0)
    row['dict_token_linear'] = round(fvu(encode_token(X, Dn, b, We, K)[0], X), 5)
    row['dict_token_omp_ls'] = round(fvu(encode_omp(X, Dn, b, K)[0], X), 5)
    row['train_loss_first'] = round(sum(losses[:50]) / 50, 6)
    row['train_loss_last'] = round(sum(losses[-50:]) / 50, 6)
    if Dtrue is not None:
        row['atom_recovery_mmcs'] = round(mmcs(Dn, Dtrue), 4)

    Db, bb, Web, _ = train_dict(X, N_ATOMS, K, mode='batch', seed=0)
    xh, nnz = encode_batch(X, Db, bb, Web, K)
    row['dict_batch_topk'] = round(fvu(xh, X), 5)
    row['dict_batch_nnz'] = int(nnz)

    Dm_, bm_, Wem_, _ = train_dict(X, N_ATOMS, K, mode='token', seed=0,
                                   nested=[N_ATOMS // 8, N_ATOMS // 2, N_ATOMS])
    row['dict_matryoshka'] = round(fvu(encode_token(X, Dm_, bm_, Wem_, K)[0], X), 5)

    best_sparse = min(row['dict_token_linear'], row['dict_token_omp_ls'],
                      row['dict_batch_topk'], row['dict_matryoshka'])
    row['best_sparse'] = round(best_sparse, 5)
    row['sparse_beats_svd'] = bool(best_sparse < row['svd'])
    res['plants'][pname] = row
    print(f'\nplant {pname}: (all at ~{TARGET_BITS/1e6:.2f} Mbit)', flush=True)
    for kk in ['svd', 'dict_token_linear', 'dict_token_omp_ls', 'dict_batch_topk',
               'dict_matryoshka', 'atom_recovery_mmcs']:
        if kk in row:
            print(f'    {kk:22s} {row[kk]}', flush=True)

sel_sparse = res['plants']['sparse']['sparse_beats_svd']          # sparse plant -> sparse must win
sel_lowrank = not res['plants']['lowrank']['sparse_beats_svd']    # low-rank plant -> svd must win
res['selectivity'] = {'sparse_plant_sparse_wins': sel_sparse,
                      'lowrank_plant_svd_wins': sel_lowrank,
                      'passes': int(sel_sparse) + int(sel_lowrank), 'of': 2}
print(f'\nSELECTIVITY {int(sel_sparse) + int(sel_lowrank)}/2  '
      f'(sparse-plant sparse wins: {sel_sparse}; lowrank-plant svd wins: {sel_lowrank})', flush=True)
json.dump(res, open(OUT, 'w'), indent=2)
print(f'wrote {OUT}', flush=True)
