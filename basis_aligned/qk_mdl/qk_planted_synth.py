"""PLANTED-SYNTHETIC UNIT TESTS (spec section 8A, checks 1 and 3 + triple recovery).

DGP: m0=48 orthogonal true atoms in R^64, three attribute groups of 16 (COLOR/SHAPE/TEXTURE
style). Each of n=8192 synthetic tokens activates one atom per group with coefficients in
[0.5, 1.5]; the joint distribution over (color, shape) pairs is PLANTED non-uniform (block
structure), texture independent. E = Z D_true exactly (no noise).

Check 1 (planted-orthogonal recovery): our TopK dictionary (train_dict, m=48, k=3) must
recover D_true to permutation/sign — greedy cosine matching, mean max |cos| > 0.99 — and the
code-level third-moment core built from learned codes must reproduce the analytic one.
Check 3 (permutation null): independently permuting each feature's activation column across
tokens must collapse the planted off-diagonal triple structure.
Triple recovery: precision@K between the top-K off-diagonal entries of the matched learned
core and the ground-truth core.
"""
import sys
import torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from qk_sae_lib import train_dict, encode_token

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
n, d, m0, G = 8192, 64, 48, 3
gsize = m0 // G

D_true = torch.linalg.qr(torch.randn(d, d))[0][:m0].to(DEV)       # (48, 64) orthonormal rows

# planted joint: color c in 0..15, shape s: block-coupled (P high when same block of 4)
g = torch.Generator().manual_seed(1)
color = torch.randint(0, gsize, (n,), generator=g)
block = color // 4
shape = torch.where(torch.rand(n, generator=g) < 0.7,
                    block * 4 + torch.randint(0, 4, (n,), generator=g),
                    torch.randint(0, gsize, (n,), generator=g))
texture = torch.randint(0, gsize, (n,), generator=g)
Z = torch.zeros(n, m0, device=DEV)
coef = lambda: (0.5 + torch.rand(n, generator=g)).to(DEV)
Z[torch.arange(n), color.to(DEV)] = coef()
Z[torch.arange(n), gsize + shape.to(DEV)] = coef()
Z[torch.arange(n), 2 * gsize + texture.to(DEV)] = coef()
E = Z @ D_true                                                     # (n, 64), exact

# ---- Check 1: dictionary recovery ----
Dn, b, We = train_dict(E, m0, 3, seed=0, steps=12000)
C = (Dn @ D_true.T).abs()                                          # (m0, m0) |cos|
perm = torch.full((m0,), -1, dtype=torch.long)
used = torch.zeros(m0, dtype=torch.bool)
for _ in range(m0):                                                # greedy match
    idx = (C * ~used[None, :].to(DEV)).argmax()
    r, c = idx // m0, idx % m0
    if perm[r] == -1 and not used[c]:
        perm[r] = c
        used[c] = True
    C[r, :] = -1
mcc = float((Dn @ D_true.T).abs().max(1).values.mean())
frac9 = float(((Dn @ D_true.T).abs().max(1).values > 0.9).float().mean())
print(f'CHECK 1 dictionary recovery: mean max|cos| {mcc:.4f}, frac>0.9 {frac9:.3f} '
      f'-> {"PASS" if mcc > 0.99 else "FAIL"}', flush=True)

# codes in true-atom order
S = encode_token(E, Dn, b, We, 3)                                  # recon, not codes — recompute codes
z = (E - b) @ We.T
idx = z.abs().topk(3, dim=1).indices
S_codes = torch.zeros(n, m0, device=DEV)
S_codes.scatter_(1, idx, torch.gather(z, 1, idx))
inv = torch.empty(m0, dtype=torch.long)
inv[perm] = torch.arange(m0)
S_al = S_codes[:, inv.to(DEV)]                                     # align learned cols to true atoms
sign = torch.sign((S_al * Z).sum(0)).clamp(min=-1)
sign[sign == 0] = 1
S_al = S_al * sign[None, :]


def core(Scodes):
    """Dense third-moment core over features (m0 small enough to materialize)."""
    return torch.einsum('ta,tb,tc->abc', Scodes, Scodes, Scodes) / len(Scodes)


M_true = core(Z).cpu()
M_learn = core(S_al).cpu()
rel = float((M_learn - M_true).norm() / M_true.norm())
print(f'CHECK 1 core reproduction: rel Frobenius error {rel:.4f} '
      f'-> {"PASS" if rel < 0.05 else "FAIL"}', flush=True)

# ---- Triple recovery: top-K off-diagonal entries ----
def offdiag_topk(M, K=200):
    m = M.shape[0]
    a, bq, c = torch.meshgrid(torch.arange(m), torch.arange(m), torch.arange(m), indexing='ij')
    mask = (a < bq) & (bq < c)
    vals = M[mask]
    order = vals.abs().argsort(descending=True)[:K]
    flat = torch.stack([a[mask], bq[mask], c[mask]], 1)
    return {tuple(r.tolist()) for r in flat[order]}


top_t = offdiag_topk(M_true)
top_l = offdiag_topk(M_learn)
prec = len(top_t & top_l) / 200
print(f'TRIPLE RECOVERY: precision@200 {prec:.3f} -> {"PASS" if prec > 0.8 else "FAIL"}',
      flush=True)

# ---- Check 3: permutation null ----
gp = torch.Generator().manual_seed(2)
S_perm = S_al.clone()
for col in range(m0):
    S_perm[:, col] = S_perm[torch.randperm(n, generator=gp).to(DEV), col]
M_null = core(S_perm).cpu()
def offmass(M):
    m = M.shape[0]
    diag = sum(float(M[i, i, i] ** 2) for i in range(m))
    return float((M ** 2).sum()) - diag
ratio = offmass(M_null) / max(offmass(M_learn), 1e-12)
prec_null = len(top_t & offdiag_topk(M_null)) / 200
print(f'CHECK 3 permutation null: off-diagonal mass ratio null/real {ratio:.3f}, '
      f'null precision@200 {prec_null:.3f} -> '
      f'{"PASS" if ratio < 0.5 and prec_null < 0.2 else "FAIL"}', flush=True)
print('SYNTH DONE', flush=True)
