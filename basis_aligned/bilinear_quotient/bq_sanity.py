"""Sanity checks for bq_common: the metric, the gauge null, the SBD routine.
Run before trusting any experiment result. Prints PASS/FAIL per check."""

import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bq_common import (init_params, forward, forward_Q, interaction, lam_inner,
                       lam_relerr, gauge_refactor, sbd, block_mass, whiten,
                       effective_rank, row_space_kernel, reader_moment)

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
ok = []


def check(name, cond, detail=''):
    ok.append(bool(cond))
    print(f'{"PASS" if cond else "FAIL"}  {name}   {detail}')


# 1. interaction form reproduces the layer
p = init_params(6, 10, 3, seed=0)
x = torch.randn(2048, 6)
check('Q reproduces forward', (forward(p, x) - forward_Q(interaction(p), x)).abs().max() < 1e-10,
      f'maxdiff {(forward(p, x) - forward_Q(interaction(p), x)).abs().max():.2e}')

# 2. Λ inner product == Monte-Carlo E[y·ŷ] under N(0,G)
p2 = init_params(6, 10, 3, seed=1)
A, B = interaction(p), interaction(p2)
Graw = torch.randn(6, 8)
G = Graw @ Graw.T / 8
Ls = torch.linalg.cholesky(G)
xs = torch.randn(400000, 6) @ Ls.T
mc = (forward_Q(A, xs) * forward_Q(B, xs)).sum(1).mean()
an = lam_inner(A, B, G)
check('Λ inner == E[y·ŷ]', abs(mc - an) / abs(an) < 0.02, f'mc {mc:.4f} analytic {an:.4f}')

# 3. whitening: Λ(G) on Q equals Λ(I) on whitened Q
check('whiten consistency', abs(lam_inner(A, B, G) - lam_inner(whiten(A, G), whiten(B, G))) /
      abs(lam_inner(A, B, G)) < 1e-10)

# 4. gauge refactor is exactly function preserving
pg, res = gauge_refactor(p, seed=3)
check('gauge refactor exact', res < 1e-18, f'rel Λ residual {res:.2e}')
check('gauge refactor destroys hidden basis',
      pg['L'].shape[0] != p['L'].shape[0] or
      lam_relerr(interaction(p), interaction(p)) == 0.0, f"h {p['L'].shape[0]} -> {pg['L'].shape[0]}")

# 5. SBD recovers a planted block structure
d, sizes_true = 9, [3, 2, 4]
Qs = []
blocks = [(0, 3), (3, 5), (5, 9)]
for _ in range(6):
    M = torch.zeros(d, d)
    for a, b in blocks:
        sub = torch.randn(b - a, b - a)
        M[a:b, a:b] = sub + sub.T
    Qs.append(M)
Qs = torch.stack(Qs)
Wr = torch.linalg.qr(torch.randn(d, d))[0]           # hide the block basis
Qh = torch.einsum('ab,ibc,dc->iad', Wr, Qs, Wr)
P, sizes, info = sbd(Qh)
mass, _ = block_mass(Qh, P, sizes)
check('SBD recovers planted blocks', sorted(sizes) == sorted(sizes_true) and mass > 1 - 1e-8,
      f'sizes {sizes} (true {sizes_true}) in-block mass {mass:.6f} commutant dim {info["commutant_dim"]}')

# 6. SBD on a generic (irreducible) family finds one block
Qg = torch.randn(6, 7, 7)
Qg = Qg + Qg.transpose(1, 2)
Pg, sg, ig = sbd(Qg)
check('SBD on generic family: single block', sg == [7], f'sizes {sg} commutant dim {ig["commutant_dim"]}')

# 7. rank / kernel bookkeeping
p3 = init_params(5, 3, 4, seed=7)                      # h=3 < dim Sym² => rank ≤ 3
Q3 = interaction(p3)
r, s = effective_rank(Q3)
rows, ker, _ = row_space_kernel(Q3)
check('effective rank ≤ h', r <= 3, f'rank {r}')
check('kernel dim complements', rows.shape[0] + ker.shape[0] == 5 * 6 // 2,
      f'{rows.shape[0]} + {ker.shape[0]}')
# every kernel direction is Λ-orthogonal to every form
ip = torch.einsum('kab,iab->ki', ker, Q3).abs().max()
check('kernel ⟂ forms', ip < 1e-10, f'max |⟨ker,Q⟩| {ip:.2e}')

# 8. reader moment is PSD and trace-consistent
Qu = interaction(init_params(5, 8, 6, seed=11))
M = reader_moment(Qu)
ev = torch.linalg.eigvalsh(M)
check('reader moment PSD', ev.min() > -1e-10, f'min eig {ev.min():.2e}')
check('reader moment trace == Σ‖Q_u‖²_F', abs(M.trace() - (Qu ** 2).sum()) < 1e-9)

print(f'\n{sum(ok)}/{len(ok)} checks passed')
sys.exit(0 if all(ok) else 1)
