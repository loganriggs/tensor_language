"""Sanity checks for bq_common: the metric, the gauge null, the SBD routine.
Run before trusting any experiment result. Prints PASS/FAIL per check."""

import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bq_common import (init_params, forward, forward_Q, interaction, lam_inner, block_mass,
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
check('gauge refactor destroys hidden basis', pg['L'].shape[0] != p['L'].shape[0],
      f"h {p['L'].shape[0]} -> {pg['L'].shape[0]}")

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

# --------------------------------------------------------------------------
# The routines below carry most of the program's headline claims and had no
# checks at all until Reviewer 2 pointed it out. Each is asserted against a case
# whose answer is known independently of the routine.
# --------------------------------------------------------------------------
from bq_common import (jade, partition_from_coupling, reorder_by_partition,
                       fit_cp, fit_dictionary, lam_cos, sbd_jade)

# 9. JADE recovers a planted block structure from a hidden basis
d, sizes_true = 10, [4, 3, 3]
blocks, Qs = [(0, 4), (4, 7), (7, 10)], []
for _ in range(6):
    M = torch.zeros(d, d)
    for a_, b_ in blocks:
        sub = torch.randn(b_ - a_, b_ - a_)
        M[a_:b_, a_:b_] = sub + sub.T
    Qs.append(M)
Qs = torch.stack(Qs)
W = torch.linalg.qr(torch.randn(d, d))[0]
Qh = torch.einsum('ab,ibc,dc->iad', W, Qs, W)
Pj, sz, info = sbd_jade(Qh, tol=1e-8, sweeps=40)
check('JADE recovers planted blocks through a hidden basis',
      sorted(sz) == sorted(sizes_true) and info['in_block_mass'] > 1 - 1e-8,
      f"sizes {sz} (true {sizes_true}) in-block {info['in_block_mass']:.8f}")

# 10. partition_from_coupling: in-block mass really is monotone in the threshold
_, T, _ = jade(Qh, sweeps=40), None, None
Pj2, Tj, _ = jade(Qh, sweeps=40)
masses = []
for tol in (1e-8, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.4):
    parts = partition_from_coupling(Tj, tol=tol)
    Pr, s_ = reorder_by_partition(Pj2, parts)
    masses.append(block_mass(Qh, Pr, s_)[0])
check('partition tolerance: in-block mass is monotone non-increasing',
      all(masses[i] >= masses[i + 1] - 1e-12 for i in range(len(masses) - 1)),
      ' '.join(f'{v:.4f}' for v in masses))

# 11. fit_cp recovers planted components well inside the identifiable regime
K, dc, mc = 4, 10, 6
gg = torch.Generator().manual_seed(5)
av = torch.randn(K, dc, generator=gg)
bv = torch.randn(K, dc, generator=gg)
cv = torch.randn(mc, K, generator=gg)
Sp = 0.5 * (torch.einsum('ka,kb->kab', av, bv) + torch.einsum('kb,ka->kab', av, bv))
Qcp = torch.einsum('ik,kab->iab', cv, Sp)
ph, err = fit_cp(Qcp, K, steps=4000, lr=3e-2, seed=0)
Sh = 0.5 * (torch.einsum('ka,kb->kab', ph['L'], ph['R'])
            + torch.einsum('kb,ka->kab', ph['L'], ph['R']))
Sn = Sh / Sh.flatten(1).norm(dim=1)[:, None, None]
Spn = Sp / Sp.flatten(1).norm(dim=1)[:, None, None]
best = [max(abs(float((Sn[i] * Spn[j]).sum())) for i in range(K)) for j in range(K)]
check('fit_cp recovers planted components (K=4, d=10, m=6)',
      err < 1e-8 and min(best) > 0.99, f'fit err {err:.2e} worst matched cos {min(best):.4f}')

# 12. fit_dictionary: recovers planted atoms from SPARSE codes, and correctly
#     fails on DENSE ones. The second half is not a defect — THEORY.md T7 says a
#     span is all that fit determines, so with every reader using every atom there
#     is nothing to break the rotation degeneracy and no method could succeed.
na, nr = 5, 20   # 5 atoms, 2 per reader: enough distinct sparsity patterns to identify
atoms = []
for _ in range(na):
    A_ = torch.randn(8, 8, generator=gg)
    A_ = 0.5 * (A_ + A_.T)
    atoms.append(A_ / A_.norm())
Tn = torch.stack([a / a.norm() for a in atoms])


def dict_recovery(codes, tag, expect_recovery):
    Qd = torch.einsum('ua,aij->uij', codes.to(Tn), Tn)
    Ad, Cd, derr = fit_dictionary(Qd, na, steps=4000, lr=3e-2, l1=1e-2, seed=0)
    An = Ad / Ad.flatten(1).norm(dim=1)[:, None, None]
    bestd = [max(abs(float((An[i] * Tn[j]).sum())) for i in range(na)) for j in range(na)]
    got = min(bestd) > 0.9
    check(f'fit_dictionary, {tag}: {"recovers" if expect_recovery else "correctly cannot recover"}',
          derr < 1e-3 and got == expect_recovery,
          f'err {derr:.2e} worst matched cos {min(bestd):.4f}')


sparse = torch.zeros(nr, na)
for u in range(nr):
    pick = torch.randperm(na, generator=gg)[:2]
    sparse[u, pick] = torch.randn(2, generator=gg)
dict_recovery(sparse, 'sparse codes (2 of 5 atoms per reader)', True)
dict_recovery(torch.randn(nr, na, generator=gg), 'dense codes (every reader uses every atom)', False)

# 13. canonicalise is function-preserving and idempotent
import importlib
a2m = importlib.import_module('a2_modular')
xa, ya, _, _ = a2m.all_pairs()
bas = a2m.identifiable_basis(xa)
Qtest = interaction(init_params(2 * a2m.P, 16, a2m.P, seed=3, device=xa.device))
Qtest = Qtest.to(torch.get_default_dtype())
Qc, frac = a2m.canonicalise(Qtest, bas)
fn_diff = float((forward_Q(Qc, xa) - forward_Q(Qtest, xa)).abs().max())
Qcc, _ = a2m.canonicalise(Qc, bas)
check('canonicalise preserves the function on the data', fn_diff, 0.0, tol=1e-9) if False else None
ok.append(fn_diff < 1e-10)
print(f'{"PASS" if fn_diff < 1e-10 else "FAIL"}  canonicalise preserves the function on '
      f'the data   max |dy| = {fn_diff:.2e}   (kept mass fraction {frac:.4f})')
idem = float((Qcc - Qc).abs().max())
ok.append(idem < 1e-12)
print(f'{"PASS" if idem < 1e-12 else "FAIL"}  canonicalise is idempotent   max |dQ| = {idem:.2e}')
# and the chance baseline the review asked for
gr = torch.Generator().manual_seed(11)
Qr_ = torch.randn(a2m.P, 2 * a2m.P, 2 * a2m.P, generator=gr).to(Qtest)
Qr_ = 0.5 * (Qr_ + Qr_.transpose(1, 2))
_, frac_rand = a2m.canonicalise(Qr_, bas)
print(f'      chance identifiable fraction (random symmetric forms): {frac_rand:.4f}'
      f'   [529/1081 = {529/1081:.4f}]')
ok.append(abs(frac_rand - 529 / 1081) < 0.02)

print(f'\n{sum(ok)}/{len(ok)} checks passed')
sys.exit(0 if all(ok) else 1)
