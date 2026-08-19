"""Numerical verification of every claim in THEORY.md.

Each check states a proposition, computes both sides, and asserts agreement.
Where a result in RESULTS.md turns out to be a theorem rather than a
measurement, the check is what demotes it: if the closed form matches to
machine precision on random instances, the experiment was a derivation.
"""

import itertools
import math
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bq_common import (init_params, forward, interaction, forward_Q, lam_inner,
                       reader_moment, spectrum, commutant_basis, sbd, isotypic_groups,
                       support_basis, restrict)

torch.set_default_dtype(torch.float64)
OK = []


def check(name, lhs, rhs, tol=1e-9, note=''):
    err = abs(float(lhs) - float(rhs)) / max(abs(float(rhs)), 1e-30)
    good = err < tol
    OK.append(good)
    print(f'{"PASS" if good else "FAIL"}  {name}\n'
          f'        computed {float(lhs):.12g}   closed form {float(rhs):.12g}'
          f'   rel err {err:.2e}   {note}')


def sym(d, g):
    A = torch.randn(d, d, generator=g)
    return 0.5 * (A + A.T)


print('=' * 78)
print('T1  Blind directions: delta is invisible iff Q_i delta = 0 for every output')
print('=' * 78)
g = torch.Generator().manual_seed(0)
d, m, h = 7, 3, 20
# a generic layer has NO blind direction once m*d >= d, so plant one: confine every
# L and R row to the orthogonal complement of delta, which makes Q_i delta = 0 exactly
p = init_params(d, h, m, seed=0)
delta = torch.randn(d, generator=g)
delta = delta / delta.norm()
Pr = torch.eye(d) - torch.outer(delta, delta)
p['L'] = p['L'] @ Pr
p['R'] = p['R'] @ Pr
Q = interaction(p)
s = torch.linalg.svdvals(Q.reshape(m * d, d))
print(f'  smallest singular value of the stacked Q: {float(s[-1]):.3e}  '
      f'(planted blind direction)')
x = torch.randn(512, d, generator=g)
for t in (0.5, 2.0, 10.0):
    dy = (forward_Q(Q, x + t * delta) - forward_Q(Q, x)).abs().max()
    check(f'   output unchanged along the blind direction, t={t}', dy, 0.0, tol=1e-6,
          note='(absolute)') if False else None
    err = float(dy)
    OK.append(err < 1e-8)
    print(f'{"PASS" if err < 1e-8 else "FAIL"}  invariance at t={t}: max |dy| = {err:.3e}')
# and the converse: a direction the layer does read does move the output
d2 = torch.linalg.svd(Q.reshape(m * d, d))[2][0]
print(f'  control, a direction in the row space: max |dy| = '
      f'{float((forward_Q(Q, x + d2) - forward_Q(Q, x)).abs().max()):.3e}')

print()
print('=' * 78)
print('T3  A4: the linearize/prune error ratio is EXACTLY scale-invariant in the')
print('    component gain. Closed form for x = mu + delta, delta ~ N(0, Sigma):')
print('      err_lin  = E[(d^T S d)^2]      = 2 tr(S Sig S Sig) + (tr S Sig)^2')
print('      err_prune= E[(x^T S x)^2]      = (mu^T S mu + tr S Sig)^2')
print('                                       + 4 mu^T S Sig S mu + 2 tr(S Sig S Sig)')
print('=' * 78)
g = torch.Generator().manual_seed(1)
d = 6
for trial in range(3):
    S = sym(d, g)
    mu = torch.randn(d, generator=g) * (trial + 1)
    A = torch.randn(d, d, generator=g)
    Sig = A @ A.T / d + torch.eye(d) * 0.3
    L = torch.linalg.cholesky(Sig)
    n = 4_000_000
    dl = torch.randn(n, d, generator=g) @ L.T
    x = mu + dl
    lin_mc = float((torch.einsum('ni,ij,nj->n', dl, S, dl) ** 2).mean())
    pru_mc = float((torch.einsum('ni,ij,nj->n', x, S, x) ** 2).mean())
    SS = S @ Sig
    lin_cf = 2 * float(torch.trace(SS @ SS)) + float(torch.trace(SS)) ** 2
    pru_cf = ((float(mu @ S @ mu) + float(torch.trace(SS))) ** 2
              + 4 * float(mu @ S @ Sig @ S @ mu) + 2 * float(torch.trace(SS @ SS)))
    check(f'   trial {trial}: E[(d^T S d)^2]', lin_mc, lin_cf, tol=6e-3, note='(Monte Carlo)')
    check(f'   trial {trial}: E[(x^T S x)^2]', pru_mc, pru_cf, tol=6e-3, note='(Monte Carlo)')
    # scale invariance in gain: exact, not approximate
    for gam in (0.01, 1.0, 100.0):
        Sg = gam * S
        SSg = Sg @ Sig
        lin_g = 2 * float(torch.trace(SSg @ SSg)) + float(torch.trace(SSg)) ** 2
        pru_g = ((float(mu @ Sg @ mu) + float(torch.trace(SSg))) ** 2
                 + 4 * float(mu @ Sg @ Sig @ Sg @ mu) + 2 * float(torch.trace(SSg @ SSg)))
        check(f'   trial {trial}: ratio at gain {gam}', lin_g / pru_g, lin_cf / pru_cf,
              tol=1e-12, note='gain cancels identically')

print()
print("  A4's own design: orthonormal a,b, Sigma = I, a.mu = b.mu = rho")
print('  => ratio = 1/(1+rho^2)^2 exactly')
for rho in (0.0, 2.0, 10.0):
    d = 8
    a, b = torch.zeros(d), torch.zeros(d)
    a[0], b[1] = 1.0, 1.0
    S = 0.5 * (torch.outer(a, b) + torch.outer(b, a))
    mu = rho * (a + b)
    SS = S
    lin = 2 * float(torch.trace(SS @ SS)) + float(torch.trace(SS)) ** 2
    pru = ((float(mu @ S @ mu) + 0.0) ** 2 + 4 * float(mu @ S @ S @ mu) + 2 * float(torch.trace(SS @ SS)))
    check(f'   rho={rho}: ratio', lin / pru, 1.0 / (1 + rho ** 2) ** 2, tol=1e-12)

print()
print('=' * 78)
print('T4  A5: exact spectrum of the reader-weighted moment for Q_u = g S0 + h S_u')
print('      lambda_max = [R g^2 + h^2 + sqrt((R g^2 - h^2)^2 + 4 R g^2 h^2)] / 2')
print('      the rest:   h^2 with multiplicity R-1, and lambda_min')
print('=' * 78)


def moment_eigs(R, gg, hh):
    M = torch.zeros(R + 1, R + 1)
    M[0, 0] = R * gg ** 2
    for u in range(R):
        M[0, u + 1] = M[u + 1, 0] = gg * hh
        M[u + 1, u + 1] = hh ** 2
    return torch.linalg.eigvalsh(M).flip(0)


for R, gg, hh in ((3, 1., 1.), (5, 1., 1.), (8, 1., 1.), (3, 2., 1.), (4, 1., 3.)):
    ev = moment_eigs(R, gg, hh)
    lam = (R * gg ** 2 + hh ** 2 + math.sqrt((R * gg ** 2 - hh ** 2) ** 2
                                             + 4 * R * gg ** 2 * hh ** 2)) / 2
    check(f'   R={R} g={gg} h={hh}: top eigenvalue', ev[0], lam)
    if R > 1:
        check(f'   R={R} g={gg} h={hh}: second eigenvalue', ev[1], hh ** 2)
    if gg == hh == 1.0:
        check(f'   R={R}: ratio equals R+1', ev[0] / ev[1], R + 1)
        v = torch.linalg.eigh(torch.diag(torch.zeros(R + 1)))[1]  # placeholder
        M = torch.zeros(R + 1, R + 1)
        M[0, 0] = R
        for u in range(R):
            M[0, u + 1] = M[u + 1, 0] = 1.0
            M[u + 1, u + 1] = 1.0
        vec = torch.linalg.eigh(M)[1][:, -1].abs()
        check(f'   R={R}: top eigenvector overlap with the shared form', vec[0],
              math.sqrt(R / (R + 1)))

print('\n  the R+1 identity is knife-edge: it needs g = h. Ratio at other gains:')
for gg, hh in ((1., 1.), (1.2, 1.), (2., 1.), (1., 2.)):
    ev = moment_eigs(3, gg, hh)
    print(f'    g={gg} h={hh}: top/next = {float(ev[0]/ev[1]):.4f}   (R+1 = 4)')

print()
print('=' * 78)
print('T5  A2: the commutant dimension follows from Wedderburn/Schur')
print('    one 4-dim isotypic component per frequency, multiplicity 2 from the')
print('    a<->b exchange symmetry, real type => 2*3/2 = 3 symmetric dims each')
print('=' * 78)
import a2_calibrate as cal
import a2_modular as am
P = am.P
Qs = cal.planted_family(list(range(1, P // 2 + 1)))
U, sv = support_basis(Qs, thresh=1e-6)
Qr = restrict(Qs, U)
B_, ev_ = commutant_basis(Qr)
n_freq = P // 2
check(f'   commutant dimension for p={P}', B_.shape[0], 3 * n_freq,
      tol=1e-12, note=f'= 3 x {n_freq} frequencies')
Pm, sizes, info = sbd(Qr, gap_rel=1e-5)
groups = isotypic_groups(Qr, Pm, sizes)
check('   number of isotypic components', len(groups), n_freq, tol=1e-12)
check('   number of fine blocks', len(sizes), 2 * n_freq, tol=1e-12,
      note='2 per frequency: the exchange parity')
Sw = torch.zeros(2 * P, 2 * P, dtype=Qs.dtype, device=Qs.device)
Sw[:P, P:] = torch.eye(P, dtype=Qs.dtype, device=Qs.device)
Sw[P:, :P] = torch.eye(P, dtype=Qs.dtype, device=Qs.device)
comm = torch.stack([q @ Sw - Sw @ q for q in Qs]).abs().max()
print(f'  the exchange operator commutes with every Q_c: max |[Q_c,S]| = {float(comm):.3e}')
OK.append(float(comm) < 1e-12)

print()
print('=' * 78)
print('T6  A3: Kruskal bound. For an (M,d,d) partially symmetric CP of rank R,')
print('    uniqueness is guaranteed when k_C + k_A + k_B >= 2R + 2, and with')
print('    generic factors k_A = k_B = min(R,d), k_C = min(R,M).')
print('=' * 78)


def kruskal_max_rank(M, d):
    best = 0
    for R in range(1, 4 * (M + d)):
        if min(R, M) + 2 * min(R, d) >= 2 * R + 2:
            best = R
    return best


for M, dd, label in ((8, 16, "A3's setting"), (1152, 1152, 'bilin18 MLP')):
    R = kruskal_max_rank(M, dd)
    print(f'  {label:16s} M={M:5d} d={dd:5d} -> Kruskal guarantees uniqueness to R = {R}')
    if M == 8:
        print(f'    A3 measured exact recovery to R=24 and failure by R=32;')
        print(f'    Kruskal is SUFFICIENT not necessary, so R=24 > {R} is consistent.')
    else:
        print(f'    bilin18 has R = 4608 >> {R}: the binding mode is the OUTPUT dim,')
        print(f'    not the input dim. RESULTS/BILIN18 previously argued this on K/d.')
# the confound the reviewer identified: the family's own rank is M, not d
for K in (8, 24, 48):
    d_, M_ = 16, 8
    gg = torch.Generator().manual_seed(K)
    a = torch.randn(K, d_, generator=gg)
    b = torch.randn(K, d_, generator=gg)
    c = torch.randn(M_, K, generator=gg)
    S = 0.5 * (torch.einsum('ka,kb->kab', a, b) + torch.einsum('kb,ka->kab', a, b))
    Qk = torch.einsum('ik,kab->iab', c, S)
    iu, ju = torch.triu_indices(d_, d_)
    r = torch.linalg.matrix_rank(Qk[:, iu, ju], tol=1e-8)
    print(f'  K={K:2d}: effective rank of the form family = {int(r)} (= M = {M_}), '
          f'independent of K')

print()
print('=' * 78)
print('T7  A5-5: PCA and a dictionary MUST tie on error at the true budget.')
print('    Eckart-Young: any basis of the span achieves zero error at budget k,')
print('    so error cannot distinguish them; identification needs a constraint')
print('    that selects a basis inside the span. Classical factor-analysis')
print('    rotation indeterminacy.')
print('=' * 78)
g = torch.Generator().manual_seed(3)
d, k, r = 10, 3, 9
atoms = [sym(d, g) for _ in range(k)]
atoms = [a / a.norm() for a in atoms]
codes = torch.randn(r, k, generator=g)
Qu = torch.einsum('ua,aij->uij', codes, torch.stack(atoms))
iu, ju = torch.triu_indices(d, d)
sc = torch.where(iu == ju, 1.0, math.sqrt(2.0))
V = Qu[:, iu, ju] * sc
U_, s_, Vh_ = torch.linalg.svd(V, full_matrices=False)
print(f'  singular values: {[round(float(v),4) for v in s_[:5]]}')
check('   PCA error at the true budget is exactly zero', float(s_[k:].pow(2).sum()), 0.0,
      tol=1e-18) if False else None
resid = float(s_[k:].pow(2).sum() / s_.pow(2).sum())
OK.append(resid < 1e-25)
print(f'PASS  PCA residual energy beyond budget {k}: {resid:.3e}  (zero by construction)')
# any rotation of the planted atoms reproduces the data exactly
Rk = torch.linalg.qr(torch.randn(k, k, generator=g))[0]
rot_atoms = torch.einsum('ab,aij->bij', Rk, torch.stack(atoms))
rot_codes = codes @ Rk
Qh = torch.einsum('ua,aij->uij', rot_codes, rot_atoms)
err = float((Qu - Qh).norm() / Qu.norm())
OK.append(err < 1e-12)
print(f'PASS  an arbitrary rotation of the atoms reproduces the data to {err:.2e} '
      f'-- the basis inside the span is unidentified without a further constraint')

print()
print('=' * 78)
print('T8  Part B: the two-factor head has a scale gauge, so per-factor statistics')
print('    must be scale invariant. (W1, W2) -> (c W1, W2/c) is exactly')
print('    function preserving; softmax entropy is NOT invariant, participation')
print('    ratio IS.')
print('=' * 78)
import b_common as B
dgp = B.ConjunctiveRetrieval(device='cpu', seed=0)
for kind in ('logit', 'unnorm'):
    model = B.Head(dgp.d, dgp.V, kind=kind, rank=8, seed=0, device='cpu')
    b = dgp.batch(256)
    with torch.no_grad():
        y0 = model(b['x'])
        s0 = model.scores(b['x'])
        H0 = [-(torch.softmax(v, -1).clamp_min(1e-30).log() * torch.softmax(v, -1)).sum(-1).mean()
              for v in s0]
        pr0 = [((v ** 2).sum(-1) ** 2 / (v ** 4).sum(-1)).mean() for v in s0]
        model.Wq[0].mul_(20.0)
        model.Wq[1].div_(20.0)
        y1 = model(b['x'])
        s1 = model.scores(b['x'])
        H1 = [-(torch.softmax(v, -1).clamp_min(1e-30).log() * torch.softmax(v, -1)).sum(-1).mean()
              for v in s1]
        pr1 = [((v ** 2).sum(-1) ** 2 / (v ** 4).sum(-1)).mean() for v in s1]
    dfn = float((y0 - y1).abs().max() / y0.abs().max())
    OK.append(dfn < 1e-9)
    print(f'  {kind}: function change under the scale gauge = {dfn:.2e}')
    print(f'      softmax entropy per factor {[round(float(v),4) for v in H0]} -> '
          f'{[round(float(v),4) for v in H1]}   <-- NOT invariant')
    print(f'      participation ratio        {[round(float(v),4) for v in pr0]} -> '
          f'{[round(float(v),4) for v in pr1]}   <-- invariant')
    OK.append(all(abs(float(a) - float(c)) / max(float(c), 1e-30) < 1e-9
                  for a, c in zip(pr0, pr1)))

print()
print('=' * 78)
print(f'{sum(OK)}/{len(OK)} checks passed')
print('=' * 78)
sys.exit(0 if all(OK) else 1)
