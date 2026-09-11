"""Exact block updates for F(x)=sum_j (a_j.x) U_j V_j.T x.

Output weights passed here already include the full-U isometric metric.
Conditional optima do not guarantee joint/global recovery.
"""
import torch
from shared_input_factor_v1 import native_partner


def partner_update(a, left, right, writers, rank):
    """Reuse the known transformed-SVD optimum at a fixed unit reader."""
    m = native_partner(a, left, right, writers)
    transformed = m + (2**.5 - 1) * (m @ a)[:, None] * a[None, :]
    u, s, vh = torch.linalg.svd(transformed, full_matrices=False)
    u = u[:, :rank] * s[:rank]
    v = vh[:rank].T
    v = v + (2**-.5 - 1) * a[:, None] * (a @ v)[None, :]
    return u, v


def reader_force(left, right, writers, u, v):
    """s_i=sum_ok E_oik M_ok, E in signed product form, M=U V.T."""
    z = writers.T @ u
    return .5 * (left.T @ (z * (right @ v)).sum(1)
                 + right.T @ (z * (left @ v)).sum(1))


def sphere_psd_lowrank(u, v, rhs, rtol=1e-12):
    """Minimize .5 a.T M.T M a - rhs.T a for ||a||=1, rank(M)<d.

    Q+lambda I is PSD and (Q+lambda I)a=rhs. These conditions give
    the global conditional certificate. Tiny numerical eigenvalues may be
    discarded; diagnostics replay KKT using the original factors.
    """
    d, rank = v.shape
    if not 0 < rank < d:
        raise ValueError('Requires 0 < partner factor width < input dimension')
    basis, triangular = torch.linalg.qr(v, mode='reduced')
    core = triangular @ (u.T @ u) @ triangular.T
    values, vectors = torch.linalg.eigh((core + core.T) / 2)
    threshold = rtol * max(float(values[-1]), torch.finfo(values.dtype).tiny)
    keep = values > threshold
    values, basis = values[keep], basis @ vectors[:, keep]
    projection = basis.T @ rhs
    perpendicular = rhs - basis @ projection
    force_norm = rhs.norm()
    inverse = projection / values
    base = basis @ inverse
    hard = (float(perpendicular.norm()) <= rtol * float(force_norm)
            and float(base.norm()) <= 1)
    if hard:
        index = int(basis.square().sum(1).argmin())
        null = torch.zeros_like(rhs)
        null[index] = 1
        null -= basis @ basis[index]
        null /= null.norm()
        a = base + (1 - base.square().sum()).clamp_min(0).sqrt() * null
        lagrange = 0.
    else:
        low, high = 0., float(force_norm)
        for _ in range(100):
            middle = (low + high) / 2
            norm2 = (projection / (values + middle)).square().sum()
            norm2 += perpendicular.square().sum() / middle**2
            if float(norm2) > 1:
                low = middle
            else:
                high = middle
        lagrange = high
        a = basis @ (projection / (values + lagrange)) + perpendicular / lagrange
    a /= a.norm()
    qa = v @ ((u.T @ u) @ (v.T @ a))
    qscale = torch.linalg.norm(core)
    residual = (qa + lagrange * a - rhs).norm() / (qscale + force_norm).clamp_min(1e-300)
    return a, dict(lagrange=lagrange, hard_case=hard, retained_rank=len(values),
                  kkt_relative_residual=float(residual), unit_error=float(abs(a.norm()-1)))


def reader_update(left, right, writers, u, v):
    return sphere_psd_lowrank(u, v, 2 * reader_force(left, right, writers, u, v))


def group_cp(a, u, v):
    return a.expand(v.shape[1], -1), v.T, u


def residual_cp(target, groups, excluded):
    parts = [target] + [(a.expand(v.shape[1], -1), v.T, -u)
                        for j, (a, u, v) in enumerate(groups) if j != excluded]
    return tuple(torch.cat([part[k] for part in parts], dim=1 if k == 2 else 0)
                 for k in range(3))


def dense_cp(left, right, writers):
    return torch.einsum('ok,ki,kj->oij', writers, left, right).add(
        torch.einsum('ok,kj,ki->oij', writers, left, right)).mul(.5)
