"""Exact degree-zero/two Hermite coefficients of a native homogeneous quartic.

Only the output-by-input-by-input quadratic projection is materialized. The
native quartic tensor is never expanded. Intended for fixed teacher weights.
"""
import torch


def project(teacher):
    C, L2, R2, D1, L, R = teacher
    A = L2 @ D1
    B = R2 @ D1
    tau = (L * R).sum(1)
    ta, tb = A @ tau, B @ tau
    LL, RR, RL = L @ L.T, R @ R.T, R @ L.T
    quadratic_gram = .5 * (LL * RR + RL * RL.T)
    mean = C @ (ta * tb + 2 * ((A @ quadratic_gram) * B).sum(1))
    weights = (C * ta) @ B + (C * tb) @ A
    projections = []
    for c, weight in zip(C, weights):
        base = L.T @ (weight[:, None] * R)
        base = .5 * (base + base.T)
        K = A.T @ (c[:, None] * B)
        K = .5 * (K + K.T)
        cross = L.T @ (K * RL) @ R
        Q = base + cross + cross.T + L.T @ (K * RR) @ L + R.T @ (K * LL) @ R
        projections.append(.5 * (Q + Q.T))
    return mean, torch.stack(projections)


def cp_mean(factors):
    a,b,c,d=factors
    dot=lambda u,v:(u*v).sum(1)
    return dot(a,b)*dot(c,d)+dot(a,c)*dot(b,d)+dot(a,d)*dot(b,c)


def cp_cross(teacher, mean, quadratic, factors):
    """E[F(x) phi(x)] exactly for x~N(0,I), phi quartic CP atoms."""
    from quartic_cp import directional
    import itertools
    cross = 24 * directional(*teacher, factors).T
    for i,j in itertools.combinations(range(4),2):
        k,l=[s for s in range(4) if s not in (i,j)]
        dot=(factors[i]*factors[j]).sum(1)
        cross=cross+2*dot*torch.einsum('ki,vij,kj->vk',factors[k],quadratic,factors[l])
    return cross+mean[:,None]*cp_mean(factors)[None,:]
