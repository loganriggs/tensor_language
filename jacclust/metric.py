"""Exact Jacobian kernel for a bilinear layer  y = D (Lx ⊙ Rx).

J(x) = D [diag(Lx) R + diag(Rx) L]  is LINEAR in x, so the Frobenius inner product
(equivalently the trace form: <A,B>_F = tr(A^T B)) collapses to a quadratic form:

    <J(x), J(x')>_F  =  x^T G x'

with G computable from the weights alone. Each term via
    tr(P^T diag(a) M diag(a') P') = a^T (M ⊙ P P'^T) a',     M = D^T D

    G = L^T (M ⊙ R R^T) L
      + L^T (M ⊙ R L^T) R
      + R^T (M ⊙ L R^T) L
      + R^T (M ⊙ L L^T) R

So Jacobian cosine similarity == cosine similarity of z = G^{1/2} x, and no Jacobian is
ever materialized. Verified to float64 precision by `tests/test_kernel.py` (P1).
"""

import torch


def jacobian(D, L, R, x):
    """Closed-form J(x) = D[diag(Lx) R + diag(Rx) L]. No autodiff."""
    a, b = L @ x, R @ x
    return D @ (a[:, None] * R + b[:, None] * L)


def gram(D, L, R):
    """The d x d PSD matrix G with <J(x), J(x')>_F = x^T G x'."""
    M = D.T @ D
    return (L.T @ (M * (R @ R.T)) @ L
            + L.T @ (M * (R @ L.T)) @ R
            + R.T @ (M * (L @ R.T)) @ L
            + R.T @ (M * (L @ L.T)) @ R)


def embed(G, X, r=None):
    """z_i = Λ^{1/2} Vᵀ x_i — same inner products as G^{1/2} x. Optionally keep top-r eigendirections."""
    w, V = torch.linalg.eigh(G)
    w = w.clamp_min(0.0)
    idx = torch.argsort(-w)
    w, V = w[idx], V[:, idx]
    if r is not None:
        w, V = w[:r], V[:, :r]
    return (X @ V) * w.sqrt()[None, :]


def frobenius_cosine(A, B):
    """cos_F(A,B) = tr(A^T B) / (||A||_F ||B||_F) — identical to cosine on flattened matrices."""
    num = (A * B).sum()
    return num / (A.norm() * B.norm()).clamp_min(1e-30)
