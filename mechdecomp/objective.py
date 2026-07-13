"""Masked-projector objective, contraction-ordered (spec §1.3, §1.5).

Ŷ = Σ_j c_ij · a_ij · (W d_j),  a_ij = d_jᵀ x_i.  Never materialize per-point
d_out × d_in matrices: A = Dᵀ X (m, N);  Ŷ = (W D) @ (C ⊙ A) with C (m, N).
"""

import torch


def predict(W: torch.Tensor, D: torch.Tensor, C: torch.Tensor,
            X: torch.Tensor) -> torch.Tensor:
    """Ŷ (d_out, N). D: (d_in, m) unit columns. C: (m, N) codes."""
    A = D.T @ X                       # (m, N)
    return (W @ D) @ (C * A)          # (d_out, N)


def loss(W, D, C, X, lam: float = 0.0) -> torch.Tensor:
    r = predict(W, D, C, X) - W @ X
    l = (r ** 2).sum()
    if lam:
        l = l + lam * C.abs().sum()
    return l


def r2(W, D, C, X) -> float:
    Y = W @ X
    resid = ((predict(W, D, C, X) - Y) ** 2).sum()
    return float(1 - resid / (Y ** 2).sum())
