"""§1.2 per-datapoint min-norm solutions and the negative result (Tier 0.2)."""

import torch


def M_x(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Min-norm rank-1 per-point solution (W x) xᵀ / ||x||²."""
    return torch.outer(W @ x, x) / (x @ x)


def diagonal_mass_fraction(W, x, E, S_idx):
    """Fraction of Frobenius mass of M_x on the |S| 'diagonal' terms u_j e_jᵀ."""
    M = M_x(W, x)
    diag = sum(torch.outer(W @ E[:, j], E[:, j]) * float((E[:, j] @ x) ** 2 / (x @ x))
               for j in S_idx)
    # project M onto the span of the |S| diagonal atoms (they are orthogonal in
    # the flattened inner product when features are orthonormal and W is generic)
    num = 0.0
    for j in S_idx:
        A = torch.outer(W @ E[:, j], E[:, j])
        num += (M * A).sum() ** 2 / (A * A).sum()
    return (num / (M * M).sum()).item()


def cos_to_mechanism(W, x, E, S_idx):
    """cos between flattened M_x and the true mechanism W P_S."""
    M = M_x(W, x)
    P = sum(torch.outer(E[:, j], E[:, j]) for j in S_idx)
    T = W @ P
    return ((M * T).sum() / (M.norm() * T.norm())).item()
