"""§1.1 of mechanism_decomposition_spec.md: closed-form low-rank data-conditioned
solutions.

Problem 1 (functional core):  min_{rank r} ||M X − W X||_F²  →  M* = [W X† X]_r X†
Problem 2 (edit):             min_{rank r} ||M X − Y||_F²     →  M* = [Y X† X]_r X†

with X† = Xᵀ(XXᵀ)^{-1} (requires rank(X) = d_in, assumption A1). Optimal value
Σ_{j>r} σ_j²(Y X† X) + ||Y (I − X† X)||_F². Everything torch, CPU/GPU agnostic.
"""

import torch


def pinv_rows(X: torch.Tensor, ridge: float = 0.0) -> torch.Tensor:
    """X† = Xᵀ (X Xᵀ + ridge·I)^{-1} for X (d_in, N) with full row rank."""
    d = X.shape[0]
    G = X @ X.T
    if ridge:
        G = G + ridge * torch.eye(d, dtype=X.dtype, device=X.device)
    return X.T @ torch.linalg.inv(G)


def svd_truncate(M: torch.Tensor, r: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]


def rank_r_solution(Y: torch.Tensor, X: torch.Tensor, r: int,
                    ridge: float = 0.0) -> torch.Tensor:
    """Global minimizer M* = [Y X† X]_r X† of ||M X − Y||_F² over rank-≤r M.
    Y: (d_out, N) targets (Problem 1: Y = W X). X: (d_in, N)."""
    Xp = pinv_rows(X, ridge)                    # (N, d_in)
    # X†X is (N,N); avoid it via Y X† X = (Y X†) X with B = Y X† (d_out, d_in)
    B = Y @ Xp
    core = B @ X                                 # = Y X† X  (d_out, N)
    return svd_truncate(core, r) @ Xp            # (d_out, d_in)


def optimal_value(Y: torch.Tensor, X: torch.Tensor, r: int,
                  ridge: float = 0.0) -> torch.Tensor:
    """Σ_{j>r} σ_j²(Y X† X) + ||Y (I − X† X)||_F², computed without (N,N) matrices."""
    Xp = pinv_rows(X, ridge)
    B = Y @ Xp
    core = B @ X                                 # Y X† X
    S = torch.linalg.svdvals(core)
    tail = (S[r:] ** 2).sum()
    resid = Y - core                             # Y (I − X†X)
    return tail + (resid ** 2).sum()


def achieved_loss(M: torch.Tensor, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return ((M @ X - Y) ** 2).sum()
