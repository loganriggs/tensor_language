"""E-step: batched nonnegative lasso with the Hadamard-Gram factorization (§1.5).

Per point: design columns a_ij (W d_j) ⇒ Gram G_i = (a_i a_iᵀ) ⊙ G_W with
G_W = (WD)ᵀ(WD) precomputed once. Linear term b_i = a_i ⊙ ((WD)ᵀ W x_i).
Cyclic coordinate descent, vectorized over the batch dimension.
"""

import torch


@torch.no_grad()
def solve_codes(W, D, X, lam: float, n_iter: int = 60, nonneg: bool = True,
                C0: torch.Tensor | None = None, gain_weighted: bool = False,
                ridge: float = 0.0, gate_weighted_l1: bool = False) -> torch.Tensor:
    """gate_weighted_l1: penalize the EFFECTIVE coefficient c_ij·|d_jᵀx_i| instead of c_ij.
    The mechanism's contribution is c·(dᵀx)·(W d), so a plain L1 on c is gate-warped: atoms
    with small |dᵀx| pay extra for the same output contribution — a real bias against
    feature directions with modest projections (Logan, 2026-07-09)."""
    WD = W @ D                                  # (d_out, m)
    G_W = WD.T @ WD                             # (m, m)
    A = D.T @ X                                 # (m, N)
    B = A * (WD.T @ (W @ X))                    # (m, N): b_i = a_i ⊙ (WDᵀ W x_i)
    m, N = A.shape
    lam_j = lam * WD.norm(dim=0) if gain_weighted else torch.full((m,), lam, dtype=A.dtype, device=A.device)
    # per-(atom,point) penalty: plain L1 -> lam_j; effective-coeff L1 -> lam_j * |a_ij|
    LAM = lam_j[:, None] * A.abs() if gate_weighted_l1 else lam_j[:, None].expand_as(A)
    C = torch.zeros_like(A) if C0 is None else C0.clone()
    diag = (A ** 2) * torch.diag(G_W)[:, None]  # G_i[jj] = a_ij² G_W[jj]  (m, N)
    if ridge:
        diag = diag + ridge                     # elastic-net: tames WD-collinearity blowup
    diag = diag.clamp_min(1e-12)
    for _ in range(n_iter):
        for j in range(m):
            # residual correlation for coord j across the batch:
            # (G_i c_i)_j = a_ij * Σ_k G_W[jk] a_ik c_ik
            s = A[j] * (G_W[j] @ (A * C)) - diag[j] * C[j]
            cj = (B[j] - s - LAM[j]) / diag[j]
            if nonneg:
                cj = cj.clamp_min(0.0)
            else:
                neg = (B[j] - s + LAM[j]) / diag[j]
                cj = torch.where(cj > 0, cj, torch.where(neg < 0, neg, torch.zeros_like(cj)))
            C[j] = cj
    return C
