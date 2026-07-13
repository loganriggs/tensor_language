"""M-step: EXACT per-atom updates (spec §3 preference (a)).

Per atom j, holding others fixed, the subproblem is
    min_d Σ_i || c_ij (dᵀ x_i) W d − r_i ||²,
nonlinear because d enters twice. Solved by a short fixed-point alternation:
with β_i = c_ij (dᵀ x_i) frozen, the optimum satisfies W d = (Σ β_i r_i)/(Σ β_i²)
(least squares through W), then β is refreshed from the new d. 2-3 iterations
suffice. The failed alternative (Adam on D with norm-detach) walks away from the
optimum along noise-floor gradients — kept out deliberately; see results_mechdecomp.md.
"""

import torch

from .objective import predict


def rowspace_basis(W, tol=1e-4):
    """Orthonormal basis of W's row space (input directions W actually reads)."""
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = int((S > tol * S[0]).sum())
    return Vh[:r].T                                   # (d_in, r)


def update_dictionary(W, D, C, X, inner: int = 3, active_only: bool = True,
                      rowspace: torch.Tensor | None = None, Wpinv: torch.Tensor | None = None) -> torch.Tensor:
    D = D.clone()
    if Wpinv is None:
        Wpinv = torch.linalg.pinv(W)              # rank-aware, stable any-rank; exact full-rank
    if rowspace is not None:
        P = rowspace @ rowspace.T                     # projector onto row(W)
    WD = W @ D
    A = D.T @ X
    Yhat = WD @ (C * A)
    Y = W @ X
    for j in range(D.shape[1]):
        cj = C[j]
        if active_only and (cj.abs() > 1e-10).sum() == 0:
            continue
        contrib_j = torch.outer(WD[:, j], cj * A[j])
        R = Y - (Yhat - contrib_j)                     # residual this atom must fit
        d = D[:, j].clone()
        for _ in range(inner):
            beta = cj * (d @ X)                        # (N,)
            denom = (beta ** 2).sum().clamp_min(1e-12)
            rbar = (R * beta[None, :]).sum(1) / denom  # target for W d
            d = Wpinv @ rbar
            n = d.norm()
            if n < 1e-9:
                break
            d = d / n
        if rowspace is not None:
            d = P @ d
            n = d.norm()
            if n > 1e-9:
                d = d / n
        D[:, j] = d
        WD[:, j] = W @ d
        A[j] = d @ X
        Yhat = Y - R + torch.outer(WD[:, j], cj * A[j])
    return D


def resample_dead(W, D, C, X, thresh: float = 1e-8) -> tuple[torch.Tensor, int]:
    """Reinit atoms with zero support to the input directions of the worst-
    reconstructed datapoints."""
    support = (C.abs() > thresh).sum(1)
    dead = torch.where(support == 0)[0]
    if len(dead) == 0:
        return D, 0
    resid = ((predict(W, D, C, X) - W @ X) ** 2).sum(0)
    worst = torch.argsort(-resid)[:len(dead)]
    D = D.clone()
    D[:, dead] = X[:, worst] / X[:, worst].norm(dim=0, keepdim=True)
    return D, len(dead)
