"""Snapshot balancing for a reachable-and-observable causal port.

Reachable snapshots are physical residual-write/edit vectors. Observable snapshots
are downstream output gradients or fixed linear response tests. Their normalized
cross matrix is invariant under an invertible change of residual coordinates. Its
singular values therefore measure directions that are both reachable and observable,
unlike PCA of writes or gradients alone.

This is a linearized finite-port construction. It is not a nonlinear whole-transformer
certificate; finite secants and held-out compositions must validate any retained port.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class BalancedPort:
    rank: int
    state_width: int
    hankel_singular_values: torch.Tensor
    primal_basis: torch.Tensor
    dual_basis: torch.Tensor
    projection: torch.Tensor
    response_tail_squared_frobenius: float
    biorthogonality_max_abs_error: float


def _validate(reachable: torch.Tensor, observable: torch.Tensor, rank: int) -> None:
    if any(
        not torch.is_tensor(value) or value.device.type != "cpu" or value.ndim != 2
        or value.dtype != torch.float64 or not bool(torch.isfinite(value).all())
        for value in (reachable, observable)
    ):
        raise ValueError("snapshots must be finite CPU float64 matrices")
    if reachable.shape[1] != observable.shape[1] or min(reachable.shape[0], observable.shape[0]) < 1:
        raise ValueError("reachable and observable snapshots have incompatible state widths")
    if type(rank) is not int or not 1 <= rank <= min(reachable.shape[0], observable.shape[0], reachable.shape[1]):
        raise ValueError("balanced-port rank is invalid")


def fit_balanced_port(
    reachable: torch.Tensor,
    observable: torch.Tensor,
    rank: int,
    *,
    relative_support_tolerance: float = 1e-12,
) -> BalancedPort:
    """Fit primal/dual snapshot modes from the observable-reachable cross map.

    If row snapshots encode state columns x and an invertible coordinate change is
    x'=T x, use ``reachable'=reachable @ T.T`` and
    ``observable'=observable @ inv(T)``. The cross map, singular values, and physical
    input-output prediction are invariant; primal and dual modes transform covariantly.
    """
    _validate(reachable, observable, rank)
    if not isinstance(relative_support_tolerance, float) or not (
        0 < relative_support_tolerance < 1
    ):
        raise ValueError("support tolerance must lie strictly between zero and one")
    reach_factor = reachable.T / math.sqrt(reachable.shape[0])
    observe_factor = observable / math.sqrt(observable.shape[0])
    cross = observe_factor @ reach_factor
    u, singular, vh = torch.linalg.svd(cross, full_matrices=False)
    if singular.numel() < rank or float(singular[rank - 1]) <= (
        relative_support_tolerance * max(float(singular[0]), 1.0)
    ):
        raise RuntimeError("requested balanced rank exceeds numerical response support")
    inverse_root = singular[:rank].rsqrt()
    primal = reach_factor @ (vh[:rank].T * inverse_root)
    dual = observe_factor.T @ (u[:, :rank] * inverse_root)
    biorthogonality = dual.T @ primal
    error = float((biorthogonality - torch.eye(rank, dtype=torch.float64)).abs().max())
    if error > 1e-9:
        raise RuntimeError("balanced primal/dual bases failed biorthogonality")
    projection = primal @ dual.T
    tail = float(singular[rank:].square().sum())
    return BalancedPort(
        rank=rank,
        state_width=reachable.shape[1],
        hankel_singular_values=singular,
        primal_basis=primal,
        dual_basis=dual,
        projection=projection,
        response_tail_squared_frobenius=tail,
        biorthogonality_max_abs_error=error,
    )


def projected_response(
    port: BalancedPort, reachable: torch.Tensor, observable: torch.Tensor,
) -> torch.Tensor:
    _validate(reachable, observable, port.rank)
    if reachable.shape[1] != port.state_width:
        raise ValueError("response state width differs from fitted port")
    reach_factor = reachable.T / math.sqrt(reachable.shape[0])
    observe_factor = observable / math.sqrt(observable.shape[0])
    return observe_factor @ port.projection @ reach_factor

