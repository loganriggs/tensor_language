"""Fit-only sufficient-statistic solvers for compiler-v2 native programs."""

from __future__ import annotations

from typing import Any, Sequence

import torch


CAUSAL_FLOOR = 0.05
L1_RATIOS = (0.30, 0.10, 0.03, 0.01, 0.003, 0.001, 0.0)
POWER_ITERATIONS = 64
FISTA_ITERATIONS = 500


def causal_constant(target: torch.Tensor, adjoint: torch.Tensor) -> torch.Tensor:
    """Loss-optimal 64-vector constant under the registered Fisher-plus-floor metric."""

    target = target.double()
    adjoint = adjoint.double()
    if target.ndim != 2 or target.shape != adjoint.shape:
        raise ValueError("target and adjoint must be aligned matrices")
    n, coefficients = target.shape
    gradient_energy = adjoint.square().sum(dim=1).mean()
    if float(gradient_energy) <= 0.0:
        raise ValueError("causal adjoints have zero energy")
    metric = adjoint.T @ adjoint / (n * gradient_energy)
    metric += (CAUSAL_FLOOR / (coefficients * coefficients)) * torch.eye(
        coefficients, dtype=torch.float64, device=target.device
    )
    rhs = (adjoint.T @ (adjoint * target).sum(dim=1)) / (n * gradient_energy)
    rhs += CAUSAL_FLOOR * target.mean(dim=0) / (coefficients * coefficients)
    return torch.linalg.solve(metric, rhs)


def native_quadratic_statistics(
    phi: torch.Tensor,
    projected_decoder: torch.Tensor,
    target: torch.Tensor,
    *,
    adjoint: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return H,b,intercept for 0.5*a'H*a-b'a under a native gate vector.

    ``phi[:,i]`` is the i-th bilinear product and row ``Q[i]`` is its projected
    output direction.  Scaling H and b by the same positive constant does not
    change the unpenalized optimum; here the exact registered loss scaling is
    retained so lambda ratios are comparable within an objective.
    """

    phi = phi.double()
    q = projected_decoder.double()
    target = target.double()
    if phi.ndim != 2 or q.ndim != 2 or target.ndim != 2:
        raise ValueError("native statistics inputs must be matrices")
    if phi.shape[0] != target.shape[0] or q.shape != (phi.shape[1], target.shape[1]):
        raise ValueError("native statistics dimensions do not align")
    if not all(torch.isfinite(value).all() for value in (phi, q, target)):
        raise ValueError("native statistics inputs must be finite")
    n, coefficients = target.shape
    if adjoint is None:
        intercept = target.mean(dim=0)
        feature_offset = phi.mean(dim=0)
        design = phi - feature_offset
    else:
        adjoint = adjoint.double()
        if adjoint.shape != target.shape or not torch.isfinite(adjoint).all():
            raise ValueError("causal adjoints must align and be finite")
        intercept = causal_constant(target, adjoint)
        feature_offset = torch.zeros(
            phi.shape[1], dtype=torch.float64, device=phi.device
        )
        design = phi
    residual = target - intercept
    gram_phi = design.T @ design
    gram_q = q @ q.T
    cross = residual @ q.T

    if adjoint is None:
        hessian = (gram_phi * gram_q) / (n * coefficients)
        linear = (design * cross).sum(dim=0) / (n * coefficients)
    else:
        gradient_energy = adjoint.square().sum(dim=1).mean()
        q_dot_g = adjoint @ q.T
        directional_design = design * q_dot_g
        directional_target = (adjoint * residual).sum(dim=1)
        hessian = directional_design.T @ directional_design / (n * gradient_energy)
        linear = directional_design.T @ directional_target / (n * gradient_energy)
        hessian += (
            CAUSAL_FLOOR * (gram_phi * gram_q) / (n * coefficients * coefficients)
        )
        linear += (
            CAUSAL_FLOOR * (design * cross).sum(dim=0)
            / (n * coefficients * coefficients)
        )
    hessian = 0.5 * (hessian + hessian.T)
    return hessian, linear, intercept, feature_offset


def materialize_native_intercept(
    base_intercept: torch.Tensor,
    feature_offset: torch.Tensor,
    projected_decoder: torch.Tensor,
    support: torch.Tensor,
    amplitudes: torch.Tensor,
) -> torch.Tensor:
    """Convert an offset-feature solution into the serialized 64-vector bias."""

    support = torch.as_tensor(
        support, dtype=torch.long, device=feature_offset.device
    ).flatten()
    amplitudes = amplitudes.to(feature_offset.device).double().flatten()
    if support.numel() != amplitudes.numel():
        raise ValueError("support and amplitudes do not align")
    q = projected_decoder.to(feature_offset.device).double().index_select(0, support)
    offset = feature_offset.double().index_select(0, support) * amplitudes
    return base_intercept.double() - offset @ q


def largest_eigenvalue(
    hessian: torch.Tensor, *, iterations: int = POWER_ITERATIONS, seed: int = 271828
) -> float:
    hessian = hessian.double()
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Hessian must be square")
    vector = torch.randn(
        hessian.shape[0], generator=torch.Generator(device=hessian.device).manual_seed(seed),
        dtype=torch.float64, device=hessian.device,
    )
    vector /= vector.norm().clamp_min(1e-30)
    for _ in range(iterations):
        vector = hessian @ vector
        norm = vector.norm()
        if float(norm) <= 0.0:
            return 0.0
        vector /= norm
    return max(0.0, float(vector @ (hessian @ vector)))


def soft_threshold(value: torch.Tensor, threshold: float) -> torch.Tensor:
    return value.sign() * (value.abs() - threshold).clamp_min(0.0)


def fista_l1_path(
    hessian: torch.Tensor,
    linear: torch.Tensor,
    *,
    ratios: Sequence[float] = L1_RATIOS,
    iterations: int = FISTA_ITERATIONS,
) -> list[dict[str, Any]]:
    """Deterministic warm-start FISTA path for the convex native gate problem."""

    hessian = hessian.double()
    linear = linear.double()
    if hessian.ndim != 2 or hessian.shape != (linear.numel(), linear.numel()):
        raise ValueError("native gate Hessian/linear dimensions do not align")
    if not all(0.0 <= float(ratio) <= 1.0 for ratio in ratios):
        raise ValueError("L1 ratios must lie in [0,1]")
    lipschitz = largest_eigenvalue(hessian)
    if lipschitz <= 0.0:
        raise ValueError("native gate Hessian has zero curvature")
    lambda_max = float(linear.abs().max())
    state = torch.zeros_like(linear)
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        penalty = float(ratio) * lambda_max
        extrapolated = state.clone()
        momentum = 1.0
        for _ in range(iterations):
            prior = state
            gradient = hessian @ extrapolated - linear
            state = soft_threshold(extrapolated - gradient / lipschitz,
                                   penalty / lipschitz)
            next_momentum = 0.5 * (1.0 + (1.0 + 4.0 * momentum * momentum) ** 0.5)
            extrapolated = state + ((momentum - 1.0) / next_momentum) * (state - prior)
            momentum = next_momentum
        smooth = 0.5 * state @ (hessian @ state) - linear @ state
        rows.append({
            "lambda_ratio": float(ratio),
            "lambda": penalty,
            "gates": state.clone(),
            "nonzero": int((state != 0.0).sum()),
            "smooth_objective_without_constant": float(smooth),
            "penalized_objective_without_constant": float(
                smooth + penalty * state.abs().sum()
            ),
            "lipschitz": lipschitz,
            "iterations": int(iterations),
        })
    return rows


def refit_support(
    hessian: torch.Tensor,
    linear: torch.Tensor,
    support: torch.Tensor,
    *,
    ridge: float = 1e-8,
) -> torch.Tensor:
    support = torch.as_tensor(support, dtype=torch.long, device=hessian.device).flatten()
    if support.numel() == 0 or support.unique().numel() != support.numel():
        raise ValueError("native refit support must be nonempty and unique")
    sub_h = hessian.index_select(0, support).index_select(1, support).double()
    sub_b = linear.index_select(0, support).double()
    scale = max(1.0, float(torch.diag(sub_h).abs().max()))
    sub_h = sub_h + ridge * scale * torch.eye(
        support.numel(), dtype=torch.float64, device=sub_h.device
    )
    return torch.linalg.solve(sub_h, sub_b)


def select_refit_frontier(
    hessian: torch.Tensor,
    linear: torch.Tensor,
    path: Sequence[dict[str, Any]],
    k_grid: Sequence[int],
) -> dict[int, dict[str, Any]]:
    """For each K, choose the fit-only path support with best refit smooth loss."""

    output: dict[int, dict[str, Any]] = {}
    for k in k_grid:
        if not 1 <= int(k) <= linear.numel():
            raise ValueError(f"invalid native support size: {k}")
        candidates = []
        for row in path:
            gates = row["gates"]
            support = torch.topk(gates.abs(), int(k), sorted=True).indices
            amplitudes = refit_support(hessian, linear, support)
            smooth = 0.5 * amplitudes @ (
                hessian.index_select(0, support).index_select(1, support) @ amplitudes
            ) - linear.index_select(0, support) @ amplitudes
            candidates.append({
                "support": support.clone(),
                "amplitudes": amplitudes.clone(),
                "fit_smooth_objective_without_constant": float(smooth),
                "source_lambda_ratio": float(row["lambda_ratio"]),
            })
        output[int(k)] = min(
            candidates,
            key=lambda row: (row["fit_smooth_objective_without_constant"],
                             row["source_lambda_ratio"]),
        )
    return output
