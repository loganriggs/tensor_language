"""CPU linear algebra for a typed, finite-horizon observable closure test.

This is deliberately not a model collector.  A future source-closed experiment must
provide frozen observable rows at consecutive early-layer interfaces.  The functions
here answer two mathematical questions only:

1. What is the optimal reduced-rank linear transition between two observable spaces,
   measured in a registered positive-semidefinite downstream metric?
2. Does composing two fitted transitions incur only the residual predicted by the
   exact two-step error identity?

The second question distinguishes a small *law* from two unrelated local fits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReducedRankTransition:
    coefficient: torch.Tensor
    requested_rank: int
    fitted_rank: int
    source_numerical_rank: int
    metric_support_rank: int
    singular_values: torch.Tensor
    weighted_squared_error: float
    weighted_target_energy: float
    normalized_closure_defect: float


@dataclass(frozen=True)
class TwoStepClosureReport:
    first_local_norm_in_final_metric: float
    second_local_norm: float
    composed_norm: float
    triangle_upper_bound: float
    identity_maximum_absolute_error: float
    direct_norm: float | None
    composition_to_direct_ratio: float | None


def _matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.ndim != 2
        or value.numel() == 0
        or not value.is_floating_point()
        or value.device.type != "cpu"
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} must be a finite, nonempty, gradient-free CPU matrix")
    return value.detach().clone().double()


def augment_constant(features: torch.Tensor) -> torch.Tensor:
    """Prepend a constant observable, making affine transitions explicit."""
    checked = _matrix("features", features)
    return torch.cat((torch.ones((checked.shape[0], 1), dtype=torch.float64), checked), dim=1)


def _metric_factor(
    width: int, metric: torch.Tensor | None, *, support_rtol: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if support_rtol <= 0:
        raise ValueError("support_rtol must be positive")
    if metric is None:
        identity = torch.eye(width, dtype=torch.float64)
        return identity, identity, width
    checked = _matrix("target metric", metric)
    if checked.shape != (width, width):
        raise ValueError("target metric has the wrong shape")
    if not torch.allclose(checked, checked.T, atol=1e-11, rtol=1e-11):
        raise ValueError("target metric must be symmetric")
    eigenvalues, eigenvectors = torch.linalg.eigh(checked)
    scale = max(float(eigenvalues.abs().max()), 1.0)
    if float(eigenvalues.min()) < -support_rtol * scale:
        raise ValueError("target metric must be positive semidefinite")
    keep = eigenvalues > support_rtol * scale
    support = int(keep.sum())
    if support == 0:
        raise ValueError("target metric must have positive support")
    vectors = eigenvectors[:, keep]
    roots = eigenvalues[keep].clamp_min(0).sqrt()
    factor = vectors * roots
    inverse = (vectors / roots).T
    return factor, inverse, support


def weighted_frobenius_norm(rows: torch.Tensor, metric: torch.Tensor | None = None) -> float:
    """Return sqrt(sum_i rows[i] @ metric @ rows[i].T)."""
    checked = _matrix("rows", rows)
    factor, _, _ = _metric_factor(checked.shape[1], metric, support_rtol=1e-12)
    return float(torch.linalg.vector_norm(checked @ factor))


def fit_reduced_rank_transition(
    source: torch.Tensor,
    target: torch.Tensor,
    rank: int,
    *,
    target_metric: torch.Tensor | None = None,
    support_rtol: float = 1e-12,
) -> ReducedRankTransition:
    """Fit the metric-weighted reduced-rank regression optimum.

    For ``Z = target @ metric**(1/2)``, first project Z onto the column
    space of ``source`` and then take its rank-r truncated SVD.  This is the exact
    reduced-rank least-squares solution on the metric support.
    """
    x = _matrix("source", source)
    y = _matrix("target", target)
    if x.shape[0] != y.shape[0]:
        raise ValueError("source and target must have the same number of rows")
    if type(rank) is not int or rank <= 0:
        raise ValueError("rank must be a positive integer")
    factor, inverse_factor, metric_support = _metric_factor(
        y.shape[1], target_metric, support_rtol=support_rtol,
    )
    u_x, singular_x, vh_x = torch.linalg.svd(x, full_matrices=False)
    source_scale = max(float(singular_x[0]), 1.0)
    keep_x = singular_x > support_rtol * source_scale
    source_rank = int(keep_x.sum())
    maximum_rank = min(source_rank, metric_support)
    if rank > maximum_rank:
        raise ValueError("rank exceeds the estimable source/metric support")

    # P_X Z, formed without constructing the n by n projector.
    z = y @ factor
    u_support = u_x[:, keep_x]
    projected = u_support @ (u_support.T @ z)
    u_fit, singular_fit, vh_fit = torch.linalg.svd(projected, full_matrices=False)
    projected_rank_r = (u_fit[:, :rank] * singular_fit[:rank]) @ vh_fit[:rank]

    # pinv(X) @ projected_rank_r, with the same registered numerical support.
    source_pinv = (vh_x[keep_x].T / singular_x[keep_x]) @ u_support.T
    transformed_coefficient = source_pinv @ projected_rank_r
    coefficient = transformed_coefficient @ inverse_factor
    prediction = x @ coefficient
    residual = (y - prediction) @ factor
    error = float(residual.square().sum())
    energy = float(z.square().sum())
    defect = error / energy if energy > 0 else (0.0 if error == 0 else float("inf"))
    fitted_rank = int((singular_fit > support_rtol * max(float(singular_fit[0]), 1.0)).sum())
    return ReducedRankTransition(
        coefficient=coefficient,
        requested_rank=rank,
        fitted_rank=min(fitted_rank, rank),
        source_numerical_rank=source_rank,
        metric_support_rank=metric_support,
        singular_values=singular_fit.detach().clone(),
        weighted_squared_error=error,
        weighted_target_energy=energy,
        normalized_closure_defect=defect,
    )


def two_step_closure_report(
    source: torch.Tensor,
    middle: torch.Tensor,
    target: torch.Tensor,
    first_coefficient: torch.Tensor,
    second_coefficient: torch.Tensor,
    *,
    final_metric: torch.Tensor | None = None,
    direct_coefficient: torch.Tensor | None = None,
) -> TwoStepClosureReport:
    """Audit composition using an exact residual identity and triangle certificate.

    With ``E01 = middle - source @ B01`` and
    ``E12 = target - middle @ B12``,

        target - source @ B01 @ B12 = E12 + E01 @ B12.

    Thus the composed defect is bounded without assuming independent errors.
    """
    x, y, z = (_matrix("source", source), _matrix("middle", middle), _matrix("target", target))
    b01 = _matrix("first coefficient", first_coefficient)
    b12 = _matrix("second coefficient", second_coefficient)
    if x.shape[0] != y.shape[0] or y.shape[0] != z.shape[0]:
        raise ValueError("all observable matrices must share rows")
    if b01.shape != (x.shape[1], y.shape[1]) or b12.shape != (y.shape[1], z.shape[1]):
        raise ValueError("transition coefficient shapes do not match the typed interfaces")
    first_error = y - x @ b01
    second_error = z - y @ b12
    propagated_first = first_error @ b12
    composed_error = z - x @ b01 @ b12
    identity_error = composed_error - (second_error + propagated_first)
    first_norm = weighted_frobenius_norm(propagated_first, final_metric)
    second_norm = weighted_frobenius_norm(second_error, final_metric)
    composed_norm = weighted_frobenius_norm(composed_error, final_metric)
    direct_norm: float | None = None
    ratio: float | None = None
    if direct_coefficient is not None:
        direct = _matrix("direct coefficient", direct_coefficient)
        if direct.shape != (x.shape[1], z.shape[1]):
            raise ValueError("direct coefficient shape does not match source and target")
        direct_norm = weighted_frobenius_norm(z - x @ direct, final_metric)
        ratio = composed_norm / direct_norm if direct_norm > 0 else (
            1.0 if composed_norm == 0 else float("inf")
        )
    return TwoStepClosureReport(
        first_local_norm_in_final_metric=first_norm,
        second_local_norm=second_norm,
        composed_norm=composed_norm,
        triangle_upper_bound=first_norm + second_norm,
        identity_maximum_absolute_error=float(identity_error.abs().max()),
        direct_norm=direct_norm,
        composition_to_direct_ratio=ratio,
    )
