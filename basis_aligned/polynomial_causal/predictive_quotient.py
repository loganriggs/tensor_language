"""Gauge-invariant finite-horizon predictive quotient for a learned code.

This is a pure CPU mathematics module.  It loads no model or corpus and grants no
experimental authority.  Given a natural code covariance C and a downstream local
response metric O, it solves the one-interface balanced reduction problem defined in
MATHEMATICAL_REVIEW_2026-08-28_0630.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def _symmetric_psd(name: str, value: torch.Tensor, *, rtol: float) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 2 or value.shape[0] != value.shape[1] \
            or value.shape[0] == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be one finite square matrix")
    matrix = value.detach().cpu().double().contiguous().clone()
    scale = max(1.0, float(torch.linalg.matrix_norm(matrix, ord=2)))
    symmetry_error = float(torch.max(torch.abs(matrix - matrix.T)))
    if symmetry_error > rtol * scale:
        raise ValueError(f"{name} is not symmetric within tolerance")
    matrix = ((matrix + matrix.T) / 2).contiguous()
    eigenvalues = torch.linalg.eigvalsh(matrix)
    if float(eigenvalues[0]) < -rtol * scale:
        raise ValueError(f"{name} is not positive semidefinite within tolerance")
    return matrix


def covariance_from_codes(codes: torch.Tensor) -> torch.Tensor:
    """Return the unbiased covariance of a complete [sample, code] trajectory."""

    if not torch.is_tensor(codes) or codes.ndim != 2 or codes.shape[0] < 2 or (
        codes.shape[1] == 0
    ) or not bool(torch.isfinite(codes).all()):
        raise ValueError("codes must be finite [sample>=2, code_dim]")
    values = codes.detach().cpu().double()
    centered = values - values.mean(dim=0)
    return (centered.T @ centered / (len(centered) - 1)).contiguous()


def observability_from_vjp_sketches(gradients: torch.Tensor) -> torch.Tensor:
    """Estimate E[J^T F J] from Fisher-whitened output VJP sketches.

    ``gradients[c, p]`` is J_c^T r_cp, where the registered output probe has
    conditional covariance E[r r^T | c] = F_c.  Averaging outer products is then an
    unbiased estimator of the local downstream Fisher/response Gramian.
    """

    if not torch.is_tensor(gradients) or gradients.ndim != 3 or min(
        gradients.shape
    ) <= 0 or not bool(torch.isfinite(gradients).all()):
        raise ValueError("VJP sketches must be finite [context, probe, code_dim]")
    values = gradients.detach().cpu().double().reshape(-1, gradients.shape[-1])
    return (values.T @ values / len(values)).contiguous()


@dataclass(frozen=True)
class PredictiveQuotient:
    """Balanced coordinates and exact quadratic tail certificate for one interface."""

    covariance: torch.Tensor
    observability: torch.Tensor
    covariance_sqrt: torch.Tensor
    covariance_inverse_sqrt: torch.Tensor
    balanced_operator: torch.Tensor
    eigenvalues: torch.Tensor
    hankel_singular_values: torch.Tensor
    whitened_directions: torch.Tensor
    natural_directions: torch.Tensor
    support_rank: int
    psd_rtol: float
    support_rtol: float

    @property
    def code_dim(self) -> int:
        return int(self.covariance.shape[0])

    @property
    def total_quadratic_response(self) -> float:
        return float(self.eigenvalues.sum())

    def _rank(self, rank: int) -> int:
        if type(rank) is not int or not 0 <= rank <= self.support_rank:
            raise ValueError("rank must lie in the covariance support")
        return rank

    def projector(self, rank: int) -> torch.Tensor:
        """Optimal rank-d code reconstruction map in natural coordinates."""

        rank = self._rank(rank)
        directions = self.whitened_directions[:, :rank]
        return (
            self.covariance_sqrt @ directions @ directions.T
            @ self.covariance_inverse_sqrt
        ).contiguous()

    def discarded_quadratic_response(self, rank: int) -> float:
        """Minimum E[(z-z_hat)^T O (z-z_hat)] at the requested rank."""

        rank = self._rank(rank)
        return float(self.eigenvalues[rank:self.support_rank].sum())

    def retained_response_fraction(self, rank: int) -> float:
        rank = self._rank(rank)
        total = self.total_quadratic_response
        if total <= 0:
            return 1.0
        return 1.0 - self.discarded_quadratic_response(rank) / total

    def rank_for_fraction(self, fraction: float) -> int:
        if not isinstance(fraction, (int, float)) or not math.isfinite(float(fraction)) \
                or not 0 <= float(fraction) <= 1:
            raise ValueError("retained fraction must lie in [0,1]")
        return next(
            rank for rank in range(self.support_rank + 1)
            if self.retained_response_fraction(rank) >= float(fraction)
        )

    def quadratic_response(self, delta: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(delta) or delta.shape[-1] != self.code_dim or not bool(
            torch.isfinite(delta).all()
        ):
            raise ValueError("delta must be finite and end in code_dim")
        values = delta.to(dtype=torch.float64, device="cpu")
        return torch.einsum("...i,ij,...j->...", values, self.observability, values)


def solve_predictive_quotient(
    covariance: torch.Tensor,
    observability: torch.Tensor,
    *,
    psd_rtol: float = 1e-10,
    support_rtol: float = 1e-12,
) -> PredictiveQuotient:
    """Solve the covariance/observability balanced interface reduction.

    If z has covariance C and the local downstream distortion is
    (z-z_hat)^T O (z-z_hat), the optimal rank-d linear reconstruction has discarded
    distortion sum_{i>d} lambda_i, where lambda_i are the descending eigenvalues of
    C^{1/2} O C^{1/2}.  This is an Eckart--Young/Ky Fan statement after whitening.
    """

    if not isinstance(psd_rtol, (int, float)) or not isinstance(
        support_rtol, (int, float)
    ) or not 0 < float(psd_rtol) < 1 or not 0 < float(support_rtol) < 1:
        raise ValueError("solver tolerances must lie strictly in (0,1)")
    covariance = _symmetric_psd("covariance", covariance, rtol=float(psd_rtol))
    observability = _symmetric_psd(
        "observability", observability, rtol=float(psd_rtol),
    )
    if covariance.shape != observability.shape:
        raise ValueError("covariance and observability dimensions differ")
    c_values, c_vectors = torch.linalg.eigh(covariance)
    c_values = torch.clamp(c_values, min=0)
    maximum = float(c_values[-1])
    if maximum <= 0:
        raise ValueError("code covariance has empty support")
    support = c_values > float(support_rtol) * maximum
    support_rank = int(support.sum())
    active_values = c_values[support]
    active_vectors = c_vectors[:, support]
    covariance_sqrt = (
        active_vectors @ torch.diag(torch.sqrt(active_values)) @ active_vectors.T
    ).contiguous()
    covariance_inverse_sqrt = (
        active_vectors @ torch.diag(torch.rsqrt(active_values)) @ active_vectors.T
    ).contiguous()
    balanced = covariance_sqrt @ observability @ covariance_sqrt
    balanced = ((balanced + balanced.T) / 2).contiguous()
    # Solve only inside supp(C).  An n-dimensional eigensolve may arbitrarily mix
    # response-null supported directions with directions outside supp(C), making the
    # nominal full-support projector depend on an irrelevant null-space gauge.
    active_balanced = active_vectors.T @ balanced @ active_vectors
    active_balanced = ((active_balanced + active_balanced.T) / 2).contiguous()
    active_response, active_coordinates = torch.linalg.eigh(active_balanced)
    active_order = torch.argsort(active_response, descending=True)
    active_response = torch.clamp(active_response[active_order], min=0)
    active_directions = active_vectors @ active_coordinates[:, active_order]
    inactive_vectors = c_vectors[:, ~support]
    vectors = torch.cat((active_directions, inactive_vectors), dim=1).contiguous()
    values = torch.cat((
        active_response,
        torch.zeros(covariance.shape[0] - support_rank, dtype=torch.float64),
    )).contiguous()
    for column in range(vectors.shape[1]):
        pivot = int(torch.argmax(torch.abs(vectors[:, column])))
        if float(vectors[pivot, column]) < 0:
            vectors[:, column].neg_()
    natural = (covariance_sqrt @ vectors).contiguous()
    return PredictiveQuotient(
        covariance=covariance, observability=observability,
        covariance_sqrt=covariance_sqrt,
        covariance_inverse_sqrt=covariance_inverse_sqrt,
        balanced_operator=balanced, eigenvalues=values,
        hankel_singular_values=torch.sqrt(values).contiguous(),
        whitened_directions=vectors, natural_directions=natural,
        support_rank=support_rank, psd_rtol=float(psd_rtol),
        support_rtol=float(support_rtol),
    )
