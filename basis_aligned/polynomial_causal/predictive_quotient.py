"""Gauge-invariant finite-horizon predictive quotient for code edits.

This is a pure CPU mathematics module.  It loads no model or corpus and grants no
experimental authority.  Given a natural code covariance C and a downstream local
response metric O, it solves the one-interface balanced edit-reduction problem defined in
MATHEMATICAL_REVIEW_2026-08-28_0630.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F


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
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    if float(eigenvalues[0]) < -rtol * scale:
        raise ValueError(f"{name} is not positive semidefinite within tolerance")
    # A tolerance-accepted tiny negative eigenvalue must not survive into a nominal
    # quadratic metric.  Return the nearest PSD matrix in this eigensystem.
    matrix = eigenvectors @ torch.diag(torch.clamp(eigenvalues, min=0)) @ eigenvectors.T
    return ((matrix + matrix.T) / 2).contiguous()


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

    Every entry along the leading dimensions is one ``J_c^T r_cp`` vector, where the
    registered output probe has conditional covariance ``E[r r^T | c] = F_c``.
    Averaging outer products is then an unbiased estimator of the local downstream
    Fisher/response Gramian.
    """

    outer_sum, count = vjp_outer_product_sum(gradients)
    return (outer_sum / count).contiguous()


def vjp_outer_product_sum(gradients: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return mergeable float64 sufficient statistics for VJP sketches."""

    if not torch.is_tensor(gradients) or gradients.ndim < 2 or min(
        gradients.shape
    ) <= 0 or not bool(torch.isfinite(gradients).all()):
        raise ValueError("VJP sketches must be finite [..., code_dim]")
    values = gradients.detach().cpu().double().reshape(-1, gradients.shape[-1])
    return (values.T @ values).contiguous(), int(len(values))


def categorical_fisher_probe_ids(
    logits: torch.Tensor,
    probe_seeds: Sequence[int],
    *,
    score_start: int = 64,
    score_stop: int = 256,
) -> torch.Tensor:
    """Draw reproducible inverse-CDF categorical probes from detached model logits.

    The returned CPU tensor has shape ``[probe, batch, scored_position]``.  Random
    uniforms are generated in float64 on CPU, so the registered random stream does not
    depend on the accelerator RNG. The categorical CDF is evaluated at the registered
    float32 softmax precision; target IDs must be hashed into the experiment receipt.
    """

    if not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[0] <= 0 or (
        logits.shape[2] <= 1
    ) or not bool(torch.isfinite(logits.detach()).all()):
        raise ValueError("logits must be finite [batch, position, vocabulary]")
    if type(score_start) is not int or type(score_stop) is not int or not (
        0 <= score_start < score_stop <= logits.shape[1]
    ):
        raise ValueError("scored support is outside the logit trajectory")
    seeds = tuple(probe_seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(
        set(seeds)
    ) != len(seeds):
        raise ValueError("probe seeds must be distinct nonnegative integers")

    scored = logits.detach()[:, score_start:score_stop].float()
    batch, positions, vocabulary = scored.shape
    probabilities = torch.softmax(scored, dim=-1).reshape(-1, vocabulary)
    cdf = torch.cumsum(probabilities, dim=-1)
    # Guarantee that roundoff at the last bin cannot return vocabulary as an index.
    cdf[:, -1] = 1.0
    uniforms = torch.stack([
        torch.rand(
            batch * positions, generator=torch.Generator(device="cpu").manual_seed(seed),
            dtype=torch.float64,
        )
        for seed in seeds
    ], dim=1).to(device=cdf.device, dtype=cdf.dtype)
    # ``right=True`` skips zero-probability CDF plateaus even for the possible exact
    # CPU-uniform draw u=0. Since torch.rand never returns 1, the final bin is safe.
    sampled = torch.searchsorted(cdf, uniforms, right=True).T
    if sampled.shape != (len(seeds), batch * positions) or bool((sampled < 0).any()) or (
        bool((sampled >= vocabulary).any())
    ):
        raise RuntimeError("categorical Fisher probe sampling failed")
    return sampled.reshape(len(seeds), batch, positions).detach().cpu().long().contiguous()


def fisher_vjp_sketches(
    codes: torch.Tensor,
    logits: torch.Tensor,
    probe_token_ids: torch.Tensor,
    *,
    score_start: int = 64,
    score_stop: int = 256,
) -> torch.Tensor:
    """Backpropagate categorical score probes to every scored code position.

    For independent ``y_t ~ p_t``, ``grad_z sum_t log p_t(y_t)`` has conditional
    second moment equal to the sum of the output-position Fisher blocks.  The returned
    CPU float64 array is ``[probe, batch, input_position, code_dim]``.  Keeping the
    per-position gradients before the outer-product reduction matches the project's
    intervention: one code direction is written at one registered position per row.

    This function intentionally consumes the supplied autograd graph on its final
    probe.  A source-closed caller must therefore own the tensors and call it once.
    """

    if not torch.is_tensor(codes) or codes.ndim != 3 or codes.shape[0] <= 0 or (
        codes.shape[2] <= 0
    ) or not codes.requires_grad or not bool(torch.isfinite(codes.detach()).all()):
        raise ValueError("codes must be finite graph-bearing [batch, position, code_dim]")
    if not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[:2] != (
        codes.shape[:2]
    ) or logits.shape[2] <= 1 or not logits.requires_grad or not bool(
        torch.isfinite(logits.detach()).all()
    ):
        raise ValueError("logits must be finite graph-bearing and align with codes")
    if type(score_start) is not int or type(score_stop) is not int or not (
        0 <= score_start < score_stop <= codes.shape[1]
    ):
        raise ValueError("scored support is outside the code/logit trajectory")
    expected = (codes.shape[0], score_stop - score_start)
    if not torch.is_tensor(probe_token_ids) or probe_token_ids.dtype != torch.long or (
        probe_token_ids.ndim != 3
    ) or tuple(probe_token_ids.shape[1:]) != expected or probe_token_ids.shape[0] <= 0 or (
        bool((probe_token_ids < 0).any())
    ) or bool((probe_token_ids >= logits.shape[2]).any()):
        raise ValueError("probe token IDs must be [probe, batch, scored_position]")

    scored_logp = F.log_softmax(logits[:, score_start:score_stop].float(), dim=-1)
    sketches = []
    targets = probe_token_ids.to(device=logits.device)
    for index in range(targets.shape[0]):
        log_likelihood = torch.gather(
            scored_logp, -1, targets[index].unsqueeze(-1),
        ).sum()
        (gradient,) = torch.autograd.grad(
            log_likelihood, codes, retain_graph=index + 1 < targets.shape[0],
            create_graph=False, allow_unused=False,
        )
        selected = gradient[:, score_start:score_stop]
        if tuple(selected.shape) != (
            codes.shape[0], score_stop - score_start, codes.shape[2],
        ) or not bool(torch.isfinite(selected).all()):
            raise RuntimeError("Fisher VJP has malformed shape or values")
        sketches.append(selected.detach().cpu().double().contiguous())
    return torch.stack(sketches).contiguous()


@dataclass(frozen=True)
class PredictiveQuotient:
    """Balanced coordinates and exact quadratic edit-tail certificate."""

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
        """Optimal rank-d edit reconstruction map in natural coordinates."""

        rank = self._rank(rank)
        directions = self.whitened_directions[:, :rank]
        return (
            self.covariance_sqrt @ directions @ directions.T
            @ self.covariance_inverse_sqrt
        ).contiguous()

    def discarded_quadratic_response(self, rank: int) -> float:
        """Minimum E[(delta-delta_hat)^T O (delta-delta_hat)] at this rank."""

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


@dataclass(frozen=True)
class PredictiveQuotientStability:
    """Scale-free comparison of two independently estimated quotient objects."""

    relative_trace_difference: float
    normalized_spectrum_l1: float
    left_fraction_rank: int
    right_fraction_rank: int
    comparison_rank: int
    normalized_chordal_distance: float


def selected_rank_with_gap(
    quotient: PredictiveQuotient,
    *,
    retained_fraction: float = 0.95,
    minimum_gap_ratio: float = 2.0,
) -> int | None:
    """Apply the preregistered retained-response plus spectral-gap rank rule."""

    if not isinstance(quotient, PredictiveQuotient) or not isinstance(
        retained_fraction, (int, float)
    ) or not 0 < float(retained_fraction) < 1 or not isinstance(
        minimum_gap_ratio, (int, float)
    ) or not math.isfinite(float(minimum_gap_ratio)) or float(minimum_gap_ratio) <= 1:
        raise ValueError("rank rule inputs are malformed")
    rank = quotient.rank_for_fraction(float(retained_fraction))
    if rank <= 0 or rank >= quotient.support_rank:
        return None
    denominator = float(quotient.eigenvalues[rank])
    numerator = float(quotient.eigenvalues[rank - 1])
    if denominator <= 0:
        gap = math.inf if numerator > 0 else 1.0
    else:
        gap = numerator / denominator
    return rank if gap >= float(minimum_gap_ratio) else None


def compare_predictive_quotients(
    left: PredictiveQuotient,
    right: PredictiveQuotient,
    *,
    comparison_rank: int,
    retained_fraction: float = 0.95,
) -> PredictiveQuotientStability:
    """Compare stochastic/data splits without choosing new thresholds.

    Chordal distance is computed between orthogonal projectors in the common
    covariance-whitened coordinates.  This is the standard subspace distance; the
    natural-coordinate reconstruction maps need not themselves be orthogonal.
    """

    if not isinstance(left, PredictiveQuotient) or not isinstance(
        right, PredictiveQuotient
    ) or left.code_dim != right.code_dim or left.support_rank != right.support_rank or (
        not torch.allclose(left.covariance, right.covariance, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("stability comparison requires one common covariance support")
    if type(comparison_rank) is not int or not 1 <= comparison_rank < left.support_rank:
        raise ValueError("comparison rank must be a nontrivial covariance-support cut")
    if not isinstance(retained_fraction, (int, float)) or not (
        0 < float(retained_fraction) < 1
    ):
        raise ValueError("retained fraction must lie strictly in (0,1)")
    left_trace = left.total_quadratic_response
    right_trace = right.total_quadratic_response
    mean_trace = (left_trace + right_trace) / 2
    if left_trace <= 0 or right_trace <= 0 or mean_trace <= 0:
        raise ValueError("stability comparison requires positive response trace")
    left_spectrum = left.eigenvalues / left_trace
    right_spectrum = right.eigenvalues / right_trace
    left_vectors = left.whitened_directions[:, :comparison_rank]
    right_vectors = right.whitened_directions[:, :comparison_rank]
    left_projector = left_vectors @ left_vectors.T
    right_projector = right_vectors @ right_vectors.T
    chordal = float(torch.linalg.matrix_norm(left_projector - right_projector)) / math.sqrt(
        2 * comparison_rank
    )
    return PredictiveQuotientStability(
        relative_trace_difference=abs(left_trace - right_trace) / mean_trace,
        normalized_spectrum_l1=float(torch.sum(torch.abs(
            left_spectrum - right_spectrum
        ))),
        left_fraction_rank=left.rank_for_fraction(float(retained_fraction)),
        right_fraction_rank=right.rank_for_fraction(float(retained_fraction)),
        comparison_rank=comparison_rank,
        normalized_chordal_distance=chordal,
    )


def solve_predictive_quotient(
    covariance: torch.Tensor,
    observability: torch.Tensor,
    *,
    psd_rtol: float = 1e-10,
    support_rtol: float = 1e-12,
) -> PredictiveQuotient:
    """Solve the covariance/observability balanced interface reduction.

    If an edit delta is drawn independently with covariance C and its constant/mean
    downstream metric is O, the optimal rank-d linear edit reconstruction has
    discarded distortion sum_{i>d} lambda_i, where lambda_i are the descending
    eigenvalues of C^{1/2} O C^{1/2}.  This is an Eckart--Young/Ky Fan statement after
    whitening.  It is not an exact paired-state reconstruction theorem when a
    context-specific O_x is correlated with the natural state z_x.
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
