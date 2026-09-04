"""CPU-only rank-one causal-spectral candidate construction.

For per-example donor differences ``d_i``, downstream gradients ``g_i``, and
positive full-head effects ``e_i``, this module constructs

    S = mean_i sym(d_i g_i^T / e_i).

The top *algebraic* eigenvector of ``S`` is a rank-one candidate direction for
a later finite interchange test.  It is not itself evidence that the direction
is a circuit: the construction is only a local, gradient-based approximation.

This file intentionally imports no model, data, runner, queue, or CUDA code.
All calculations are detached CPU float64 calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


DEFAULT_DENOMINATOR_FLOOR = 1e-12
DEFAULT_UNIT_ATOL = 1e-10


class CausalSpectralInputError(ValueError):
    """The local-response inputs do not define a safe spectral candidate."""


@dataclass(frozen=True)
class CausalSpectralDiagnostics:
    """Numerical diagnostics for a rank-one causal-spectral candidate."""

    sample_count: int
    ambient_dimension: int
    spectrum: torch.Tensor
    top_eigenvalue: float
    eigengap: float
    full_space_local_closure_mean_ratio: float
    full_space_local_closure_mean_absolute_error: float
    full_space_local_closure_max_absolute_error: float
    operator_trace: float
    operator_trace_identity_error: float
    symmetry_error: float
    unit_norm_error: float


@dataclass(frozen=True)
class CausalSpectralRankOne:
    """A canonical unit direction, its projector, and construction evidence."""

    direction: torch.Tensor
    projector: torch.Tensor
    operator: torch.Tensor
    diagnostics: CausalSpectralDiagnostics


def _require_cpu_float_matrix(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise CausalSpectralInputError(f"{name} must be a rank-2 torch tensor")
    if value.device.type != "cpu":
        raise CausalSpectralInputError(f"{name} must be on CPU")
    if not value.is_floating_point():
        raise CausalSpectralInputError(f"{name} must have a floating dtype")
    if not bool(torch.isfinite(value).all()):
        raise CausalSpectralInputError(f"{name} contains a non-finite value")
    return value.detach().to(dtype=torch.float64)


def _require_effects(
    value: object,
    *,
    sample_count: int,
    denominator_floor: float,
) -> torch.Tensor:
    if not math.isfinite(denominator_floor) or denominator_floor <= 0:
        raise CausalSpectralInputError(
            "denominator_floor must be finite and strictly positive"
        )
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise CausalSpectralInputError(
            "full_head_effects must be a rank-1 torch tensor"
        )
    if value.device.type != "cpu":
        raise CausalSpectralInputError("full_head_effects must be on CPU")
    if not value.is_floating_point():
        raise CausalSpectralInputError("full_head_effects must have a floating dtype")
    if value.shape[0] != sample_count:
        raise CausalSpectralInputError(
            "full_head_effects must have one value per example"
        )
    effects = value.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(effects).all()):
        raise CausalSpectralInputError("full_head_effects contains a non-finite value")
    if not bool((effects > denominator_floor).all()):
        raise CausalSpectralInputError(
            "full_head_effects must all be positive and strictly above "
            "denominator_floor"
        )
    return effects


def _require_sample_weights(
    value: object | None,
    *,
    sample_count: int,
) -> torch.Tensor:
    """Return normalized nonnegative CPU float64 sample weights."""

    if value is None:
        return torch.full(
            (sample_count,), 1.0 / sample_count, dtype=torch.float64
        )
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise CausalSpectralInputError("sample_weights must be a rank-1 torch tensor")
    if value.device.type != "cpu" or not value.is_floating_point():
        raise CausalSpectralInputError("sample_weights must be a floating CPU tensor")
    if value.shape[0] != sample_count:
        raise CausalSpectralInputError("sample_weights must have one value per example")
    weights = value.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
        raise CausalSpectralInputError("sample_weights must be finite and nonnegative")
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise CausalSpectralInputError("sample_weights must have a positive sum")
    return weights / total


def canonicalize_direction_sign(direction: torch.Tensor) -> torch.Tensor:
    """Choose one deterministic representative of a rank-one projector.

    The largest-magnitude coordinate is made positive.  ``torch.argmax`` uses
    the first coordinate in an exact tie, so the convention is deterministic.
    """

    if not isinstance(direction, torch.Tensor) or direction.ndim != 1:
        raise CausalSpectralInputError("direction must be a rank-1 torch tensor")
    if direction.device.type != "cpu" or not direction.is_floating_point():
        raise CausalSpectralInputError("direction must be a floating CPU tensor")
    result = direction.detach().to(dtype=torch.float64).clone()
    if result.numel() == 0 or not bool(torch.isfinite(result).all()):
        raise CausalSpectralInputError("direction must be nonempty and finite")
    norm = torch.linalg.vector_norm(result)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise CausalSpectralInputError("direction must have nonzero finite norm")
    result /= norm
    pivot = int(torch.argmax(torch.abs(result)))
    if float(result[pivot]) < 0:
        result = -result
    return result


def causal_spectral_rank_one(
    head_deltas: torch.Tensor,
    downstream_gradients: torch.Tensor,
    full_head_effects: torch.Tensor,
    *,
    denominator_floor: float = DEFAULT_DENOMINATOR_FLOOR,
    sample_weights: torch.Tensor | None = None,
) -> CausalSpectralRankOne:
    """Return the top-algebraic rank-one local causal candidate.

    Full-space local closure compares the gradient prediction ``d_i dot g_i``
    with the actual finite full-head effect ``e_i``.  Its ideal normalized
    ratio is one.  This diagnoses the local approximation; it does not validate
    any lower-rank finite intervention.
    """

    deltas = _require_cpu_float_matrix(head_deltas, name="head_deltas")
    gradients = _require_cpu_float_matrix(
        downstream_gradients, name="downstream_gradients"
    )
    if deltas.shape != gradients.shape:
        raise CausalSpectralInputError(
            "head_deltas and downstream_gradients must have identical shapes"
        )
    sample_count, ambient_dimension = deltas.shape
    if sample_count == 0:
        raise CausalSpectralInputError("at least one example is required")
    if ambient_dimension < 2:
        raise CausalSpectralInputError(
            "ambient dimension must be at least two to define an eigengap"
        )
    effects = _require_effects(
        full_head_effects,
        sample_count=sample_count,
        denominator_floor=denominator_floor,
    )
    weights = _require_sample_weights(sample_weights, sample_count=sample_count)

    scaled_deltas = deltas / effects[:, None]
    raw_operator = torch.einsum(
        "n,ni,nj->ij", weights, scaled_deltas, gradients
    )
    operator = 0.5 * (raw_operator + raw_operator.T)
    if not bool(torch.isfinite(operator).all()):
        raise CausalSpectralInputError(
            "normalization produced a non-finite causal-spectral operator"
        )

    spectrum, eigenvectors = torch.linalg.eigh(operator)
    if not bool(torch.isfinite(spectrum).all()) or not bool(
        torch.isfinite(eigenvectors).all()
    ):
        raise CausalSpectralInputError("eigendecomposition produced non-finite values")
    direction = canonicalize_direction_sign(eigenvectors[:, -1])
    projector = torch.outer(direction, direction)

    full_space_responses = torch.sum(deltas * gradients, dim=1) / effects
    if not bool(torch.isfinite(full_space_responses).all()):
        raise CausalSpectralInputError(
            "normalization produced a non-finite full-space local response"
        )
    mean_full_space_response = float(torch.sum(weights * full_space_responses))
    closure_errors = torch.abs(full_space_responses - 1.0)
    operator_trace = float(torch.trace(operator))
    diagnostics = CausalSpectralDiagnostics(
        sample_count=sample_count,
        ambient_dimension=ambient_dimension,
        spectrum=spectrum.clone(),
        top_eigenvalue=float(spectrum[-1]),
        eigengap=float(spectrum[-1] - spectrum[-2]),
        full_space_local_closure_mean_ratio=mean_full_space_response,
        full_space_local_closure_mean_absolute_error=float(
            torch.sum(weights * closure_errors)
        ),
        full_space_local_closure_max_absolute_error=float(torch.max(closure_errors)),
        operator_trace=operator_trace,
        operator_trace_identity_error=abs(
            operator_trace - mean_full_space_response
        ),
        symmetry_error=float(torch.max(torch.abs(operator - operator.T))),
        unit_norm_error=abs(float(torch.linalg.vector_norm(direction)) - 1.0),
    )
    return CausalSpectralRankOne(
        direction=direction,
        projector=projector,
        operator=operator,
        diagnostics=diagnostics,
    )


def normalized_rank_one_local_responses(
    head_deltas: torch.Tensor,
    downstream_gradients: torch.Tensor,
    full_head_effects: torch.Tensor,
    direction: torch.Tensor,
    *,
    denominator_floor: float = DEFAULT_DENOMINATOR_FLOOR,
    unit_atol: float = DEFAULT_UNIT_ATOL,
) -> torch.Tensor:
    """Score the local response of a unit rank-one projector per example.

    The response is ``(d_i dot u) (g_i dot u) / e_i`` and therefore is exactly
    invariant to replacing ``u`` with ``-u``.
    """

    deltas = _require_cpu_float_matrix(head_deltas, name="head_deltas")
    gradients = _require_cpu_float_matrix(
        downstream_gradients, name="downstream_gradients"
    )
    if deltas.shape != gradients.shape or deltas.shape[0] == 0:
        raise CausalSpectralInputError(
            "head_deltas and downstream_gradients must have the same nonempty shape"
        )
    effects = _require_effects(
        full_head_effects,
        sample_count=deltas.shape[0],
        denominator_floor=denominator_floor,
    )
    if not math.isfinite(unit_atol) or unit_atol <= 0:
        raise CausalSpectralInputError("unit_atol must be finite and positive")
    if not isinstance(direction, torch.Tensor) or direction.ndim not in (1, 2):
        raise CausalSpectralInputError("direction must have shape [dimension] or [dimension, 1]")
    if direction.ndim == 2:
        if direction.shape[1] != 1:
            raise CausalSpectralInputError("a rank-one frame must have one column")
        direction = direction[:, 0]
    if direction.device.type != "cpu" or not direction.is_floating_point():
        raise CausalSpectralInputError("direction must be a floating CPU tensor")
    vector = direction.detach().to(dtype=torch.float64)
    if vector.shape != (deltas.shape[1],) or not bool(torch.isfinite(vector).all()):
        raise CausalSpectralInputError("direction has the wrong shape or is non-finite")
    norm = float(torch.linalg.vector_norm(vector))
    if not math.isfinite(norm) or abs(norm - 1.0) > unit_atol:
        raise CausalSpectralInputError("direction must be unit norm")
    return (deltas @ vector) * (gradients @ vector) / effects


def deterministic_rank_one_haar_frames(
    ambient_dimension: int,
    seeds: Sequence[int],
) -> torch.Tensor:
    """Return deterministic matched-rank Haar controls of shape ``[K, D, 1]``.

    Normalizing an isotropic Gaussian gives a Haar-uniform one-dimensional
    subspace.  Canonical signs make serialized frame representatives stable;
    they do not alter the sampled projectors.
    """

    if type(ambient_dimension) is not int or ambient_dimension < 2:
        raise CausalSpectralInputError("ambient_dimension must be an integer >= 2")
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
        raise CausalSpectralInputError("seeds must be a nonempty sequence of integers")
    seeds = tuple(seeds)
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise CausalSpectralInputError("seeds must be a nonempty sequence of integers")
    if len(set(seeds)) != len(seeds):
        raise CausalSpectralInputError("Haar-control seeds must be unique")
    if any(seed < 0 or seed >= 2**63 for seed in seeds):
        raise CausalSpectralInputError("Haar-control seeds must lie in [0, 2**63)")

    frames = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        direction = torch.randn(
            ambient_dimension, generator=generator, dtype=torch.float64
        )
        frames.append(canonicalize_direction_sign(direction)[:, None])
    return torch.stack(frames, dim=0)
