"""Pure CPU statistics for the MLP1 empirical-moment discriminator.

Numerical conventions
---------------------
* Inputs are copied to contiguous CPU ``float64`` tensors.  Population moments
  use divisor ``N`` and deterministic, caller-specified update order.
* Covariances are symmetrized before ``torch.linalg.eigh``.  Eigenvectors are
  sign-oriented by their first largest-absolute coordinate.  Mean-augmented
  bases use the normalized mean followed by ordered, twice-reorthogonalized
  centered PCs.  The latter is the explicit deterministic QR convention used
  here; frozen experiments must serialize projectors, not rerun an eigensolver.
* A probe Gram is ``N**-1 sum_n <H_i(x_n), H_j(x_n)>``.  The Wick routines are
  exact only for the noncentral Gaussian surrogate with the supplied mean and
  covariance; they make no Gaussian claim about natural activations.
* The document bootstrap resamples ``D`` documents ``D`` times per draw, uses
  common draws for every returned coordinate, and takes the exact, uninterpolated
  ``ceil(confidence * draws)`` order statistic of the replicatewise maximum
  absolute centered error.

This module deliberately has no filesystem, model, checkpoint, or row-lifecycle
capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

import numpy as np
import torch


FLOAT64_EPS = torch.finfo(torch.float64).eps
PROJECTOR_RELATIVE_FROBENIUS_TOLERANCE = 1e-10
MEAN_RATIO_THRESHOLD = 1e-8


def _owned_cpu_float64(value: torch.Tensor, *, ndim: int, name: str) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.ndim != ndim
        or value.device.type != "cpu"
        or value.dtype not in (torch.float32, torch.float64)
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} must be a finite, graph-free CPU float tensor")
    return value.detach().to(dtype=torch.float64).contiguous().clone()


def _owned_exact_cpu_float64(
    value: torch.Tensor, *, ndim: int, name: str,
) -> torch.Tensor:
    result = _owned_cpu_float64(value, ndim=ndim, name=name)
    if value.dtype != torch.float64:
        raise ValueError(f"{name} must be float64; deployed float32 belongs to replay code")
    return result


def _orient_columns(matrix: torch.Tensor) -> torch.Tensor:
    oriented = matrix.clone()
    for column in range(oriented.shape[1]):
        vector = oriented[:, column]
        pivot = int(torch.argmax(torch.abs(vector)))
        if float(vector[pivot]) < 0.0:
            oriented[:, column].neg_()
    return oriented


def _ordered_reorthogonalize(
    candidates: torch.Tensor,
    *,
    required: int,
    excluded: Sequence[torch.Tensor] = (),
) -> torch.Tensor:
    """Ordered QR via two-pass modified Gram--Schmidt, with fixed signs."""
    if required < 0 or required > candidates.shape[0]:
        raise ValueError("invalid requested basis rank")
    if required == 0:
        return torch.empty((candidates.shape[0], 0), dtype=torch.float64)
    accepted: list[torch.Tensor] = []
    exclusions = [item.clone() for item in excluded]
    threshold = 100.0 * FLOAT64_EPS * max(1.0, float(torch.linalg.norm(candidates)))
    for raw in candidates.T:
        vector = raw.clone()
        for _ in range(2):
            for direction in (*exclusions, *accepted):
                vector -= torch.dot(direction, vector) * direction
        norm = float(torch.linalg.norm(vector))
        if norm <= threshold:
            continue
        vector /= norm
        pivot = int(torch.argmax(torch.abs(vector)))
        if float(vector[pivot]) < 0.0:
            vector.neg_()
        accepted.append(vector)
        if len(accepted) == required:
            break
    if len(accepted) != required:
        raise ValueError("candidate columns do not span the requested subspace")
    return torch.stack(accepted, dim=1)


@dataclass(frozen=True)
class PopulationMoments:
    count: int
    mean: torch.Tensor
    centered_outer_sum: torch.Tensor

    @property
    def covariance(self) -> torch.Tensor:
        covariance = self.centered_outer_sum / self.count
        return (covariance + covariance.T) / 2.0

    @property
    def input_rms(self) -> float:
        second_moment = (
            torch.trace(self.centered_outer_sum) / self.count
            + torch.dot(self.mean, self.mean)
        )
        return math.sqrt(max(0.0, float(second_moment)))


class StreamingPopulationMoments:
    """Fixed-order Chan accumulation of population sufficient statistics."""

    def __init__(self, dimension: int):
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self._dimension = dimension
        self._count = 0
        self._mean = torch.zeros(dimension, dtype=torch.float64)
        self._m2 = torch.zeros((dimension, dimension), dtype=torch.float64)

    def update(self, rows: torch.Tensor) -> None:
        batch = _owned_cpu_float64(rows, ndim=2, name="rows")
        if batch.shape[1] != self._dimension or batch.shape[0] == 0:
            raise ValueError("rows have the wrong or empty shape")
        batch_count = int(batch.shape[0])
        batch_mean = batch.mean(dim=0)
        centered = batch - batch_mean
        batch_m2 = centered.T @ centered
        if self._count == 0:
            self._count = batch_count
            self._mean.copy_(batch_mean)
            self._m2.copy_(batch_m2)
            return
        total = self._count + batch_count
        delta = batch_mean - self._mean
        self._m2 += batch_m2 + torch.outer(delta, delta) * (
            self._count * batch_count / total
        )
        self._mean += delta * (batch_count / total)
        self._count = total

    def merge(self, other: PopulationMoments) -> None:
        _validate_population_moments(other)
        if other.mean.numel() != self._dimension:
            raise ValueError("moment dimensions differ")
        if self._count == 0:
            self._count = other.count
            self._mean.copy_(other.mean)
            self._m2.copy_(other.centered_outer_sum)
            return
        total = self._count + other.count
        delta = other.mean - self._mean
        self._m2 += other.centered_outer_sum + torch.outer(delta, delta) * (
            self._count * other.count / total
        )
        self._mean += delta * (other.count / total)
        self._count = total

    def finalize(self) -> PopulationMoments:
        if self._count <= 0:
            raise ValueError("cannot finalize empty moments")
        return PopulationMoments(
            count=self._count,
            mean=self._mean.clone(),
            centered_outer_sum=((self._m2 + self._m2.T) / 2.0).clone(),
        )


def _validate_population_moments(moments: PopulationMoments) -> None:
    if (
        not isinstance(moments, PopulationMoments)
        or not isinstance(moments.count, int)
        or isinstance(moments.count, bool)
        or moments.count <= 0
        or moments.mean.ndim != 1
        or moments.mean.dtype != torch.float64
        or moments.mean.device.type != "cpu"
        or moments.mean.requires_grad
        or tuple(moments.centered_outer_sum.shape)
        != (moments.mean.numel(), moments.mean.numel())
        or moments.centered_outer_sum.dtype != torch.float64
        or moments.centered_outer_sum.device.type != "cpu"
        or moments.centered_outer_sum.requires_grad
        or not bool(torch.isfinite(moments.mean).all())
        or not bool(torch.isfinite(moments.centered_outer_sum).all())
    ):
        raise ValueError("population moments are malformed")


@dataclass(frozen=True)
class ProjectorFamily:
    ranks: tuple[int, ...]
    bases: tuple[torch.Tensor, ...]
    projectors: tuple[torch.Tensor, ...]

    def at_rank(self, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.ranks.index(rank)
        return self.bases[index], self.projectors[index]


@dataclass(frozen=True)
class PopulationProjectors:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    mean_ratio: float
    mean_present: bool
    pca_no_mean: ProjectorFamily
    mean_plus_pca: ProjectorFamily
    degenerate_boundaries: tuple[bool, ...]


def _validate_projector(projector: torch.Tensor, tolerance: float) -> None:
    denominator = max(1.0, float(torch.linalg.norm(projector)))
    symmetry = float(torch.linalg.norm(projector - projector.T)) / denominator
    idempotence = float(torch.linalg.norm(projector @ projector - projector)) / denominator
    if symmetry > tolerance or idempotence > tolerance:
        raise ArithmeticError("constructed projector failed replay tolerance")


def build_population_projectors(
    moments: PopulationMoments,
    ranks: Iterable[int],
    *,
    mean_ratio_threshold: float = MEAN_RATIO_THRESHOLD,
    projector_tolerance: float = PROJECTOR_RELATIVE_FROBENIUS_TOLERANCE,
) -> PopulationProjectors:
    _validate_population_moments(moments)
    dimension = moments.mean.numel()
    ordered_ranks = tuple(sorted(set(ranks)))
    if (
        not ordered_ranks
        or any(not isinstance(rank, int) or isinstance(rank, bool) for rank in ordered_ranks)
        or ordered_ranks[0] <= 0
        or ordered_ranks[-1] > dimension
        or not math.isfinite(mean_ratio_threshold)
        or mean_ratio_threshold < 0.0
        or not math.isfinite(projector_tolerance)
        or projector_tolerance <= 0.0
    ):
        raise ValueError("projector construction arguments are malformed")

    covariance = moments.covariance
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = _orient_columns(eigenvectors.flip(1))
    largest = max(1.0, abs(float(eigenvalues[0])))
    negative_tolerance = 100.0 * FLOAT64_EPS * largest
    if float(eigenvalues[-1]) < -negative_tolerance:
        raise ValueError("covariance is not positive semidefinite within tolerance")
    eigenvalues = eigenvalues.clamp_min(0.0)

    input_rms = moments.input_rms
    mean_norm = float(torch.linalg.norm(moments.mean))
    mean_ratio = mean_norm / input_rms if input_rms > 0.0 else 0.0
    mean_present = mean_ratio > mean_ratio_threshold
    mean_direction = moments.mean / mean_norm if mean_present else None

    pca_bases: list[torch.Tensor] = []
    pca_projectors: list[torch.Tensor] = []
    mean_bases: list[torch.Tensor] = []
    mean_projectors: list[torch.Tensor] = []
    boundaries: list[bool] = []
    for rank in ordered_ranks:
        pca_basis = eigenvectors[:, :rank].clone()
        pca_projector = pca_basis @ pca_basis.T
        _validate_projector(pca_projector, projector_tolerance)
        pca_bases.append(pca_basis)
        pca_projectors.append(pca_projector)

        if mean_present:
            assert mean_direction is not None
            remainder = _ordered_reorthogonalize(
                eigenvectors,
                required=rank - 1,
                excluded=(mean_direction,),
            )
            mean_basis = torch.cat((mean_direction[:, None], remainder), dim=1)
        else:
            mean_basis = pca_basis.clone()
        mean_projector = mean_basis @ mean_basis.T
        _validate_projector(mean_projector, projector_tolerance)
        mean_bases.append(mean_basis)
        mean_projectors.append(mean_projector)

        if rank == dimension:
            boundaries.append(False)
        else:
            gap = abs(float(eigenvalues[rank - 1] - eigenvalues[rank]))
            boundaries.append(gap < 100.0 * FLOAT64_EPS * largest)

    return PopulationProjectors(
        eigenvalues=eigenvalues.clone(),
        eigenvectors=eigenvectors.clone(),
        mean_ratio=mean_ratio,
        mean_present=mean_present,
        pca_no_mean=ProjectorFamily(
            ordered_ranks, tuple(pca_bases), tuple(pca_projectors),
        ),
        mean_plus_pca=ProjectorFamily(
            ordered_ranks, tuple(mean_bases), tuple(mean_projectors),
        ),
        degenerate_boundaries=tuple(boundaries),
    )


def deterministic_haar_basis(
    dimension: int,
    rank: int,
    *,
    seed: int,
    mean_direction: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate the addendum's deterministic PCG64DXSM Haar basis."""
    if (
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
        or not isinstance(rank, int) or isinstance(rank, bool) or not 0 < rank <= dimension
        or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
    ):
        raise ValueError("Haar arguments are malformed")
    excluded: tuple[torch.Tensor, ...] = ()
    if mean_direction is not None:
        mean = _owned_cpu_float64(mean_direction, ndim=1, name="mean_direction")
        if mean.numel() != dimension or float(torch.linalg.norm(mean)) == 0.0:
            raise ValueError("mean direction is malformed")
        mean /= torch.linalg.norm(mean)
        excluded = (mean,)
        if rank == dimension:
            raise ValueError("cannot draw a full basis orthogonal to a mean direction")
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    candidates = torch.from_numpy(generator.standard_normal((dimension, rank)))
    return _ordered_reorthogonalize(candidates, required=rank, excluded=excluded)


@dataclass(frozen=True)
class BilinearFactors:
    """Exact bias-free factors for ``down @ ((left @ x) * (right @ x))``.

    These are analytical float64 factors, not cached teacher writes or deployed
    float32 factors.  Bias terms are intentionally outside this type because all
    registered residual probes cancel them.
    """

    down: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor

    def __post_init__(self) -> None:
        down = _owned_exact_cpu_float64(self.down, ndim=2, name="down")
        left = _owned_exact_cpu_float64(self.left, ndim=2, name="left")
        right = _owned_exact_cpu_float64(self.right, ndim=2, name="right")
        if left.shape != right.shape or down.shape[1] != left.shape[0]:
            raise ValueError("bilinear factor shapes are incompatible")
        object.__setattr__(self, "down", down)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    @property
    def input_dimension(self) -> int:
        return int(self.left.shape[1])

    @property
    def output_dimension(self) -> int:
        return int(self.down.shape[0])


def combine_bilinear_factors(
    terms: Sequence[BilinearFactors], coefficients: Sequence[float],
) -> BilinearFactors:
    if not terms or len(terms) != len(coefficients):
        raise ValueError("factor combination is empty or mismatched")
    input_dimension = terms[0].input_dimension
    output_dimension = terms[0].output_dimension
    if any(
        term.input_dimension != input_dimension or term.output_dimension != output_dimension
        for term in terms
    ) or any(not math.isfinite(float(coefficient)) for coefficient in coefficients):
        raise ValueError("factor combination is incompatible or nonfinite")
    return BilinearFactors(
        down=torch.cat(
            [term.down * float(coefficient) for term, coefficient in zip(terms, coefficients)],
            dim=1,
        ),
        left=torch.cat([term.left for term in terms], dim=0),
        right=torch.cat([term.right for term in terms], dim=0),
    )


def teacher_minus_candidate_factors(
    teacher: BilinearFactors, candidate: BilinearFactors,
) -> BilinearFactors:
    """Return the registered signed residual ``H = teacher - candidate``."""
    return combine_bilinear_factors((teacher, candidate), (1.0, -1.0))


def bilinear_output(factors: BilinearFactors, rows: torch.Tensor) -> torch.Tensor:
    inputs = _owned_cpu_float64(rows, ndim=2, name="rows")
    if inputs.shape[1] != factors.input_dimension:
        raise ValueError("bilinear input dimension differs")
    gates = (inputs @ factors.left.T) * (inputs @ factors.right.T)
    return gates @ factors.down.T


def evaluate_probe_bank(
    probes: Sequence[BilinearFactors], rows: torch.Tensor,
) -> torch.Tensor:
    if not probes:
        raise ValueError("probe bank is empty")
    outputs = [bilinear_output(probe, rows) for probe in probes]
    if len({tuple(output.shape) for output in outputs}) != 1:
        raise ValueError("probe outputs have different shapes")
    return torch.stack(outputs, dim=1)


class ProbeGramAccumulator:
    def __init__(self, probes: int):
        if not isinstance(probes, int) or isinstance(probes, bool) or probes <= 0:
            raise ValueError("probe count must be positive")
        self._probes = probes
        self._count = 0
        self._sum = torch.zeros((probes, probes), dtype=torch.float64)

    def update(self, outputs: torch.Tensor) -> None:
        values = _owned_cpu_float64(outputs, ndim=3, name="probe outputs")
        if values.shape[0] == 0 or values.shape[1] != self._probes:
            raise ValueError("probe outputs have the wrong or empty shape")
        self._sum += torch.einsum("npo,nqo->pq", values, values)
        self._count += int(values.shape[0])

    def finalize(self) -> torch.Tensor:
        if self._count == 0:
            raise ValueError("cannot finalize an empty probe Gram")
        gram = self._sum / self._count
        return ((gram + gram.T) / 2.0).clone()


@dataclass(frozen=True)
class DocumentProbeGramStatistics:
    document_ids: tuple[str, ...]
    gram_sums: torch.Tensor
    row_counts: torch.Tensor

    @property
    def pooled_gram(self) -> torch.Tensor:
        gram = self.gram_sums.sum(dim=0) / int(self.row_counts.sum())
        return (gram + gram.T) / 2.0


class DocumentProbeGramAccumulator:
    """First-seen ordered per-document sufficient statistics."""

    def __init__(self, probes: int):
        if not isinstance(probes, int) or isinstance(probes, bool) or probes <= 0:
            raise ValueError("probe count must be positive")
        self._probes = probes
        self._order: list[str] = []
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}

    def update(self, outputs: torch.Tensor, document_ids: Sequence[str]) -> None:
        values = _owned_cpu_float64(outputs, ndim=3, name="probe outputs")
        if (
            values.shape[1] != self._probes
            or len(document_ids) != values.shape[0]
            or any(not isinstance(item, str) or not item for item in document_ids)
        ):
            raise ValueError("document-labelled probe outputs are malformed")
        for document_id in dict.fromkeys(document_ids):
            if document_id not in self._sums:
                self._order.append(document_id)
                self._sums[document_id] = torch.zeros(
                    (self._probes, self._probes), dtype=torch.float64,
                )
                self._counts[document_id] = 0
        # Grouped einsums avoid a Python-level matrix multiplication for every row.
        # The first-seen document order remains independent of batching.
        for document_id in dict.fromkeys(document_ids):
            indices = torch.tensor(
                [index for index, item in enumerate(document_ids) if item == document_id],
                dtype=torch.int64,
            )
            block = values[indices]
            self._sums[document_id] += torch.einsum("npo,nqo->pq", block, block)
            self._counts[document_id] += int(block.shape[0])

    def finalize(self) -> DocumentProbeGramStatistics:
        if not self._order:
            raise ValueError("cannot finalize empty document statistics")
        return DocumentProbeGramStatistics(
            document_ids=tuple(self._order),
            gram_sums=torch.stack([self._sums[item] for item in self._order]),
            row_counts=torch.tensor(
                [self._counts[item] for item in self._order], dtype=torch.int64,
            ),
        )


def noncentral_gaussian_cross_inner_product(
    first: BilinearFactors,
    second: BilinearFactors,
    moments: PopulationMoments,
    *,
    block_size: int = 128,
) -> torch.Tensor:
    """Gaussian-surrogate ``E[<first(x), second(x)>]`` without a fourth tensor."""
    _validate_population_moments(moments)
    if (
        first.input_dimension != second.input_dimension
        or first.output_dimension != second.output_dimension
        or moments.mean.numel() != first.input_dimension
        or not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("Wick contraction arguments are incompatible")
    covariance = moments.covariance
    mean = moments.mean
    left2_sigma = second.left @ covariance
    right2_sigma = second.right @ covariance
    left2_mean = second.left @ mean
    right2_mean = second.right @ mean
    centered2 = torch.einsum("gi,ij,gj->g", second.left, covariance, second.right)
    total = torch.zeros((), dtype=torch.float64)
    gates = first.left.shape[0]
    for start in range(0, gates, block_size):
        stop = min(gates, start + block_size)
        left1 = first.left[start:stop]
        right1 = first.right[start:stop]
        left1_sigma = left1 @ covariance
        right1_sigma = right1 @ covariance
        ll = left1_sigma @ second.left.T
        rr = right1_sigma @ second.right.T
        lr = left1_sigma @ second.right.T
        rl = right1_sigma @ second.left.T
        left1_mean = left1 @ mean
        right1_mean = right1 @ mean
        centered1 = torch.einsum("gi,ij,gj->g", left1, covariance, right1)
        fourth = (
            torch.outer(centered1, centered2)
            + ll * rr
            + lr * rl
            + torch.outer(left1_mean * right1_mean, centered2)
            + torch.outer(centered1, left2_mean * right2_mean)
            + torch.outer(left1_mean, left2_mean) * rr
            + torch.outer(left1_mean, right2_mean) * rl
            + torch.outer(right1_mean, left2_mean) * lr
            + torch.outer(right1_mean, right2_mean) * ll
            + torch.outer(left1_mean * right1_mean, left2_mean * right2_mean)
        )
        # The expanded formula above is E[(l1 x)(r1 x)(l2 x)(r2 x)].
        # This identity is also checked against E[g1]E[g2] bookkeeping.
        if not bool(torch.isfinite(fourth).all()):
            raise ArithmeticError("nonfinite Wick contraction")
        down_cross = first.down[:, start:stop].T @ second.down
        total += torch.sum(down_cross * fourth)
    return total


def noncentral_gaussian_probe_gram(
    probes: Sequence[BilinearFactors],
    moments: PopulationMoments,
    *,
    block_size: int = 128,
) -> torch.Tensor:
    if not probes:
        raise ValueError("probe bank is empty")
    count = len(probes)
    gram = torch.empty((count, count), dtype=torch.float64)
    for first in range(count):
        for second in range(first, count):
            value = noncentral_gaussian_cross_inner_product(
                probes[first], probes[second], moments, block_size=block_size,
            )
            gram[first, second] = value
            gram[second, first] = value
    return gram


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    data = _owned_cpu_float64(values, ndim=1, name="rank values")
    if data.numel() == 0:
        raise ValueError("cannot rank an empty vector")
    order = torch.argsort(data, stable=True)
    ranks = torch.empty_like(data)
    position = 0
    while position < data.numel():
        end = position + 1
        while end < data.numel() and bool(data[order[end]] == data[order[position]]):
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2.0
        position = end
    return ranks


def spearman_average_rank(first: torch.Tensor, second: torch.Tensor) -> float | None:
    left = average_ranks(first)
    right = average_ranks(second)
    if left.numel() != right.numel():
        raise ValueError("Spearman vectors differ in length")
    left -= left.mean()
    right -= right.mean()
    denominator = torch.linalg.norm(left) * torch.linalg.norm(right)
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


@dataclass(frozen=True)
class SimultaneousBootstrapBand:
    point: torch.Tensor
    draws: torch.Tensor
    critical_value: float
    lower: torch.Tensor
    upper: torch.Tensor
    repetitions: int
    seed: int
    confidence: float
    critical_order_statistic_one_indexed: int
    documents_per_draw: int


def simultaneous_document_bootstrap(
    document_sums: torch.Tensor,
    row_counts: torch.Tensor,
    statistic: Callable[[torch.Tensor], torch.Tensor],
    *,
    repetitions: int,
    seed: int,
    confidence: float = 0.95,
    draw_chunk_size: int = 256,
) -> SimultaneousBootstrapBand:
    """Shared-document, row-weighted, two-sided basic max-error band.

    ``document_sums[d]`` is the sum (not mean) of row contributions in document
    ``d``.  ``statistic`` maps pooled row means of shape ``[..., *cell_shape]``
    to a final coordinate vector of shape ``[..., comparisons]``.  It must
    recompute every nonlinear contrast from each resampled pooled mean.
    """
    if not torch.is_tensor(document_sums):
        raise ValueError("document_sums must be a tensor")
    sums = _owned_cpu_float64(
        document_sums, ndim=document_sums.ndim, name="document_sums",
    )
    if sums.ndim < 2:
        raise ValueError("document sums need a document and cell axis")
    if (
        not torch.is_tensor(row_counts)
        or row_counts.ndim != 1
        or row_counts.dtype != torch.int64
        or row_counts.device.type != "cpu"
        or row_counts.requires_grad
        or row_counts.shape[0] != sums.shape[0]
        or sums.shape[0] <= 1
        or bool((row_counts <= 0).any())
        or not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions <= 0
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or not math.isfinite(confidence)
        or not 0.0 < confidence < 1.0
        or not isinstance(draw_chunk_size, int)
        or isinstance(draw_chunk_size, bool)
        or draw_chunk_size <= 0
    ):
        raise ValueError("bootstrap inputs are malformed")
    documents = int(sums.shape[0])
    point_pooled = sums.sum(dim=0) / int(row_counts.sum())
    point = statistic(point_pooled)
    if (
        not torch.is_tensor(point)
        or point.ndim != 1
        or point.dtype != torch.float64
        or point.device.type != "cpu"
        or point.requires_grad
        or point.numel() == 0
        or not bool(torch.isfinite(point).all())
    ):
        raise ValueError("bootstrap statistic must return a finite CPU float64 vector")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples: list[torch.Tensor] = []
    for start in range(0, repetitions, draw_chunk_size):
        chunk = min(draw_chunk_size, repetitions - start)
        indices = torch.randint(
            documents, (chunk, documents), generator=generator, dtype=torch.int64,
        )
        sampled_sums = sums[indices].sum(dim=1)
        sampled_counts = row_counts[indices].sum(dim=1).to(torch.float64)
        expand = (chunk,) + (1,) * (sums.ndim - 1)
        pooled = sampled_sums / sampled_counts.reshape(expand)
        values = statistic(pooled)
        if (
            not torch.is_tensor(values)
            or tuple(values.shape) != (chunk, point.numel())
            or values.dtype != torch.float64
            or values.device.type != "cpu"
            or values.requires_grad
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("bootstrap statistic returned malformed draws")
        samples.append(values.clone())
    draws = torch.cat(samples, dim=0)
    maximum_errors = torch.abs(draws - point).amax(dim=1)
    rank = math.ceil(confidence * repetitions)
    critical = float(torch.kthvalue(maximum_errors, rank).values)
    return SimultaneousBootstrapBand(
        point=point.clone(),
        draws=draws,
        critical_value=critical,
        lower=point - critical,
        upper=point + critical,
        repetitions=repetitions,
        seed=seed,
        confidence=confidence,
        critical_order_statistic_one_indexed=rank,
        documents_per_draw=documents,
    )


__all__ = [
    "BilinearFactors",
    "DocumentProbeGramAccumulator",
    "DocumentProbeGramStatistics",
    "MEAN_RATIO_THRESHOLD",
    "PROJECTOR_RELATIVE_FROBENIUS_TOLERANCE",
    "PopulationMoments",
    "PopulationProjectors",
    "ProbeGramAccumulator",
    "ProjectorFamily",
    "SimultaneousBootstrapBand",
    "StreamingPopulationMoments",
    "average_ranks",
    "bilinear_output",
    "build_population_projectors",
    "combine_bilinear_factors",
    "deterministic_haar_basis",
    "evaluate_probe_bank",
    "noncentral_gaussian_cross_inner_product",
    "noncentral_gaussian_probe_gram",
    "simultaneous_document_bootstrap",
    "spearman_average_rank",
    "teacher_minus_candidate_factors",
]
