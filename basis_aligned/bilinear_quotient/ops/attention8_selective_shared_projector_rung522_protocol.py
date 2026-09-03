"""CPU-only protocol and statistics helpers for the rung-522 red-team gates.

No function loads a model or data artifact, and CUDA tensors are rejected.  The
module turns the corrected prose protocol into deterministic, directly testable
operations over already-saved labels, frames, and response sufficient statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import operator
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


PERMUTATION_NAMESPACE = "a8-r522-four-bit-label-null-v1"
BOOTSTRAP_NAMESPACE = "a8-r522-row-bootstrap-v1"
REGISTERED_BOOTSTRAPS = 2_000
REGISTERED_REAL_SEEDS = tuple(range(52_200, 52_205))
REGISTERED_NULL_SEEDS = tuple(range(52_300, 52_316))


def _integers(values: Sequence[int] | torch.Tensor, name: str) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.device.type != "cpu":
            raise ValueError(f"{name} must be a one-dimensional CPU tensor")
        if values.is_floating_point() or values.is_complex():
            raise ValueError(f"{name} must have an integer dtype")
        raw = values.tolist()
    else:
        raw = values
    try:
        return tuple(operator.index(value) for value in raw)
    except TypeError as error:
        raise ValueError(f"{name} must contain integers") from error


def _finite(values: Sequence[float] | torch.Tensor, name: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.device.type != "cpu":
            raise ValueError(f"{name} must be a one-dimensional CPU tensor")
        result = values.detach().to(torch.float64)
    else:
        result = torch.tensor(tuple(values), dtype=torch.float64)
    if result.numel() == 0 or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be nonempty and finite")
    return result


def higher_quantile(values: Sequence[float] | torch.Tensor, probability: float) -> float:
    """NumPy-compatible ``method='higher'`` quantile without importing NumPy."""
    samples = _finite(values, "quantile values").sort().values
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0, 1]")
    index = math.ceil(probability * (samples.numel() - 1))
    return float(samples[index])


@dataclass(frozen=True)
class MembershipPermutationResult:
    permuted_codes: tuple[int, ...]
    donor_position_indices: tuple[int, ...]
    original_nonzero_count: int
    moved_nonzero_count: int
    moved_nonzero_fraction: float
    maximum_possible_moved_nonzero_count: int
    maximum_possible_moved_nonzero_fraction: float
    attains_maximum_possible_movement: bool
    sha256: str


class InsufficientPermutationMovement(RuntimeError):
    """The label-null assignment failed to attain the exact feasible maximum."""

    def __init__(self, result: MembershipPermutationResult) -> None:
        self.result = result
        super().__init__(
            "four-bit membership null did not attain maximum feasible movement: "
            f"got {result.moved_nonzero_count}, maximum "
            f"{result.maximum_possible_moved_nonzero_count}"
        )


def _permutation_digest(
    seed: int,
    group: tuple[int, int, int, int],
    recipient_id: int,
    donor_id: int,
) -> bytes:
    payload = ":".join(
        (
            PERMUTATION_NAMESPACE,
            str(seed),
            *(str(value) for value in group),
            str(recipient_id),
            str(donor_id),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()


def permute_four_bit_memberships(
    membership_codes: Sequence[int] | torch.Tensor,
    *,
    token_classes: Sequence[int] | torch.Tensor,
    token_positions: Sequence[int] | torch.Tensor,
    ce_deciles: Sequence[int] | torch.Tensor,
    parent_slice_codes: Sequence[int] | torch.Tensor,
    seed: int,
    position_ids: Sequence[int] | torch.Tensor | None = None,
) -> MembershipPermutationResult:
    """Maximally move complete four-bit codes inside every frozen stratum.

    The primary linear-assignment cost is one exactly when a nonzero recipient
    retains its original code.  A seed-keyed SHA-256 secondary cost chooses
    among assignments without changing that primary optimum.  The four bits
    are never permuted independently. Code legality ``membership subset-of
    parent`` is checked before and after permutation.
    """
    codes = _integers(membership_codes, "membership_codes")
    classes = _integers(token_classes, "token_classes")
    positions = _integers(token_positions, "token_positions")
    deciles = _integers(ce_deciles, "ce_deciles")
    parents = _integers(parent_slice_codes, "parent_slice_codes")
    size = len(codes)
    if not size or any(len(values) != size for values in (classes, positions, deciles, parents)):
        raise ValueError("all permutation metadata must have the same positive length")
    ids = tuple(range(size)) if position_ids is None else _integers(position_ids, "position_ids")
    if len(ids) != size or len(set(ids)) != size:
        raise ValueError("position_ids must be unique and aligned one-to-one with positions")
    if any(code < 0 or code > 15 for code in codes):
        raise ValueError("membership codes must lie in [0, 15]")
    if any(parent < 0 or parent > 15 for parent in parents):
        raise ValueError("parent-slice codes must lie in [0, 15]")
    if any(position < 0 for position in positions):
        raise ValueError("token positions must be nonnegative")
    if any(code & ~parent for code, parent in zip(codes, parents, strict=True)):
        raise ValueError("a membership bit is set outside its circuit parent slice")
    seed = operator.index(seed)

    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for index in range(size):
        key = (classes[index], positions[index] // 32, deciles[index], parents[index])
        groups.setdefault(key, []).append(index)

    permuted = [-1] * size
    donor_indices = [-1] * size
    theoretical_minimum_unchanged = 0
    for group in sorted(groups):
        recipients = sorted(groups[group], key=lambda index: (ids[index], index))
        donors = recipients
        count = len(recipients)
        if not any(codes[index] != 0 for index in recipients):
            for recipient in recipients:
                permuted[recipient] = codes[recipient]
                donor_indices[recipient] = recipient
            continue
        primary = np.zeros((count, count), dtype=np.int64)
        secondary = np.zeros((count, count), dtype=np.int64)
        for row, recipient in enumerate(recipients):
            for column, donor in enumerate(donors):
                primary[row, column] = int(
                    codes[recipient] != 0 and codes[recipient] == codes[donor]
                )
                secondary[row, column] = int.from_bytes(
                    _permutation_digest(
                        seed, group, ids[recipient], ids[donor]
                    )[:2],
                    "little",
                    signed=False,
                )
        primary_rows, primary_columns = linear_sum_assignment(primary)
        theoretical_minimum_unchanged += int(
            primary[primary_rows, primary_columns].sum()
        )
        # One extra primary penalty is more expensive than the maximum total
        # secondary cost of any assignment in this stratum.
        primary_weight = count * 65535 + 1
        assigned_rows, assigned_columns = linear_sum_assignment(
            primary * primary_weight + secondary
        )
        if not np.array_equal(assigned_rows, np.arange(count)):
            raise RuntimeError("linear assignment did not cover every recipient")
        for row, column in zip(assigned_rows.tolist(), assigned_columns.tolist(), strict=True):
            recipient = recipients[row]
            donor = donors[column]
            permuted[recipient] = codes[donor]
            donor_indices[recipient] = donor

    if any(code < 0 for code in permuted) or any(index < 0 for index in donor_indices):
        raise RuntimeError("internal error: membership permutation left an unassigned position")
    if any(code & ~parent for code, parent in zip(permuted, parents, strict=True)):
        raise RuntimeError("internal error: membership permutation left a parent population")
    nonzero = [index for index, code in enumerate(codes) if code != 0]
    if not nonzero:
        raise ValueError("movement gate is undefined when every FIT membership code is zero")
    moved = sum(permuted[index] != codes[index] for index in nonzero)
    fraction = moved / len(nonzero)
    maximum_moved = len(nonzero) - theoretical_minimum_unchanged
    maximum_fraction = maximum_moved / len(nonzero)
    encoded = json.dumps(
        {
            "namespace": PERMUTATION_NAMESPACE,
            "seed": seed,
            "permuted_codes": permuted,
            "donor_position_indices": donor_indices,
            "original_nonzero_count": len(nonzero),
            "maximum_possible_moved_nonzero_count": maximum_moved,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    result = MembershipPermutationResult(
        permuted_codes=tuple(permuted),
        donor_position_indices=tuple(donor_indices),
        original_nonzero_count=len(nonzero),
        moved_nonzero_count=moved,
        moved_nonzero_fraction=fraction,
        maximum_possible_moved_nonzero_count=maximum_moved,
        maximum_possible_moved_nonzero_fraction=maximum_fraction,
        attains_maximum_possible_movement=moved == maximum_moved,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )
    if not result.attains_maximum_possible_movement:
        raise InsufficientPermutationMovement(result)
    return result


def _cpu_frame(frame: torch.Tensor, name: str) -> None:
    if (
        not isinstance(frame, torch.Tensor)
        or frame.ndim != 2
        or frame.device.type != "cpu"
        or frame.dtype not in (torch.float32, torch.float64)
        or frame.shape[0] < frame.shape[1]
        or frame.shape[1] == 0
        or not bool(torch.isfinite(frame).all())
    ):
        raise ValueError(f"{name} must be a finite rank-2 float CPU frame")
    identity = torch.eye(frame.shape[1], dtype=frame.dtype)
    if float((frame.mT @ frame - identity).abs().amax()) > 1e-5:
        raise ValueError(f"{name} must be orthonormal")


def normalized_projector_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    """Gauge-invariant overlap ``||Q_left^T Q_right||_F^2 / rank`` on CPU."""
    _cpu_frame(left, "left")
    _cpu_frame(right, "right")
    if left.shape != right.shape:
        raise ValueError("matched projector frames must have identical shape")
    return float((left.mT.double() @ right.double()).square().sum() / left.shape[1])


@dataclass(frozen=True)
class SeedStability:
    seed: int
    pairwise_overlaps: tuple[float, float, float]
    minimum_overlap: float


@dataclass(frozen=True)
class StabilitySummary:
    folds: tuple[str, str, str]
    real: tuple[SeedStability, ...]
    null: tuple[SeedStability, ...]
    null_q95_higher: float
    real_strict_exceed_count: int
    passes_four_of_five: bool


def _seed_stabilities(
    frames: Mapping[int, Mapping[str, torch.Tensor]],
    folds: tuple[str, str, str],
    family: str,
    expected_shape: tuple[int, int],
) -> tuple[SeedStability, ...]:
    result = []
    for raw_seed in sorted(frames):
        seed = operator.index(raw_seed)
        by_fold = frames[raw_seed]
        if set(by_fold) != set(folds):
            raise ValueError(f"{family} seed {seed} must contain exactly the three matched folds")
        if any(tuple(by_fold[fold].shape) != expected_shape for fold in folds):
            raise ValueError("every real and null frame must have the same registered shape")
        overlaps = (
            normalized_projector_overlap(by_fold[folds[0]], by_fold[folds[1]]),
            normalized_projector_overlap(by_fold[folds[0]], by_fold[folds[2]]),
            normalized_projector_overlap(by_fold[folds[1]], by_fold[folds[2]]),
        )
        result.append(SeedStability(seed, overlaps, min(overlaps)))
    return tuple(result)


def summarize_matched_three_fold_stability(
    real_frames: Mapping[int, Mapping[str, torch.Tensor]],
    null_frames: Mapping[int, Mapping[str, torch.Tensor]],
) -> StabilitySummary:
    """Compare five real and sixteen null minima of the same three-fold statistic."""
    if tuple(sorted(real_frames)) != REGISTERED_REAL_SEEDS or tuple(
        sorted(null_frames)
    ) != REGISTERED_NULL_SEEDS:
        raise ValueError(
            "registered stability comparison requires real seeds 52200..52204 "
            "and null seeds 52300..52315"
        )
    first_seed = next(iter(real_frames))
    folds = tuple(sorted(real_frames[first_seed]))
    if len(folds) != 3:
        raise ValueError("each stability seed requires exactly three leave-one-out folds")
    typed_folds = (folds[0], folds[1], folds[2])
    first_frame = real_frames[first_seed][typed_folds[0]]
    _cpu_frame(first_frame, "first real frame")
    expected_shape = (first_frame.shape[0], first_frame.shape[1])
    real = _seed_stabilities(real_frames, typed_folds, "real", expected_shape)
    null = _seed_stabilities(null_frames, typed_folds, "null", expected_shape)
    q95 = higher_quantile([value.minimum_overlap for value in null], 0.95)
    exceed = sum(value.minimum_overlap > q95 for value in real)
    return StabilitySummary(
        folds=typed_folds,
        real=real,
        null=null,
        null_q95_higher=q95,
        real_strict_exceed_count=exceed,
        passes_four_of_five=exceed >= 4,
    )


@dataclass(frozen=True)
class SelectivitySummary:
    member_rms: float
    control_rms: float
    concentration: float
    bounded_selectivity: float
    fourfold_margin: float


@dataclass(frozen=True)
class BoundedJointSummary:
    minimum_heldout_selectivity: float
    minimum_heldout_aligned_recovery: float
    product: float


def selectivity_from_rms(
    member_rms: float, control_rms: float, *, epsilon: float = 1e-12
) -> SelectivitySummary:
    """Compute both retained and bounded selectivity statistics from RMS values."""
    member_rms, control_rms = float(member_rms), float(control_rms)
    if (
        not math.isfinite(member_rms)
        or not math.isfinite(control_rms)
        or member_rms < 0
        or control_rms < 0
    ):
        raise ValueError("member and control RMS must be finite and nonnegative")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    concentration = (
        member_rms / control_rms
        if control_rms > 0
        else (math.inf if member_rms > 0 else 0.0)
    )
    return SelectivitySummary(
        member_rms=member_rms,
        control_rms=control_rms,
        concentration=concentration,
        bounded_selectivity=(member_rms - control_rms) / (member_rms + control_rms + epsilon),
        fourfold_margin=member_rms - 4 * control_rms,
    )


def selectivity_from_effects(
    member_effects: Sequence[float] | torch.Tensor,
    control_effects: Sequence[float] | torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> SelectivitySummary:
    """Compute RMS and the corrected selectivity statistics from saved effects."""
    member = _finite(member_effects, "member_effects")
    control = _finite(control_effects, "control_effects")
    return selectivity_from_rms(
        float(member.square().mean().sqrt()),
        float(control.square().mean().sqrt()),
        epsilon=epsilon,
    )


def bounded_joint_statistic(
    heldout_selectivities: Sequence[float] | torch.Tensor,
    heldout_aligned_recoveries: Sequence[float] | torch.Tensor,
) -> BoundedJointSummary:
    """Registered bounded real-vs-control statistic over required held-out cells."""
    selectivities = _finite(heldout_selectivities, "heldout_selectivities")
    recoveries = _finite(heldout_aligned_recoveries, "heldout_aligned_recoveries")
    if selectivities.shape != recoveries.shape:
        raise ValueError("held-out selectivity and recovery cells must align")
    if bool((selectivities < -1).any()) or bool((selectivities > 1).any()):
        raise ValueError("bounded selectivities must lie in [-1, 1]")
    minimum_selectivity = float(selectivities.min())
    minimum_recovery = float(recoveries.min())
    return BoundedJointSummary(
        minimum_heldout_selectivity=minimum_selectivity,
        minimum_heldout_aligned_recovery=minimum_recovery,
        product=minimum_selectivity * minimum_recovery,
    )


@dataclass(frozen=True)
class RowPairSquares:
    """Per-paired-row sums of squared effects and contributing token counts."""

    pair_ids: tuple[int, ...]
    member_sum_squares: tuple[float, ...]
    member_counts: tuple[int, ...]
    control_sum_squares: tuple[float, ...]
    control_counts: tuple[int, ...]

    @classmethod
    def from_sequences(
        cls,
        member_sum_squares: Sequence[float] | torch.Tensor,
        member_counts: Sequence[int] | torch.Tensor,
        control_sum_squares: Sequence[float] | torch.Tensor,
        control_counts: Sequence[int] | torch.Tensor,
        *,
        pair_ids: Sequence[int] | torch.Tensor,
    ) -> "RowPairSquares":
        member_ss = _finite(member_sum_squares, "member_sum_squares")
        control_ss = _finite(control_sum_squares, "control_sum_squares")
        member_n = _integers(member_counts, "member_counts")
        control_n = _integers(control_counts, "control_counts")
        ids = _integers(pair_ids, "pair_ids")
        length = member_ss.numel()
        if (
            control_ss.numel() != length
            or len(member_n) != length
            or len(control_n) != length
            or len(ids) != length
        ):
            raise ValueError("paired row sufficient statistics must have equal lengths")
        if len(set(ids)) != len(ids):
            raise ValueError("pair_ids must uniquely identify the matched row pairs")
        if bool((member_ss < 0).any()) or bool((control_ss < 0).any()):
            raise ValueError("row sums of squares must be nonnegative")
        if any(count <= 0 for count in member_n + control_n):
            raise ValueError("every resampled row must contain a positive token count")
        return cls(
            ids,
            tuple(float(value) for value in member_ss),
            member_n,
            tuple(float(value) for value in control_ss),
            control_n,
        )

    def __len__(self) -> int:
        return len(self.member_sum_squares)


@dataclass(frozen=True)
class RowBootstrapSummary:
    point: SelectivitySummary
    draws: int
    cell_id: str
    fourfold_margin_lower95_higher: float
    bounded_selectivity_lower95_higher: float
    bounded_selectivity_improvement_lower95_higher: float | None
    fourfold_margin_samples: tuple[float, ...]
    bounded_selectivity_samples: tuple[float, ...]
    bounded_selectivity_improvement_samples: tuple[float, ...] | None
    sha256: str


def _selectivity_samples(stats: RowPairSquares, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    member_ss = torch.tensor(stats.member_sum_squares, dtype=torch.float64)
    member_n = torch.tensor(stats.member_counts, dtype=torch.float64)
    control_ss = torch.tensor(stats.control_sum_squares, dtype=torch.float64)
    control_n = torch.tensor(stats.control_counts, dtype=torch.float64)
    member_rms = (member_ss[indices].sum(1) / member_n[indices].sum(1)).sqrt()
    control_rms = (control_ss[indices].sum(1) / control_n[indices].sum(1)).sqrt()
    bounded = (member_rms - control_rms) / (member_rms + control_rms + 1e-12)
    margin = member_rms - 4 * control_rms
    return bounded, margin


def deterministic_row_bootstrap_indices(
    cluster_count: int,
    *,
    cell_id: str,
    draws: int = REGISTERED_BOOTSTRAPS,
) -> torch.Tensor:
    """Return the final-addendum SHA-counter row-cluster resamples.

    Entry ``[b,k]`` is the unsigned little-endian value of the first eight bytes
    of SHA-256(``a8-r522-row-bootstrap-v1:<cell ID>:<b>:<k>``), reduced modulo
    ``cluster_count``.  No library random-number generator or implicit seed is
    involved.
    """
    cluster_count = operator.index(cluster_count)
    draws = operator.index(draws)
    if cluster_count <= 0:
        raise ValueError("cluster_count must be positive")
    if draws <= 0:
        raise ValueError("draws must be positive")
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a nonempty full cell identifier")
    sampled = torch.empty((draws, cluster_count), dtype=torch.int64)
    prefix = f"{BOOTSTRAP_NAMESPACE}:{cell_id}:"
    for replicate in range(draws):
        for draw in range(cluster_count):
            digest = hashlib.sha256(f"{prefix}{replicate}:{draw}".encode("utf-8")).digest()
            sampled[replicate, draw] = int.from_bytes(
                digest[:8], byteorder="little", signed=False
            ) % cluster_count
    return sampled


def deterministic_row_bootstrap(
    stats: RowPairSquares,
    *,
    cell_id: str,
    draws: int = REGISTERED_BOOTSTRAPS,
    comparison: RowPairSquares | None = None,
) -> RowBootstrapSummary:
    """Paired row bootstrap for margin and bounded-selectivity improvement.

    RMS is reconstructed from per-row sums of squares and token counts after
    resampling row-pair indices.  Thus the resampling unit is the matched row
    pair while the estimand remains RMS over its contributing tokens.
    """
    if not isinstance(stats, RowPairSquares) or len(stats) == 0:
        raise ValueError("stats must contain at least one paired row")
    if comparison is not None and (
        not isinstance(comparison, RowPairSquares)
        or len(comparison) != len(stats)
        or comparison.pair_ids != stats.pair_ids
    ):
        raise ValueError("comparison must contain the same ordered paired-row identities")
    indices = deterministic_row_bootstrap_indices(
        len(stats), cell_id=cell_id, draws=draws
    )
    draws = indices.shape[0]
    bounded, margin = _selectivity_samples(stats, indices)
    improvement = None
    improvement_lower = None
    if comparison is not None:
        comparison_bounded, _ = _selectivity_samples(comparison, indices)
        improvement = bounded - comparison_bounded
        improvement_lower = higher_quantile(improvement, 0.05)

    member_rms = math.sqrt(sum(stats.member_sum_squares) / sum(stats.member_counts))
    control_rms = math.sqrt(sum(stats.control_sum_squares) / sum(stats.control_counts))
    payload = {
        "namespace": BOOTSTRAP_NAMESPACE,
        "cell_id": cell_id,
        "draws": draws,
        "fourfold_margin": margin.tolist(),
        "bounded_selectivity": bounded.tolist(),
        "bounded_selectivity_improvement": None if improvement is None else improvement.tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return RowBootstrapSummary(
        point=selectivity_from_rms(member_rms, control_rms),
        draws=draws,
        cell_id=cell_id,
        fourfold_margin_lower95_higher=higher_quantile(margin, 0.05),
        bounded_selectivity_lower95_higher=higher_quantile(bounded, 0.05),
        bounded_selectivity_improvement_lower95_higher=improvement_lower,
        fourfold_margin_samples=tuple(float(value) for value in margin),
        bounded_selectivity_samples=tuple(float(value) for value in bounded),
        bounded_selectivity_improvement_samples=(
            None if improvement is None else tuple(float(value) for value in improvement)
        ),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


@dataclass(frozen=True)
class SignFlipSummary:
    paired_improvements: tuple[float, float, float, float, float]
    observed_mean: float
    null_means: tuple[float, ...]
    null_q95_higher: float
    strictly_exceeds_q95: bool


def exact_five_pair_sign_flip_null(
    paired_improvements: Sequence[float] | torch.Tensor,
) -> SignFlipSummary:
    """Enumerate the exact 2^5 one-sided null for paired seed improvements."""
    values = _finite(paired_improvements, "paired_improvements")
    if values.numel() != 5:
        raise ValueError("the registered paired sign-flip test requires exactly five values")
    pairs = tuple(float(value) for value in values)
    null = tuple(
        sum((1 if mask & (1 << index) else -1) * value for index, value in enumerate(pairs))
        / 5
        for mask in range(32)
    )
    observed = sum(pairs) / 5
    q95 = higher_quantile(null, 0.95)
    return SignFlipSummary(
        paired_improvements=pairs,  # type: ignore[arg-type]
        observed_mean=observed,
        null_means=null,
        null_q95_higher=q95,
        strictly_exceeds_q95=observed > q95,
    )


__all__ = [
    "BOOTSTRAP_NAMESPACE",
    "BoundedJointSummary",
    "InsufficientPermutationMovement",
    "MembershipPermutationResult",
    "PERMUTATION_NAMESPACE",
    "REGISTERED_BOOTSTRAPS",
    "REGISTERED_NULL_SEEDS",
    "REGISTERED_REAL_SEEDS",
    "RowBootstrapSummary",
    "RowPairSquares",
    "SeedStability",
    "SelectivitySummary",
    "SignFlipSummary",
    "StabilitySummary",
    "deterministic_row_bootstrap",
    "deterministic_row_bootstrap_indices",
    "bounded_joint_statistic",
    "exact_five_pair_sign_flip_null",
    "higher_quantile",
    "normalized_projector_overlap",
    "permute_four_bit_memberships",
    "selectivity_from_effects",
    "selectivity_from_rms",
    "summarize_matched_three_fold_stability",
]
