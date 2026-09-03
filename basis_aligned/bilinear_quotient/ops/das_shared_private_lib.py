"""CPU-safe mathematical utilities for the rung-521 shared/private DAS test.

This module deliberately has no model, data-file, or CUDA imports.  It contains
only the exact algebra and deterministic bookkeeping shared by the preflight and
the eventual runner.  Leading dimensions of activation tensors are arbitrary;
the final dimension is always the 1,152-dimensional attention write space.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import operator
from typing import Collection, Iterable, Mapping, Sequence

import torch


DEFAULT_FOLD_NAMESPACE = "a8-shared-private-v1"
DEFAULT_MATCH_SEED = 52100
DEFAULT_MATCH_STAGES: tuple[tuple[str, ...], ...] = (
    ("token_id", "position_bin", "ce_decile"),
    ("token_id", "ce_decile"),
    ("token_class", "position_bin", "ce_decile"),
    ("token_class", "ce_decile"),
)


def _require_matrix(frame: torch.Tensor, name: str = "frame") -> None:
    if not isinstance(frame, torch.Tensor) or frame.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 torch tensor")
    if not frame.is_floating_point():
        raise ValueError(f"{name} must have a floating dtype")
    if frame.shape[0] < frame.shape[1] or frame.shape[1] == 0:
        raise ValueError(f"{name} must have shape (dimension, rank), dimension >= rank > 0")


def orthonormality_error(frame: torch.Tensor) -> torch.Tensor:
    """Return ``max|Q^T Q-I|`` without choosing a basis inside the subspace."""
    _require_matrix(frame)
    identity = torch.eye(frame.shape[1], dtype=frame.dtype, device=frame.device)
    return (frame.mT @ frame - identity).abs().amax()


def validate_orthonormal_frame(
    frame: torch.Tensor, *, atol: float = 1e-5, name: str = "frame"
) -> None:
    """Fail closed unless ``frame`` is finite and orthonormal to ``atol``."""
    _require_matrix(frame, name)
    if not bool(torch.isfinite(frame).all().detach().cpu()):
        raise ValueError(f"{name} contains a non-finite value")
    error = float(orthonormality_error(frame).detach().cpu())
    if error > atol:
        raise ValueError(f"{name} is not orthonormal: max error {error:.6g} > {atol:.6g}")


def symmetric_polar_retraction(
    matrix: torch.Tensor, *, relative_eigenvalue_floor: float | None = None
) -> torch.Tensor:
    """Retract a full-column-rank matrix to its symmetric polar factor.

    The returned frame is ``A (A^T A)^(-1/2)``.  This preserves the column
    space, is differentiable away from rank loss, and does not privilege any
    column basis.  Rank-deficient inputs are rejected rather than silently
    changing the registered rank through eigenvalue clipping.
    """
    _require_matrix(matrix, "matrix")
    if matrix.dtype not in (torch.float32, torch.float64):
        raise ValueError("symmetric polar retraction requires float32 or float64")
    if not bool(torch.isfinite(matrix).all().detach().cpu()):
        raise ValueError("matrix contains a non-finite value")

    gram = matrix.mT @ matrix
    gram = (gram + gram.mT) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    if relative_eigenvalue_floor is None:
        relative_eigenvalue_floor = 32 * torch.finfo(matrix.dtype).eps
    if not math.isfinite(relative_eigenvalue_floor) or relative_eigenvalue_floor <= 0:
        raise ValueError("relative_eigenvalue_floor must be finite and positive")
    largest = float(eigenvalues[-1].detach().cpu())
    smallest = float(eigenvalues[0].detach().cpu())
    if largest <= 0 or smallest <= largest * relative_eigenvalue_floor:
        raise ValueError(
            "matrix is rank deficient at the requested numerical precision: "
            f"smallest/largest Gram eigenvalue={smallest / max(largest, 1e-300):.6g}"
        )
    inverse_square_root = (
        eigenvectors * eigenvalues.rsqrt().unsqueeze(0)
    ) @ eigenvectors.mT
    return matrix @ inverse_square_root


def retract_frame_blocks(*blocks: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Apply one global symmetric-polar retraction and split the columns back.

    This is appropriate when every supplied block is allowed to move.  It must
    not be used to update private frames while a previously fitted shared frame
    is meant to remain frozen; use :func:`retract_against_fixed` for that case.
    """
    if not blocks:
        raise ValueError("at least one frame block is required")
    dimension = blocks[0].shape[0] if blocks[0].ndim == 2 else None
    for index, block in enumerate(blocks):
        _require_matrix(block, f"blocks[{index}]")
        if block.shape[0] != dimension:
            raise ValueError("all frame blocks must have the same ambient dimension")
    widths = [block.shape[1] for block in blocks]
    joint = symmetric_polar_retraction(torch.cat(blocks, dim=1))
    return tuple(joint[:, start : start + width] for start, width in _offsets(widths))


def _offsets(widths: Sequence[int]) -> Iterable[tuple[int, int]]:
    start = 0
    for width in widths:
        yield start, width
        start += width


def retract_against_fixed(
    candidate: torch.Tensor,
    fixed_frames: Sequence[torch.Tensor],
    *, fixed_atol: float = 1e-5,
) -> torch.Tensor:
    """Retract ``candidate`` inside the complement of frozen orthonormal frames."""
    _require_matrix(candidate, "candidate")
    if not fixed_frames:
        return symmetric_polar_retraction(candidate)
    for index, frame in enumerate(fixed_frames):
        validate_orthonormal_frame(frame, atol=fixed_atol, name=f"fixed_frames[{index}]")
        if frame.shape[0] != candidate.shape[0]:
            raise ValueError("candidate and fixed frames have different ambient dimensions")
    fixed = torch.cat(tuple(fixed_frames), dim=1)
    validate_orthonormal_frame(fixed, atol=fixed_atol, name="concatenated fixed frames")
    residual = candidate - fixed @ (fixed.mT @ candidate)
    return symmetric_polar_retraction(residual)


def projector_matrix(frame: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
    """Materialize ``QQ^T``; activation edits below avoid this dense matrix."""
    if validate:
        validate_orthonormal_frame(frame)
    else:
        _require_matrix(frame)
    return frame @ frame.mT


def projection_interchange(
    recipient: torch.Tensor,
    donor: torch.Tensor,
    frame: torch.Tensor,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Return ``y + ((d-y)Q)Q^T`` for a post-projection attention write."""
    if recipient.shape != donor.shape:
        raise ValueError("recipient and donor writes must have identical shapes")
    if recipient.ndim == 0 or recipient.shape[-1] != frame.shape[0]:
        raise ValueError("write dimension does not match the frame ambient dimension")
    if validate:
        validate_orthonormal_frame(frame)
    else:
        _require_matrix(frame)
    displacement = donor - recipient
    return recipient + (displacement @ frame) @ frame.mT


def fit_projected_mean(fit_writes: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """FIT-only mean of projected coordinates, averaged over all leading axes."""
    if fit_writes.ndim == 0 or fit_writes.shape[-1] != frame.shape[0]:
        raise ValueError("FIT write dimension does not match the frame")
    validate_orthonormal_frame(frame)
    coordinates = fit_writes @ frame
    if coordinates.ndim == 1:
        return coordinates
    return coordinates.reshape(-1, coordinates.shape[-1]).mean(dim=0)


def mean_centered_projection_removal(
    write: torch.Tensor,
    frame: torch.Tensor,
    fit_mean_coordinates: torch.Tensor,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Return ``y - (yQ-mu_Q)Q^T`` using a FIT-only projected mean."""
    if write.ndim == 0 or write.shape[-1] != frame.shape[0]:
        raise ValueError("write dimension does not match the frame")
    if fit_mean_coordinates.shape != (frame.shape[1],):
        raise ValueError("fit_mean_coordinates must contain one mean per frame column")
    if validate:
        validate_orthonormal_frame(frame)
    else:
        _require_matrix(frame)
    return write - ((write @ frame) - fit_mean_coordinates) @ frame.mT


def principal_cosines(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Cosines of principal angles, invariant to column rotations of either frame."""
    validate_orthonormal_frame(left, name="left")
    validate_orthonormal_frame(right, name="right")
    if left.shape[0] != right.shape[0]:
        raise ValueError("frames have different ambient dimensions")
    return torch.linalg.svdvals(left.mT @ right).clamp(0, 1)


def principal_angles(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Principal angles in radians, sorted from smallest to largest."""
    return torch.acos(principal_cosines(left, right))


def projector_overlap(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return ``tr(P_left P_right) = ||Q_left^T Q_right||_F^2``."""
    cosines = principal_cosines(left, right)
    return cosines.square().sum()


def normalized_projector_overlap(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Overlap divided by the smaller rank (rank 4 in rung 521)."""
    return projector_overlap(left, right) / min(left.shape[1], right.shape[1])


def projector_frobenius_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return ``||P_left-P_right||_F`` without materializing either projector."""
    overlap = projector_overlap(left, right)
    squared = left.shape[1] + right.shape[1] - 2 * overlap
    return squared.clamp_min(0).sqrt()


def _power_of_two_size(values: torch.Tensor) -> int:
    if not isinstance(values, torch.Tensor) or values.ndim == 0:
        raise ValueError("subset values must be a tensor with a leading subset axis")
    count = values.shape[0]
    if count < 1 or count & (count - 1):
        raise ValueError("leading subset-axis length must be a positive power of two")
    return count.bit_length() - 1


def mobius_from_subset_values(values: torch.Tensor) -> torch.Tensor:
    """Exact Boolean-lattice Möbius transform along the leading bitmask axis."""
    bits = _power_of_two_size(values)
    transformed = values
    tail = values.shape[1:]
    for bit in range(bits):
        step = 1 << bit
        grouped = transformed.reshape(-1, 2, step, *tail)
        lower = grouped[:, :1]
        upper = grouped[:, 1:2]
        transformed = torch.cat((lower, upper - lower), dim=1).reshape(values.shape)
    return transformed


def subset_values_from_mobius(coefficients: torch.Tensor) -> torch.Tensor:
    """Inverse transform: reconstruct every subset endpoint from interactions."""
    bits = _power_of_two_size(coefficients)
    values = coefficients
    tail = coefficients.shape[1:]
    for bit in range(bits):
        step = 1 << bit
        grouped = values.reshape(-1, 2, step, *tail)
        lower = grouped[:, :1]
        upper = grouped[:, 1:2]
        values = torch.cat((lower, upper + lower), dim=1).reshape(coefficients.shape)
    return values


def document_fold(
    document_id: int,
    *,
    namespace: str = DEFAULT_FOLD_NAMESPACE,
    modulo: int = 10,
) -> int:
    """Frozen little-endian SHA-256 document fold from the rung-521 prereg."""
    document_id = operator.index(document_id)
    if document_id < 0:
        raise ValueError("document_id must be nonnegative")
    if not namespace:
        raise ValueError("namespace must be nonempty")
    modulo = operator.index(modulo)
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    digest = hashlib.sha256(f"{namespace}:{document_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % modulo


def document_folds(
    document_ids: Iterable[int] | torch.Tensor,
    *,
    namespace: str = DEFAULT_FOLD_NAMESPACE,
    modulo: int = 10,
) -> torch.Tensor:
    """Vectorized CPU wrapper around :func:`document_fold`, preserving shape."""
    if isinstance(document_ids, torch.Tensor):
        if document_ids.is_floating_point() or document_ids.is_complex():
            raise ValueError("document IDs must have an integer dtype")
        shape = document_ids.shape
        flat = document_ids.detach().cpu().reshape(-1).tolist()
    else:
        flat = list(document_ids)
        shape = (len(flat),)
    result = [document_fold(value, namespace=namespace, modulo=modulo) for value in flat]
    return torch.tensor(result, dtype=torch.int64).reshape(shape)


def registered_split_mask(folds: torch.Tensor, split: str) -> torch.Tensor:
    """Return the frozen FIT (0--5), VALIDATION (6--7), or TEST (8--9) mask."""
    if split == "FIT":
        return (folds >= 0) & (folds <= 5)
    if split == "VALIDATION":
        return (folds >= 6) & (folds <= 7)
    if split == "TEST":
        return (folds >= 8) & (folds <= 9)
    if split == "FIT_0_2":
        return (folds >= 0) & (folds <= 2)
    if split == "FIT_3_5":
        return (folds >= 3) & (folds <= 5)
    raise ValueError(f"unknown registered split {split!r}")


def _validate_masks(
    masks: Mapping[str, torch.Tensor], universe: torch.Tensor | None
) -> tuple[tuple[str, ...], tuple[int, ...], torch.device]:
    if not masks:
        raise ValueError("at least one named mask is required")
    names = tuple(masks)
    first = masks[names[0]]
    if not isinstance(first, torch.Tensor) or first.dtype != torch.bool:
        raise ValueError("all masks must be boolean torch tensors")
    shape, device = tuple(first.shape), first.device
    for name, mask in masks.items():
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError(f"mask {name!r} is not boolean")
        if tuple(mask.shape) != shape or mask.device != device:
            raise ValueError("all masks must have identical shapes and devices")
    if universe is not None and (
        universe.dtype != torch.bool
        or tuple(universe.shape) != shape
        or universe.device != device
    ):
        raise ValueError("universe must be a matching boolean tensor")
    return names, shape, device


def mask_membership_codes(
    masks: Mapping[str, torch.Tensor], *, universe: torch.Tensor | None = None
) -> torch.Tensor:
    """Encode exact membership in up to 62 named masks as an integer bitmask."""
    names, shape, device = _validate_masks(masks, universe)
    if len(names) > 62:
        raise ValueError("membership encoding supports at most 62 masks")
    codes = torch.zeros(shape, dtype=torch.int64, device=device)
    for bit, name in enumerate(names):
        codes = codes | (masks[name].to(torch.int64) << bit)
    if universe is not None:
        codes = torch.where(universe, codes, torch.full_like(codes, -1))
    return codes


def exact_overlap_lattice(
    masks: Mapping[str, torch.Tensor], *, universe: torch.Tensor | None = None
) -> dict[int, torch.Tensor]:
    """Return all ``2^k`` exact overlap cells, including empty cells."""
    names, shape, device = _validate_masks(masks, universe)
    if universe is None:
        universe = torch.ones(shape, dtype=torch.bool, device=device)
    codes = mask_membership_codes(masks, universe=universe)
    return {code: universe & (codes == code) for code in range(1 << len(names))}


def exclusive_masks(
    masks: Mapping[str, torch.Tensor], *, universe: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """For each name, keep members absent from every other named mask."""
    names, _, _ = _validate_masks(masks, universe)
    cells = exact_overlap_lattice(masks, universe=universe)
    return {name: cells[1 << bit] for bit, name in enumerate(names)}


def overlap_lattice_counts(
    masks: Mapping[str, torch.Tensor], *, universe: torch.Tensor | None = None
) -> dict[int, int]:
    """Integer population of every exact overlap cell."""
    return {
        code: int(cell.sum().detach().cpu())
        for code, cell in exact_overlap_lattice(masks, universe=universe).items()
    }


def make_matching_strata(
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    ce_deciles: torch.Tensor,
    token_classes: torch.Tensor,
    *,
    position_bin_width: int = 32,
) -> dict[str, torch.Tensor]:
    """Build the four categorical arrays used by the frozen relaxation ladder."""
    position_bin_width = operator.index(position_bin_width)
    if position_bin_width <= 0:
        raise ValueError("position_bin_width must be positive")
    arrays = (token_ids, positions, ce_deciles, token_classes)
    if any(not isinstance(value, torch.Tensor) or value.ndim != 1 for value in arrays):
        raise ValueError("matching inputs must be one-dimensional torch tensors")
    if any(value.is_floating_point() or value.is_complex() for value in arrays):
        raise ValueError("matching inputs must have integer dtypes")
    if any(value.shape != token_ids.shape for value in arrays[1:]):
        raise ValueError("matching inputs must have equal lengths")
    if any(value.device.type != "cpu" for value in arrays):
        raise ValueError("deterministic matching metadata must reside on CPU")
    return {
        "token_id": token_ids.to(torch.int64),
        "position_bin": torch.div(
            positions.to(torch.int64), position_bin_width, rounding_mode="floor"
        ),
        "ce_decile": ce_deciles.to(torch.int64),
        "token_class": token_classes.to(torch.int64),
    }


class MatchingError(RuntimeError):
    """Raised when the registered relaxation ladder has no full matching."""


@dataclass(frozen=True)
class MatchResult:
    """One deterministic without-replacement recipient-to-match map."""

    recipient_indices: torch.Tensor
    matched_indices: torch.Tensor
    relaxation_levels: torch.Tensor
    relaxation_counts: tuple[int, ...]
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "recipient_indices": self.recipient_indices.tolist(),
            "matched_indices": self.matched_indices.tolist(),
            "relaxation_levels": self.relaxation_levels.tolist(),
            "relaxation_counts": list(self.relaxation_counts),
            "sha256": self.sha256,
        }


def _indices(values: Sequence[int] | torch.Tensor, name: str) -> list[int]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.is_floating_point() or values.is_complex():
            raise ValueError(f"{name} must be a one-dimensional integer tensor")
        if values.device.type != "cpu":
            raise ValueError(f"{name} must reside on CPU")
        result = [operator.index(value) for value in values.tolist()]
    else:
        result = [operator.index(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicates")
    return result


def _digest(seed: int, namespace: str, role: str, *values: int) -> bytes:
    payload = ":".join((namespace, str(seed), role, *(str(value) for value in values)))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _matching_hash(recipients: list[int], matches: list[int], levels: list[int]) -> str:
    payload = [
        [recipient, match, level]
        for recipient, match, level in zip(recipients, matches, levels, strict=True)
    ]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_stratified_match(
    recipient_indices: Sequence[int] | torch.Tensor,
    candidate_indices: Sequence[int] | torch.Tensor,
    strata: Mapping[str, torch.Tensor],
    *,
    stages: Sequence[Sequence[str]] = DEFAULT_MATCH_STAGES,
    seed: int = DEFAULT_MATCH_SEED,
    namespace: str = "controls",
    document_ids: torch.Tensor | None = None,
    require_different_document: bool = False,
    forbidden_matches: Mapping[int, Collection[int]] | None = None,
) -> MatchResult:
    """Find a deterministic, full, without-replacement stratified matching.

    Candidate preferences obey the frozen stage order and SHA-256 breaks ties.
    A deterministic augmenting-path matcher is used instead of a fragile greedy
    assignment, so a late recipient cannot strand an otherwise valid control or
    donor map.  Returned arrays follow the caller's recipient order.
    """
    recipients = _indices(recipient_indices, "recipient_indices")
    candidates = _indices(candidate_indices, "candidate_indices")
    if len(candidates) < len(recipients):
        raise MatchingError("fewer candidates than recipients")
    if not namespace:
        raise ValueError("namespace must be nonempty")
    seed = operator.index(seed)
    if not stages or any(not stage for stage in stages):
        raise ValueError("stages must be a nonempty sequence of nonempty field lists")
    needed_fields = {field for stage in stages for field in stage}
    if not needed_fields.issubset(strata):
        raise ValueError(f"missing matching strata: {sorted(needed_fields - set(strata))}")
    maximum_index = max(recipients + candidates, default=-1)
    for field in needed_fields:
        values = strata[field]
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 1
            or values.device.type != "cpu"
            or values.is_floating_point()
            or values.is_complex()
        ):
            raise ValueError(f"stratum {field!r} must be a one-dimensional CPU integer tensor")
        if len(values) <= maximum_index:
            raise ValueError(f"stratum {field!r} does not cover every requested index")
    if require_different_document:
        if (
            document_ids is None
            or document_ids.ndim != 1
            or document_ids.device.type != "cpu"
            or document_ids.is_floating_point()
            or document_ids.is_complex()
            or len(document_ids) <= maximum_index
        ):
            raise ValueError(
                "different-document matching requires a covering one-dimensional CPU integer tensor"
            )
    forbidden = {
        operator.index(recipient): {operator.index(candidate) for candidate in values}
        for recipient, values in (forbidden_matches or {}).items()
    }

    stratum_lists = {name: tensor.tolist() for name, tensor in strata.items() if name in needed_fields}
    document_list = document_ids.tolist() if document_ids is not None else None

    # Index candidates by every registered stage key once.  The earlier
    # implementation compared every recipient with every candidate in Python,
    # which is exact but needlessly expensive for the 144 same-half control
    # matchings in rung 521.  Walking these buckets in stage order produces the
    # identical first-compatible-stage relation without materializing all
    # recipient x candidate comparisons.
    stage_buckets: list[tuple[tuple[str, ...], dict[tuple[int, ...], list[int]]]] = []
    for stage in stages:
        fields = tuple(stage)
        buckets: dict[tuple[int, ...], list[int]] = {}
        for candidate in candidates:
            key = tuple(stratum_lists[field][candidate] for field in fields)
            buckets.setdefault(key, []).append(candidate)
        stage_buckets.append((fields, buckets))

    adjacency: dict[int, list[tuple[int, int]]] = {}
    for recipient in recipients:
        viable: list[tuple[int, int]] = []
        seen: set[int] = set()
        for level, (fields, buckets) in enumerate(stage_buckets):
            key = tuple(stratum_lists[field][recipient] for field in fields)
            for candidate in buckets.get(key, ()):
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate in forbidden.get(recipient, ()):
                    continue
                if (
                    require_different_document
                    and document_list[recipient] == document_list[candidate]
                ):
                    continue
                viable.append((level, candidate))
        viable.sort(
            key=lambda item: (
                item[0],
                _digest(seed, namespace, "candidate", recipient, item[1]),
                item[1],
            )
        )
        adjacency[recipient] = viable

    candidate_owner: dict[int, int] = {}
    recipient_match: dict[int, tuple[int, int]] = {}

    def augment(recipient: int, seen_candidates: set[int], seen_recipients: set[int]) -> bool:
        if recipient in seen_recipients:
            return False
        seen_recipients.add(recipient)
        for level, candidate in adjacency[recipient]:
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)
            owner = candidate_owner.get(candidate)
            if owner is None or augment(owner, seen_candidates, seen_recipients):
                candidate_owner[candidate] = recipient
                recipient_match[recipient] = (candidate, level)
                return True
        return False

    recipient_order = sorted(
        recipients,
        key=lambda recipient: (
            _digest(seed, namespace, "recipient", recipient),
            recipient,
        ),
    )
    for recipient in recipient_order:
        if not augment(recipient, set(), set()):
            viable_counts = {value: len(adjacency[value]) for value in recipients}
            raise MatchingError(
                "registered relaxation ladder has no full without-replacement matching; "
                f"failed recipient={recipient}, viable_counts={viable_counts}"
            )

    matches = [recipient_match[recipient][0] for recipient in recipients]
    levels = [recipient_match[recipient][1] for recipient in recipients]
    counts = tuple(levels.count(level) for level in range(len(stages)))
    return MatchResult(
        recipient_indices=torch.tensor(recipients, dtype=torch.int64),
        matched_indices=torch.tensor(matches, dtype=torch.int64),
        relaxation_levels=torch.tensor(levels, dtype=torch.int64),
        relaxation_counts=counts,
        sha256=_matching_hash(recipients, matches, levels),
    )


def deterministic_donor_maps(
    recipient_indices: Sequence[int] | torch.Tensor,
    candidate_indices: Sequence[int] | torch.Tensor,
    strata: Mapping[str, torch.Tensor],
    document_ids: torch.Tensor,
    *,
    count: int = 4,
    seed: int = DEFAULT_MATCH_SEED,
    namespace: str,
    prior_maps: Sequence[MatchResult] = (),
) -> tuple[MatchResult, ...]:
    """Construct different-document donor maps with no repeated donor per recipient.

    If recipients and candidates are the same equally sized set, each result is
    a literal fixed-point-free permutation.  Pass ``D0`` as ``prior_maps`` while
    building ``D1`` to force distinct donors for each recipient across ensembles.
    """
    count = operator.index(count)
    if count <= 0:
        raise ValueError("count must be positive")
    recipients = _indices(recipient_indices, "recipient_indices")
    forbidden: dict[int, set[int]] = {recipient: set() for recipient in recipients}
    for prior in prior_maps:
        if prior.recipient_indices.tolist() != recipients:
            raise ValueError("prior donor map has different recipients or recipient order")
        for recipient, donor in zip(recipients, prior.matched_indices.tolist(), strict=True):
            forbidden[recipient].add(donor)
    results = []
    for map_index in range(count):
        result = deterministic_stratified_match(
            recipients,
            candidate_indices,
            strata,
            seed=seed,
            namespace=f"{namespace}:{map_index}",
            document_ids=document_ids,
            require_different_document=True,
            forbidden_matches=forbidden,
        )
        results.append(result)
        for recipient, donor in zip(recipients, result.matched_indices.tolist(), strict=True):
            forbidden[recipient].add(donor)
    return tuple(results)


def deterministic_row_donor_maps(
    split_row_indices: Sequence[int] | torch.Tensor,
    row_document_ids: Sequence[int] | torch.Tensor,
    *,
    row_ce_deciles: Sequence[int] | torch.Tensor | None = None,
    count: int = 4,
    seed: int = DEFAULT_MATCH_SEED,
    namespace: str,
    prior_maps: Sequence[MatchResult] = (),
) -> tuple[MatchResult, ...]:
    """Build scalable, different-document permutations of rows in one split.

    ``row_document_ids`` and optional ``row_ce_deciles`` are aligned with
    ``split_row_indices`` rather than indexed by their values.  Each returned
    map is a permutation of the supplied row indices.  A caller can therefore
    expand row ``recipient -> donor`` at the same token position without ever
    constructing a quadratic graph over all token positions.

    When CE deciles are supplied, same-decile donors are preferred and a
    different-decile donor is relaxation level 1.  The document constraint is
    never relaxed.  Donors are distinct for each recipient across every map in
    this call and every supplied ``prior_maps`` entry.
    """
    rows = _indices(split_row_indices, "split_row_indices")
    documents = _indices_aligned(row_document_ids, len(rows), "row_document_ids")
    deciles = (
        None
        if row_ce_deciles is None
        else _indices_aligned(row_ce_deciles, len(rows), "row_ce_deciles")
    )
    count = operator.index(count)
    seed = operator.index(seed)
    if count <= 0:
        raise ValueError("count must be positive")
    if not namespace:
        raise ValueError("namespace must be nonempty")
    if len(rows) < 2:
        raise MatchingError("a row derangement requires at least two rows")

    row_to_local = {row: local for local, row in enumerate(rows)}
    forbidden: dict[int, set[int]] = {row: set() for row in rows}
    for prior in prior_maps:
        if prior.recipient_indices.tolist() != rows:
            raise ValueError("prior row donor map has different recipients or recipient order")
        donors = prior.matched_indices.tolist()
        if sorted(donors) != sorted(rows):
            raise ValueError("prior row donor map is not a permutation of the split rows")
        for recipient, donor in zip(rows, donors, strict=True):
            if documents[row_to_local[recipient]] == documents[row_to_local[donor]]:
                raise ValueError("prior row donor map contains a same-document donor")
            forbidden[recipient].add(donor)

    results: list[MatchResult] = []
    for map_index in range(count):
        map_namespace = f"{namespace}:{map_index}"
        adjacency: dict[int, list[tuple[int, int]]] = {}
        for recipient_local, recipient in enumerate(rows):
            viable = []
            for donor_local, donor in enumerate(rows):
                if documents[recipient_local] == documents[donor_local]:
                    continue
                if donor in forbidden[recipient]:
                    continue
                level = 0 if deciles is None or deciles[recipient_local] == deciles[donor_local] else 1
                viable.append((level, donor))
            viable.sort(
                key=lambda item: (
                    item[0],
                    _digest(seed, map_namespace, "row-donor", recipient, item[1]),
                    item[1],
                )
            )
            adjacency[recipient] = viable

        candidate_owner: dict[int, int] = {}
        recipient_match: dict[int, tuple[int, int]] = {}

        def augment(recipient: int, seen_donors: set[int], seen_recipients: set[int]) -> bool:
            if recipient in seen_recipients:
                return False
            seen_recipients.add(recipient)
            for level, donor in adjacency[recipient]:
                if donor in seen_donors:
                    continue
                seen_donors.add(donor)
                owner = candidate_owner.get(donor)
                if owner is None or augment(owner, seen_donors, seen_recipients):
                    candidate_owner[donor] = recipient
                    recipient_match[recipient] = (donor, level)
                    return True
            return False

        recipient_order = sorted(
            rows,
            key=lambda recipient: (
                _digest(seed, map_namespace, "row-recipient", recipient),
                recipient,
            ),
        )
        for recipient in recipient_order:
            if not augment(recipient, set(), set()):
                viable_counts = {row: len(adjacency[row]) for row in rows}
                raise MatchingError(
                    "no full different-document row permutation remains; "
                    f"map_index={map_index}, failed recipient={recipient}, "
                    f"viable_counts={viable_counts}"
                )

        donors = [recipient_match[row][0] for row in rows]
        levels = [recipient_match[row][1] for row in rows]
        if sorted(donors) != sorted(rows):
            raise RuntimeError("internal error: row donor assignment is not a permutation")
        if any(
            documents[local] == documents[row_to_local[donor]]
            for local, donor in enumerate(donors)
        ):
            raise RuntimeError("internal error: row donor assignment retained a document")
        level_count = 1 if deciles is None else 2
        result = MatchResult(
            recipient_indices=torch.tensor(rows, dtype=torch.int64),
            matched_indices=torch.tensor(donors, dtype=torch.int64),
            relaxation_levels=torch.tensor(levels, dtype=torch.int64),
            relaxation_counts=tuple(levels.count(level) for level in range(level_count)),
            sha256=_matching_hash(rows, donors, levels),
        )
        results.append(result)
        for recipient, donor in zip(rows, donors, strict=True):
            forbidden[recipient].add(donor)
    return tuple(results)


def _indices_aligned(
    values: Sequence[int] | torch.Tensor, expected_length: int, name: str
) -> list[int]:
    """Read aligned integer metadata without imposing the uniqueness of row IDs."""
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.is_floating_point() or values.is_complex():
            raise ValueError(f"{name} must be a one-dimensional integer tensor")
        if values.device.type != "cpu":
            raise ValueError(f"{name} must reside on CPU")
        result = [operator.index(value) for value in values.tolist()]
    else:
        result = [operator.index(value) for value in values]
    if len(result) != expected_length:
        raise ValueError(f"{name} must have one entry per split row")
    return result


__all__ = [
    "DEFAULT_FOLD_NAMESPACE",
    "DEFAULT_MATCH_SEED",
    "DEFAULT_MATCH_STAGES",
    "MatchResult",
    "MatchingError",
    "deterministic_donor_maps",
    "deterministic_row_donor_maps",
    "deterministic_stratified_match",
    "document_fold",
    "document_folds",
    "exact_overlap_lattice",
    "exclusive_masks",
    "fit_projected_mean",
    "make_matching_strata",
    "mask_membership_codes",
    "mean_centered_projection_removal",
    "mobius_from_subset_values",
    "normalized_projector_overlap",
    "orthonormality_error",
    "overlap_lattice_counts",
    "principal_angles",
    "principal_cosines",
    "projection_interchange",
    "projector_frobenius_distance",
    "projector_matrix",
    "projector_overlap",
    "registered_split_mask",
    "retract_against_fixed",
    "retract_frame_blocks",
    "subset_values_from_mobius",
    "symmetric_polar_retraction",
    "validate_orthonormal_frame",
]
