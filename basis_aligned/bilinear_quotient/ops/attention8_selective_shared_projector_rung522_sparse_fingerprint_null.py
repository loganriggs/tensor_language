"""Sparse, exact CPU evaluation of the rung-522 fingerprint null.

The registered null permutes a saved scalar response inside coarse metadata
strata by a SHA-defined affine bijection.  A literal implementation builds a
donor index for every position, even though the statistic reads only the
member/control positions of the 32 registered circuits.  This module preserves
the literal map exactly while materializing donor indices only for that queried
union, in bounded replicate batches.

No model, experiment runner, data artifact, or CUDA code is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import operator
from typing import Mapping, Sequence

import torch


FINGERPRINT_NULL_NAMESPACE = "a8-r522-fingerprint-null-v1"
ENGINE_SCHEMA = "a8-r522-sparse-affine-fingerprint-null-v1"
REGISTERED_REPLICATES = 20_000
REGISTERED_CIRCUITS = 32
REGISTERED_QUARTET = 4


@dataclass(frozen=True)
class IndexPairs:
    """Local response indices for one circuit's matched member/control pairs."""

    members: tuple[int, ...]
    controls: tuple[int, ...]

    @classmethod
    def from_sequences(
        cls,
        members: Sequence[int] | torch.Tensor,
        controls: Sequence[int] | torch.Tensor,
    ) -> "IndexPairs":
        member_values = _integers(members, "members")
        control_values = _integers(controls, "controls")
        if not member_values or len(member_values) != len(control_values):
            raise ValueError("member/control index arrays must have one equal positive length")
        return cls(member_values, control_values)


@dataclass(frozen=True)
class FullMapAudit:
    replicate: int
    donor_indices: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class SparseAffineNullResult:
    circuit_order: tuple[str, ...]
    quartet_tags: tuple[str, ...]
    coordinates: torch.Tensor
    separations: torch.Tensor
    null_q95_higher: float
    queried_position_count: int
    full_position_count: int
    replicate_batch_size: int
    maximum_materialized_sparse_map_elements: int
    first_full_map_sha256: str
    last_full_map_sha256: str
    algorithm_definition: Mapping[str, object]
    algorithm_definition_sha256: str
    statistic_vector_sha256: str
    algorithm_and_statistic_sha256: str


@dataclass(frozen=True)
class _PreparedGroup:
    key: tuple[int, ...]
    members: torch.Tensor
    queried_offsets: torch.Tensor
    queried_ranks: torch.Tensor


@dataclass(frozen=True)
class _PreparedDesign:
    position_ids: tuple[int, ...]
    groups: tuple[_PreparedGroup, ...]
    query_indices: tuple[int, ...]


def _integers(values: Sequence[int] | torch.Tensor, name: str) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu" or values.ndim != 1 or values.is_floating_point():
            raise ValueError(f"{name} must be a one-dimensional integer CPU tensor")
        raw = values.tolist()
    else:
        raw = values
    try:
        return tuple(operator.index(value) for value in raw)
    except TypeError as error:
        raise ValueError(f"{name} must contain integers") from error


def _finite_response(response: Sequence[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(response, torch.Tensor):
        if response.device.type != "cpu" or response.ndim not in (1, 2):
            raise ValueError("response must be a one- or two-dimensional CPU tensor")
        value = response.detach().reshape(-1).to(torch.float64).contiguous()
    else:
        value = torch.tensor(tuple(response), dtype=torch.float64)
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError("response must be nonempty and finite")
    return value


def _metadata(
    position_ids: Sequence[int] | torch.Tensor,
    token_classes: Sequence[int] | torch.Tensor,
    position_bins: Sequence[int] | torch.Tensor,
    ce_deciles: Sequence[int] | torch.Tensor,
    fold_ids: Sequence[int] | torch.Tensor | None,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    ids = _integers(position_ids, "position_ids")
    classes = _integers(token_classes, "token_classes")
    bins = _integers(position_bins, "position_bins")
    deciles = _integers(ce_deciles, "ce_deciles")
    if not ids or any(len(values) != len(ids) for values in (classes, bins, deciles)):
        raise ValueError("metadata arrays must have one equal positive length")
    if len(set(ids)) != len(ids):
        raise ValueError("position_ids must be unique")
    if fold_ids is None:
        keys = tuple(zip(classes, bins, deciles, strict=True))
    else:
        folds = _integers(fold_ids, "fold_ids")
        if len(folds) != len(ids):
            raise ValueError("fold_ids must align with all other metadata")
        keys = tuple(zip(folds, classes, bins, deciles, strict=True))
    return ids, keys


def _affine_parameters(
    key: tuple[int, ...], cell_id: str, replicate: int, count: int
) -> tuple[int, int]:
    if count <= 1:
        return 1, 0
    payload = ":".join(
        (
            FINGERPRINT_NULL_NAMESPACE,
            cell_id,
            str(replicate),
            *(str(value) for value in key),
        )
    ).encode()
    digest = hashlib.sha256(payload).digest()
    offset = int.from_bytes(digest[:8], "little", signed=False) % count
    multiplier = 1 + int.from_bytes(digest[8:16], "little", signed=False) % (count - 1)
    while math.gcd(multiplier, count) != 1:
        multiplier = 1 if multiplier == count - 1 else multiplier + 1
    return multiplier, offset


def _full_map_hash(cell_id: str, replicate: int, donors: Sequence[int]) -> str:
    encoded = json.dumps(
        {
            "namespace": FINGERPRINT_NULL_NAMESPACE,
            "cell_id": cell_id,
            "replicate": replicate,
            "donor_indices": list(donors),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def literal_full_affine_map(
    position_ids: Sequence[int] | torch.Tensor,
    *,
    token_classes: Sequence[int] | torch.Tensor,
    position_bins: Sequence[int] | torch.Tensor,
    ce_deciles: Sequence[int] | torch.Tensor,
    cell_id: str,
    replicate: int,
    fold_ids: Sequence[int] | torch.Tensor | None = None,
) -> FullMapAudit:
    """Build the complete reference map, including its registered JSON hash."""
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a nonempty string")
    replicate = operator.index(replicate)
    if replicate < 0 or replicate >= REGISTERED_REPLICATES:
        raise ValueError("replicate must lie in 0..19999")
    ids, keys = _metadata(
        position_ids, token_classes, position_bins, ce_deciles, fold_ids
    )
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    donors = [-1] * len(ids)
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda index: (ids[index], index))
        count = len(members)
        multiplier, offset = _affine_parameters(key, cell_id, replicate, count)
        for rank, recipient in enumerate(members):
            donor_rank = (multiplier * rank + offset) % count
            donors[recipient] = members[donor_rank]
    if any(index < 0 for index in donors) or len(set(donors)) != len(donors):
        raise RuntimeError("literal affine map is not a complete bijection")
    return FullMapAudit(
        replicate=replicate,
        donor_indices=tuple(donors),
        sha256=_full_map_hash(cell_id, replicate, donors),
    )


def _prepare_design(
    position_ids: Sequence[int] | torch.Tensor,
    token_classes: Sequence[int] | torch.Tensor,
    position_bins: Sequence[int] | torch.Tensor,
    ce_deciles: Sequence[int] | torch.Tensor,
    query_indices: Sequence[int] | torch.Tensor,
    fold_ids: Sequence[int] | torch.Tensor | None,
) -> _PreparedDesign:
    ids, keys = _metadata(
        position_ids, token_classes, position_bins, ce_deciles, fold_ids
    )
    queries = tuple(sorted(set(_integers(query_indices, "query_indices"))))
    if not queries or queries[0] < 0 or queries[-1] >= len(ids):
        raise ValueError("query_indices must be nonempty local response indices")
    query_offset = {index: offset for offset, index in enumerate(queries)}
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    prepared = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda index: (ids[index], index))
        ranks = {index: rank for rank, index in enumerate(members)}
        selected = [index for index in members if index in query_offset]
        if not selected:
            continue
        prepared.append(
            _PreparedGroup(
                key=key,
                members=torch.tensor(members, dtype=torch.int64),
                queried_offsets=torch.tensor(
                    [query_offset[index] for index in selected], dtype=torch.int64
                ),
                queried_ranks=torch.tensor(
                    [ranks[index] for index in selected], dtype=torch.int64
                ),
            )
        )
    covered = sum(group.queried_offsets.numel() for group in prepared)
    if covered != len(queries):
        raise RuntimeError("sparse affine preparation lost a queried position")
    return _PreparedDesign(ids, tuple(prepared), queries)


def sparse_affine_donor_indices(
    position_ids: Sequence[int] | torch.Tensor,
    *,
    token_classes: Sequence[int] | torch.Tensor,
    position_bins: Sequence[int] | torch.Tensor,
    ce_deciles: Sequence[int] | torch.Tensor,
    query_indices: Sequence[int] | torch.Tensor,
    cell_id: str,
    replicate: int,
    fold_ids: Sequence[int] | torch.Tensor | None = None,
) -> tuple[int, ...]:
    """Return exact donor indices only for sorted unique queried positions."""
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a nonempty string")
    replicate = operator.index(replicate)
    if replicate < 0 or replicate >= REGISTERED_REPLICATES:
        raise ValueError("replicate must lie in 0..19999")
    design = _prepare_design(
        position_ids,
        token_classes,
        position_bins,
        ce_deciles,
        query_indices,
        fold_ids,
    )
    result = torch.full((len(design.query_indices),), -1, dtype=torch.int64)
    for group in design.groups:
        multiplier, offset = _affine_parameters(
            group.key, cell_id, replicate, group.members.numel()
        )
        donor_ranks = (
            multiplier * group.queried_ranks + offset
        ) % group.members.numel()
        result[group.queried_offsets] = group.members[donor_ranks]
    if bool((result < 0).any()):
        raise RuntimeError("sparse affine map left a query unresolved")
    return tuple(int(value) for value in result.tolist())


def _normalize_pairs(
    circuit_pairs: Mapping[str, IndexPairs], position_count: int
) -> tuple[tuple[str, ...], dict[str, IndexPairs]]:
    if len(circuit_pairs) != REGISTERED_CIRCUITS:
        raise ValueError("fingerprint null requires exactly 32 circuits")
    ordered = tuple(sorted(circuit_pairs))
    normalized = {}
    for name in ordered:
        pair = circuit_pairs[name]
        if not isinstance(name, str) or not name or not isinstance(pair, IndexPairs):
            raise ValueError("circuit pairs must map nonempty names to IndexPairs")
        if not pair.members or len(pair.members) != len(pair.controls):
            raise ValueError(f"{name} has empty or unequal matched index arrays")
        values = pair.members + pair.controls
        if min(values) < 0 or max(values) >= position_count:
            raise ValueError(f"{name} contains an index outside the response")
        normalized[name] = pair
    return ordered, normalized


def _higher_quantile(values: torch.Tensor, probability: float) -> float:
    ordered = values.sort().values
    index = math.ceil(probability * (ordered.numel() - 1))
    return float(ordered[index])


def evaluate_sparse_affine_fingerprint_null(
    response: Sequence[float] | torch.Tensor,
    position_ids: Sequence[int] | torch.Tensor,
    *,
    token_classes: Sequence[int] | torch.Tensor,
    position_bins: Sequence[int] | torch.Tensor,
    ce_deciles: Sequence[int] | torch.Tensor,
    circuit_pairs: Mapping[str, IndexPairs],
    quartet_tags: Sequence[str],
    cell_id: str,
    fold_ids: Sequence[int] | torch.Tensor | None = None,
    replicates: int = REGISTERED_REPLICATES,
    replicate_batch_size: int = 128,
) -> SparseAffineNullResult:
    """Compute all 32 coordinates and S for streamed affine-null replicates.

    Only the union of member/control recipient positions receives a donor map.
    Complete maps are constructed separately for the first and last replicate
    solely to provide the registered audit hashes.
    """
    values = _finite_response(response)
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a nonempty string")
    replicates = operator.index(replicates)
    replicate_batch_size = operator.index(replicate_batch_size)
    if replicates <= 0 or replicates > REGISTERED_REPLICATES:
        raise ValueError("replicates must lie in 1..20000")
    if replicate_batch_size <= 0:
        raise ValueError("replicate_batch_size must be positive")
    circuit_order, pairs = _normalize_pairs(circuit_pairs, values.numel())
    quartet = tuple(quartet_tags)
    if len(quartet) != REGISTERED_QUARTET or len(set(quartet)) != len(quartet):
        raise ValueError("quartet_tags must contain exactly four distinct circuits")
    if not set(quartet) <= set(circuit_order):
        raise ValueError("quartet_tags must be part of the 32-circuit battery")

    queries = sorted(
        {
            index
            for pair in pairs.values()
            for index in pair.members + pair.controls
        }
    )
    design = _prepare_design(
        position_ids,
        token_classes,
        position_bins,
        ce_deciles,
        queries,
        fold_ids,
    )
    if len(design.position_ids) != values.numel():
        raise ValueError("response and metadata lengths differ")
    offsets = {index: offset for offset, index in enumerate(design.query_indices)}
    pair_offsets = {
        name: (
            torch.tensor([offsets[index] for index in pairs[name].members], dtype=torch.int64),
            torch.tensor([offsets[index] for index in pairs[name].controls], dtype=torch.int64),
        )
        for name in circuit_order
    }
    quartet_columns = torch.tensor(
        [circuit_order.index(name) for name in quartet], dtype=torch.int64
    )
    nonquartet_columns = torch.tensor(
        [index for index, name in enumerate(circuit_order) if name not in quartet],
        dtype=torch.int64,
    )
    coordinates = torch.empty(
        (replicates, REGISTERED_CIRCUITS), dtype=torch.float64
    )
    separations = torch.empty(replicates, dtype=torch.float64)
    response_squared = values.square()
    maximum_elements = 0

    for start in range(0, replicates, replicate_batch_size):
        stop = min(replicates, start + replicate_batch_size)
        batch_replicates = tuple(range(start, stop))
        sparse_donors = torch.full(
            (len(batch_replicates), len(design.query_indices)),
            -1,
            dtype=torch.int64,
        )
        maximum_elements = max(maximum_elements, sparse_donors.numel())
        for group in design.groups:
            count = group.members.numel()
            parameters = [
                _affine_parameters(group.key, cell_id, replicate, count)
                for replicate in batch_replicates
            ]
            multipliers = torch.tensor(
                [value[0] for value in parameters], dtype=torch.int64
            )[:, None]
            offsets_by_replicate = torch.tensor(
                [value[1] for value in parameters], dtype=torch.int64
            )[:, None]
            donor_ranks = (
                multipliers * group.queried_ranks[None, :] + offsets_by_replicate
            ) % count
            sparse_donors[:, group.queried_offsets] = group.members[donor_ranks]
        if bool((sparse_donors < 0).any()):
            raise RuntimeError("batched sparse affine map left a query unresolved")
        queried_squares = response_squared[sparse_donors]
        for column, name in enumerate(circuit_order):
            member_offsets, control_offsets = pair_offsets[name]
            member_rms = queried_squares[:, member_offsets].mean(1).sqrt()
            control_rms = queried_squares[:, control_offsets].mean(1).sqrt()
            coordinates[start:stop, column] = member_rms - control_rms
        separations[start:stop] = (
            coordinates[start:stop, quartet_columns].min(1).values
            - coordinates[start:stop, nonquartet_columns].max(1).values
        )

    first_map = literal_full_affine_map(
        position_ids,
        token_classes=token_classes,
        position_bins=position_bins,
        ce_deciles=ce_deciles,
        fold_ids=fold_ids,
        cell_id=cell_id,
        replicate=0,
    )
    last_map = literal_full_affine_map(
        position_ids,
        token_classes=token_classes,
        position_bins=position_bins,
        ce_deciles=ce_deciles,
        fold_ids=fold_ids,
        cell_id=cell_id,
        replicate=replicates - 1,
    )
    grouping_fields = (
        ["token_class", "position_bin_32", "native_CE_decile"]
        if fold_ids is None
        else ["fold_id", "token_class", "position_bin_32", "native_CE_decile"]
    )
    definition: dict[str, object] = {
        "schema": ENGINE_SCHEMA,
        "namespace": FINGERPRINT_NULL_NAMESPACE,
        "cell_id": cell_id,
        "replicates": replicates,
        "grouping_fields_in_sha_payload_order": grouping_fields,
        "recipient_order": "global_position_then_local_index",
        "map": "donor_rank=(coprime_multiplier*recipient_rank+offset)%group_size",
        "offset_source": "sha256_bytes_0_8_little_endian_mod_group_size",
        "multiplier_source": "sha256_bytes_8_16_little_endian_then_cyclic_coprime",
        "circuit_order": list(circuit_order),
        "quartet_tags": list(quartet),
        "coordinate": "RMS(member)-RMS(control)",
        "separation": "min(quartet coordinate)-max(nonquartet coordinate)",
        "evaluation": "sparse_query_union_streamed_replicate_batches",
    }
    definition_bytes = json.dumps(
        definition, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    statistic_bytes = separations.contiguous().numpy().tobytes(order="C")
    definition_hash = hashlib.sha256(definition_bytes).hexdigest()
    statistic_hash = hashlib.sha256(statistic_bytes).hexdigest()
    combined = hashlib.sha256(definition_bytes + b"\0" + statistic_bytes).hexdigest()
    return SparseAffineNullResult(
        circuit_order=circuit_order,
        quartet_tags=quartet,
        coordinates=coordinates,
        separations=separations,
        null_q95_higher=_higher_quantile(separations, 0.95),
        queried_position_count=len(design.query_indices),
        full_position_count=values.numel(),
        replicate_batch_size=replicate_batch_size,
        maximum_materialized_sparse_map_elements=maximum_elements,
        first_full_map_sha256=first_map.sha256,
        last_full_map_sha256=last_map.sha256,
        algorithm_definition=definition,
        algorithm_definition_sha256=definition_hash,
        statistic_vector_sha256=statistic_hash,
        algorithm_and_statistic_sha256=combined,
    )


__all__ = [
    "ENGINE_SCHEMA",
    "FINGERPRINT_NULL_NAMESPACE",
    "FullMapAudit",
    "IndexPairs",
    "REGISTERED_CIRCUITS",
    "REGISTERED_QUARTET",
    "REGISTERED_REPLICATES",
    "SparseAffineNullResult",
    "evaluate_sparse_affine_fingerprint_null",
    "literal_full_affine_map",
    "sparse_affine_donor_indices",
]
