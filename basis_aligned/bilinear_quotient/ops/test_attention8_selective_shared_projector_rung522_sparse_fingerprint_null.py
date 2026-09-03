"""Exactness and bounded-memory tests for the rung-522 sparse fingerprint null."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
import torch


HERE = Path(__file__).parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SPARSE = _load(
    "rung522_sparse_fingerprint_null_test_module",
    "attention8_selective_shared_projector_rung522_sparse_fingerprint_null.py",
)
PROTOCOL = _load(
    "rung522_protocol_sparse_equivalence_test_module",
    "attention8_selective_shared_projector_rung522_protocol.py",
)


def _small_metadata(size: int):
    # Deliberately nonmonotone IDs exercise the registered within-group sort.
    position_ids = tuple(10_000 + ((17 * index) % size) for index in range(size))
    token_classes = tuple(index % 3 for index in range(size))
    position_bins = tuple((index // 3) % 2 for index in range(size))
    ce_deciles = tuple((index // 6) % 2 for index in range(size))
    return position_ids, token_classes, position_bins, ce_deciles


@pytest.mark.parametrize("replicate", [0, 1, 17, 19_999])
def test_standard_sparse_queries_and_full_hash_are_bit_exact_with_protocol(replicate):
    ids, classes, bins, deciles = _small_metadata(24)
    keyword = dict(
        token_classes=classes,
        position_bins=bins,
        ce_deciles=deciles,
        cell_id="test:D0:forward",
        replicate=replicate,
    )
    registered = PROTOCOL.stratified_affine_permutation(ids, **keyword)
    literal = SPARSE.literal_full_affine_map(ids, **keyword)
    assert literal.donor_indices == registered.donor_indices
    assert literal.sha256 == registered.sha256

    # Sparse output is explicitly in sorted-unique recipient order.
    queries = (21, 4, 21, 0, 13, 7)
    sparse = SPARSE.sparse_affine_donor_indices(
        ids, query_indices=queries, **keyword
    )
    expected = tuple(registered.donor_indices[index] for index in sorted(set(queries)))
    assert sparse == expected


def _manual_fold_map(ids, folds, classes, bins, deciles, cell_id, replicate):
    groups = {}
    for index, key in enumerate(
        zip(folds, classes, bins, deciles, strict=True)
    ):
        groups.setdefault(key, []).append(index)
    donors = [-1] * len(ids)
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda index: (ids[index], index))
        count = len(members)
        if count == 1:
            multiplier, offset = 1, 0
        else:
            payload = ":".join(
                (
                    SPARSE.FINGERPRINT_NULL_NAMESPACE,
                    cell_id,
                    str(replicate),
                    *(str(value) for value in key),
                )
            ).encode()
            digest = hashlib.sha256(payload).digest()
            offset = int.from_bytes(digest[:8], "little", signed=False) % count
            multiplier = 1 + int.from_bytes(
                digest[8:16], "little", signed=False
            ) % (count - 1)
            while math.gcd(multiplier, count) != 1:
                multiplier = 1 if multiplier == count - 1 else multiplier + 1
        for rank, recipient in enumerate(members):
            donors[recipient] = members[(multiplier * rank + offset) % count]
    return tuple(donors)


def test_fold_is_leading_sha_field_and_donors_never_cross_four_field_groups():
    ids, classes, bins, deciles = _small_metadata(30)
    folds = tuple((index // 5) % 2 for index in range(30))
    cell_id = "removal:D1:reverse"
    replicate = 123
    literal = SPARSE.literal_full_affine_map(
        ids,
        token_classes=classes,
        position_bins=bins,
        ce_deciles=deciles,
        fold_ids=folds,
        cell_id=cell_id,
        replicate=replicate,
    )
    manual = _manual_fold_map(
        ids, folds, classes, bins, deciles, cell_id, replicate
    )
    assert literal.donor_indices == manual
    fields = tuple(zip(folds, classes, bins, deciles, strict=True))
    assert all(
        fields[recipient] == fields[donor]
        for recipient, donor in enumerate(literal.donor_indices)
    )
    queries = (1, 8, 9, 14, 22, 29)
    sparse = SPARSE.sparse_affine_donor_indices(
        ids,
        token_classes=classes,
        position_bins=bins,
        ce_deciles=deciles,
        fold_ids=folds,
        query_indices=queries,
        cell_id=cell_id,
        replicate=replicate,
    )
    assert sparse == tuple(manual[index] for index in queries)


def _circuit_pairs(size: int):
    result = {}
    # Only 128 of `size` positions are queried. Each circuit receives disjoint
    # two-position member and control sets, which makes literal scoring simple.
    for circuit in range(32):
        start = 4 * circuit
        result[f"c{circuit:02d}"] = SPARSE.IndexPairs.from_sequences(
            (start, start + 1), (start + 2, start + 3)
        )
    assert 4 * len(result) <= size
    return result


def _literal_scores(response, metadata, pairs, quartet, cell_id, replicates):
    ids, classes, bins, deciles = metadata
    circuit_order = tuple(sorted(pairs))
    coordinates = torch.empty((replicates, 32), dtype=torch.float64)
    separations = torch.empty(replicates, dtype=torch.float64)
    quartet_columns = [circuit_order.index(name) for name in quartet]
    nonquartet_columns = [
        column for column, name in enumerate(circuit_order) if name not in quartet
    ]
    for replicate in range(replicates):
        permutation = PROTOCOL.stratified_affine_permutation(
            ids,
            token_classes=classes,
            position_bins=bins,
            ce_deciles=deciles,
            cell_id=cell_id,
            replicate=replicate,
        )
        permuted = response[torch.tensor(permutation.donor_indices)]
        for column, name in enumerate(circuit_order):
            member = torch.tensor(pairs[name].members)
            control = torch.tensor(pairs[name].controls)
            coordinates[replicate, column] = (
                permuted[member].square().mean().sqrt()
                - permuted[control].square().mean().sqrt()
            )
        separations[replicate] = (
            coordinates[replicate, quartet_columns].min()
            - coordinates[replicate, nonquartet_columns].max()
        )
    return coordinates, separations


def test_all_32_coordinates_and_separation_match_literal_full_map_scoring():
    metadata = _small_metadata(160)
    response = torch.linspace(-3.0, 5.0, 160, dtype=torch.float64).sin() * 7.0
    pairs = _circuit_pairs(160)
    quartet = ("c02", "c11", "c19", "c27")
    cell_id = "test:fingerprint:D0:forward"
    replicates = 19
    result = SPARSE.evaluate_sparse_affine_fingerprint_null(
        response,
        metadata[0],
        token_classes=metadata[1],
        position_bins=metadata[2],
        ce_deciles=metadata[3],
        circuit_pairs=pairs,
        quartet_tags=quartet,
        cell_id=cell_id,
        replicates=replicates,
        replicate_batch_size=7,
    )
    expected_coordinates, expected_separations = _literal_scores(
        response, metadata, pairs, quartet, cell_id, replicates
    )
    torch.testing.assert_close(result.coordinates, expected_coordinates, rtol=0, atol=0)
    torch.testing.assert_close(result.separations, expected_separations, rtol=0, atol=0)

    first = PROTOCOL.stratified_affine_permutation(
        metadata[0],
        token_classes=metadata[1],
        position_bins=metadata[2],
        ce_deciles=metadata[3],
        cell_id=cell_id,
        replicate=0,
    )
    last = PROTOCOL.stratified_affine_permutation(
        metadata[0],
        token_classes=metadata[1],
        position_bins=metadata[2],
        ce_deciles=metadata[3],
        cell_id=cell_id,
        replicate=replicates - 1,
    )
    assert result.first_full_map_sha256 == first.sha256
    assert result.last_full_map_sha256 == last.sha256

    definition_bytes = json.dumps(
        result.algorithm_definition,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    statistic_bytes = result.separations.contiguous().numpy().tobytes(order="C")
    assert result.algorithm_definition_sha256 == hashlib.sha256(
        definition_bytes
    ).hexdigest()
    assert result.statistic_vector_sha256 == hashlib.sha256(statistic_bytes).hexdigest()
    assert result.algorithm_and_statistic_sha256 == hashlib.sha256(
        definition_bytes + b"\0" + statistic_bytes
    ).hexdigest()


def test_large_frame_tracks_sparse_batch_memory_not_full_permutation_volume():
    size = 10_000
    ids = tuple(range(80_000, 80_000 + size))
    classes = tuple(index % 4 for index in range(size))
    bins = tuple((index // 4) % 32 for index in range(size))
    deciles = tuple((index // 128) % 10 for index in range(size))
    response = torch.cos(torch.arange(size, dtype=torch.float64) / 31)
    pairs = _circuit_pairs(size)
    result = SPARSE.evaluate_sparse_affine_fingerprint_null(
        response,
        ids,
        token_classes=classes,
        position_bins=bins,
        ce_deciles=deciles,
        circuit_pairs=pairs,
        quartet_tags=("c00", "c01", "c02", "c03"),
        cell_id="test:bounded-memory",
        replicates=257,
        replicate_batch_size=31,
    )
    assert result.coordinates.shape == (257, 32)
    assert result.separations.shape == (257,)
    assert result.queried_position_count == 128
    assert result.queried_position_count < result.full_position_count // 10
    assert result.maximum_materialized_sparse_map_elements <= 31 * 128
    assert result.maximum_materialized_sparse_map_elements < 257 * size
    assert len(result.statistic_vector_sha256) == 64
