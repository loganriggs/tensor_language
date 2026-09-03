import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


PATH = Path(__file__).with_name("das_shared_private_lib.py")
SPEC = importlib.util.spec_from_file_location("das_shared_private_lib", PATH)
LIB = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LIB
SPEC.loader.exec_module(LIB)


def test_polar_retraction_is_orthonormal_span_preserving_and_differentiable():
    generator = torch.Generator().manual_seed(521)
    matrix = torch.randn(12, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    frame = LIB.symmetric_polar_retraction(matrix)
    assert float(LIB.orthonormality_error(frame).detach()) < 2e-14
    residual = matrix - frame @ (frame.mT @ matrix)
    assert float(residual.norm().detach()) < 2e-13
    frame.square().sum().backward()
    assert matrix.grad is not None
    assert bool(torch.isfinite(matrix.grad).all())


def test_retract_against_fixed_preserves_fixed_frame_and_is_orthogonal():
    fixed = torch.eye(9, dtype=torch.float64)[:, :3]
    candidate = torch.randn(9, 4, generator=torch.Generator().manual_seed(522), dtype=torch.float64)
    private = LIB.retract_against_fixed(candidate, [fixed])
    assert torch.equal(fixed, torch.eye(9, dtype=torch.float64)[:, :3])
    assert float((fixed.mT @ private).abs().max()) < 1e-12
    assert float(LIB.orthonormality_error(private)) < 1e-12


def test_interchange_is_projector_gauge_invariant_and_composes_for_orthogonal_frames():
    recipient = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    donor = torch.tensor([[5.0, 8.0, 13.0, 21.0]], dtype=torch.float64)
    q1 = torch.eye(4, dtype=torch.float64)[:, :2]
    q2 = torch.eye(4, dtype=torch.float64)[:, 2:]
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    first = LIB.projection_interchange(recipient, donor, q1)
    gauge = LIB.projection_interchange(recipient, donor, q1 @ rotation)
    sequential = LIB.projection_interchange(first, donor, q2)
    joint = LIB.projection_interchange(recipient, donor, torch.cat((q1, q2), dim=1))
    assert torch.equal(first, torch.tensor([[5.0, 8.0, 3.0, 4.0]], dtype=torch.float64))
    assert torch.equal(first, gauge)
    assert torch.equal(sequential, joint)
    assert torch.equal(joint, donor)


def test_mean_centered_removal_sets_projected_coordinates_to_fit_mean():
    frame = torch.eye(4, dtype=torch.float64)[:, :2]
    fit = torch.tensor([[1.0, 4.0, 7.0, 8.0], [3.0, 8.0, 9.0, 10.0]], dtype=torch.float64)
    mean = LIB.fit_projected_mean(fit, frame)
    write = torch.tensor([[11.0, 12.0, 13.0, 14.0]], dtype=torch.float64)
    removed = LIB.mean_centered_projection_removal(write, frame, mean)
    assert torch.equal(mean, torch.tensor([2.0, 6.0], dtype=torch.float64))
    assert torch.equal(removed @ frame, mean.unsqueeze(0))
    assert torch.equal(removed[:, 2:], write[:, 2:])


def test_projector_metrics_have_expected_extremes_and_ignore_gauge():
    left = torch.eye(8, dtype=torch.float64)[:, :4]
    orthogonal = torch.eye(8, dtype=torch.float64)[:, 4:]
    rotation = torch.linalg.qr(torch.arange(1, 17, dtype=torch.float64).reshape(4, 4))[0]
    gauge = left @ rotation
    assert torch.allclose(
        LIB.principal_cosines(left, gauge), torch.ones(4, dtype=torch.float64), atol=1e-14
    )
    assert float(LIB.normalized_projector_overlap(left, gauge)) == pytest.approx(1.0)
    assert float(LIB.projector_frobenius_distance(left, gauge)) < 1e-7
    assert float(LIB.normalized_projector_overlap(left, orthogonal)) == 0.0
    assert float(LIB.projector_frobenius_distance(left, orthogonal)) == pytest.approx(8**0.5)


def test_boolean_mobius_round_trip_with_trailing_axes_and_gradients():
    coefficients = torch.randn(16, 3, 2, dtype=torch.float64, requires_grad=True)
    endpoints = LIB.subset_values_from_mobius(coefficients)
    recovered = LIB.mobius_from_subset_values(endpoints)
    assert torch.allclose(recovered, coefficients, atol=1e-12, rtol=0)
    recovered.square().sum().backward()
    assert coefficients.grad is not None
    assert bool(torch.isfinite(coefficients.grad).all())
    with pytest.raises(ValueError):
        LIB.mobius_from_subset_values(torch.zeros(15))


def test_document_fold_is_the_frozen_little_endian_sha_rule_and_splits_partition():
    document_ids = torch.tensor([[0, 1, 2], [123, 999, 2]], dtype=torch.int64)
    folds = LIB.document_folds(document_ids)
    expected = []
    for document_id in document_ids.reshape(-1).tolist():
        digest = hashlib.sha256(f"a8-shared-private-v1:{document_id}".encode()).digest()
        expected.append(int.from_bytes(digest[:8], "little") % 10)
    assert folds.shape == document_ids.shape
    assert folds.reshape(-1).tolist() == expected
    assert folds[0, 2] == folds[1, 2]
    partition = sum(
        LIB.registered_split_mask(folds, split).to(torch.int64)
        for split in ("FIT", "VALIDATION", "TEST")
    )
    assert torch.equal(partition, torch.ones_like(folds))


def test_exact_overlap_lattice_partitions_universe_and_exclusive_masks():
    masks = {
        "a": torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool),
        "b": torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.bool),
        "c": torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.bool),
    }
    universe = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    lattice = LIB.exact_overlap_lattice(masks, universe=universe)
    membership_count = sum(cell.to(torch.int64) for cell in lattice.values())
    assert torch.equal(membership_count, universe.to(torch.int64))
    exclusive = LIB.exclusive_masks(masks, universe=universe)
    assert torch.equal(exclusive["a"], torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.bool))
    assert not bool(exclusive["b"].any())
    assert not bool(exclusive["c"].any())
    assert LIB.overlap_lattice_counts(masks, universe=universe)[0b111] == 1
    assert LIB.overlap_lattice_counts(masks, universe=universe)[0] == 1


def test_registered_stratified_matching_uses_each_relaxation_level_deterministically():
    token = torch.tensor([10, 11, 12, 13, 10, 11, 99, 99])
    position = torch.tensor([0, 0, 64, 96, 1, 64, 65, 0])
    decile = torch.tensor([1, 1, 2, 3, 1, 1, 2, 3])
    token_class = torch.tensor([0, 1, 2, 3, 9, 9, 2, 3])
    strata = LIB.make_matching_strata(token, position, decile, token_class)
    first = LIB.deterministic_stratified_match(range(4), range(4, 8), strata)
    second = LIB.deterministic_stratified_match(range(4), range(4, 8), strata)
    assert first.matched_indices.tolist() == [4, 5, 6, 7]
    assert first.relaxation_levels.tolist() == [0, 1, 2, 3]
    assert first.relaxation_counts == (1, 1, 1, 1)
    assert first.sha256 == second.sha256
    assert len(set(first.matched_indices.tolist())) == 4


def test_donor_maps_are_bijections_with_different_documents_and_distinct_per_recipient():
    count = 7
    recipients = torch.arange(count)
    strata = {
        "token_id": torch.zeros(count, dtype=torch.int64),
        "position_bin": torch.zeros(count, dtype=torch.int64),
        "ce_decile": torch.zeros(count, dtype=torch.int64),
        "token_class": torch.zeros(count, dtype=torch.int64),
    }
    documents = torch.arange(count, dtype=torch.int64)
    d0 = LIB.deterministic_donor_maps(
        recipients, recipients, strata, documents, count=2, namespace="D0"
    )
    d1 = LIB.deterministic_donor_maps(
        recipients, recipients, strata, documents, count=2, namespace="D1", prior_maps=d0
    )
    all_maps = d0 + d1
    for result in all_maps:
        donors = result.matched_indices
        assert sorted(donors.tolist()) == list(range(count))
        assert bool((documents[recipients] != documents[donors]).all())
    for recipient in range(count):
        donors = [result.matched_indices[recipient].item() for result in all_maps]
        assert len(set(donors)) == len(donors)
    assert len({result.sha256 for result in all_maps}) == len(all_maps)


def test_row_donor_maps_are_deterministic_permutations_and_scale_at_row_level():
    rows = torch.arange(100, 120, 2, dtype=torch.int64)
    documents = torch.arange(len(rows), dtype=torch.int64)
    deciles = torch.arange(len(rows), dtype=torch.int64) % 2
    d0 = LIB.deterministic_row_donor_maps(
        rows, documents, row_ce_deciles=deciles, count=4, namespace="D0"
    )
    d0_repeat = LIB.deterministic_row_donor_maps(
        rows, documents, row_ce_deciles=deciles, count=4, namespace="D0"
    )
    d1 = LIB.deterministic_row_donor_maps(
        rows,
        documents,
        row_ce_deciles=deciles,
        count=4,
        namespace="D1",
        prior_maps=d0,
    )
    assert [result.sha256 for result in d0] == [result.sha256 for result in d0_repeat]
    all_maps = d0 + d1
    for result in all_maps:
        assert result.recipient_indices.tolist() == rows.tolist()
        assert sorted(result.matched_indices.tolist()) == sorted(rows.tolist())
        assert sum(result.relaxation_counts) == len(rows)
        for local, donor in enumerate(result.matched_indices.tolist()):
            donor_local = rows.tolist().index(donor)
            assert documents[local] != documents[donor_local]
    for local in range(len(rows)):
        donors = [result.matched_indices[local].item() for result in all_maps]
        assert len(set(donors)) == len(donors)


def test_row_donor_maps_keep_document_constraint_when_ce_decile_must_relax():
    rows = [11, 22, 33, 44]
    documents = [0, 1, 2, 3]
    deciles = [0, 0, 1, 1]
    maps = LIB.deterministic_row_donor_maps(
        rows, documents, row_ce_deciles=deciles, count=3, namespace="relax"
    )
    assert any(result.relaxation_counts[1] > 0 for result in maps)
    for result in maps:
        assert all(
            documents[index] != documents[rows.index(donor)]
            for index, donor in enumerate(result.matched_indices.tolist())
        )


def test_row_donor_maps_fail_when_document_majority_prevents_a_permutation():
    with pytest.raises(LIB.MatchingError):
        LIB.deterministic_row_donor_maps(
            [0, 1, 2, 3], [7, 7, 7, 8], count=1, namespace="impossible"
        )


def test_matching_fails_closed_when_no_registered_candidate_exists():
    strata = {
        "token_id": torch.tensor([1, 2]),
        "position_bin": torch.tensor([0, 1]),
        "ce_decile": torch.tensor([0, 1]),
        "token_class": torch.tensor([0, 1]),
    }
    with pytest.raises(LIB.MatchingError):
        LIB.deterministic_stratified_match([0], [1], strata)
