import pytest
import torch

import hierarchical_shared_private_rrr as hybrid
import simultaneous_shared_output_rrr as independent


def _problem(n_sites=3, dimension=4, seed=4):
    generator = torch.Generator().manual_seed(seed)
    solved = []
    merits = []
    for _ in range(n_sites):
        value = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
        solved.append(value)
        merits.append(value.T @ value)
    return tuple(solved), tuple(merits)


def _projector(value):
    return value @ value.T


def test_known_shared_and_private_directions_are_recovered_exactly():
    solved = tuple(torch.eye(4, dtype=torch.float64) for _ in range(3))
    merits = (
        torch.diag(torch.tensor([10., 9., 1., 0.], dtype=torch.float64)),
        torch.diag(torch.tensor([10., 1., 8., 0.], dtype=torch.float64)),
        torch.diag(torch.tensor([10., 1., 0., 7.], dtype=torch.float64)),
    )
    shared = torch.eye(4, dtype=torch.float64)[:, :1]
    base = 4 * (3 + 1)
    fit = hybrid.fit_hierarchical_shared_private(
        solved, merits, shared, total_float_budget=base + 2 * 4 * 3,
    )
    assert fit.allocation.private_ranks == (1, 1, 1)
    assert fit.explained_shared_merit == pytest.approx(30.0)
    assert fit.explained_private_merit == pytest.approx(24.0)
    expected = (1, 2, 3)
    for site, coordinate in enumerate(expected):
        target = torch.zeros(4, 4, dtype=torch.float64)
        target[coordinate, coordinate] = 1
        assert torch.allclose(_projector(fit.private_bases[site]), target, atol=1e-12)


def test_residual_bases_are_orthogonal_to_shared_and_each_other():
    solved, merits = _problem()
    shared = hybrid.global_shared_basis(merits, 2)
    budget = 4 * 4 * 2 + 2 * 4 * 4
    fit = hybrid.fit_hierarchical_shared_private(
        solved, merits, shared, total_float_budget=budget,
    )
    assert fit.combined_orthogonality_max_abs < 1e-10
    for basis in fit.private_bases:
        assert float((shared.T @ basis).abs().max()) < 1e-10 if basis.numel() else True


def test_shared_gauge_rotation_leaves_site_projectors_and_maps_invariant():
    solved, merits = _problem(seed=8)
    shared = hybrid.global_shared_basis(merits, 2)
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=torch.float64)
    rotated = shared @ rotation
    budget = 4 * 4 * 2 + 2 * 4 * 3
    first = hybrid.fit_hierarchical_shared_private(
        solved, merits, shared, total_float_budget=budget,
    )
    second = hybrid.fit_hierarchical_shared_private(
        solved, merits, rotated, total_float_budget=budget,
    )
    for site in range(3):
        p1 = _projector(first.shared_basis) + _projector(first.private_bases[site])
        p2 = _projector(second.shared_basis) + _projector(second.private_bases[site])
        assert torch.allclose(p1, p2, atol=2e-10, rtol=2e-10)
    for left, right in zip(hybrid.coefficient_maps(first),
                           hybrid.coefficient_maps(second), strict=True):
        assert torch.allclose(left, right, atol=2e-10, rtol=2e-10)


def test_q0_zero_reduces_to_exact_price_independent_allocation():
    solved, merits = _problem()
    shared = torch.empty((4, 0), dtype=torch.float64)
    budget = independent.grouped_map_price(3, 1, 4, 4, 2).grouped_float_count
    fit = hybrid.fit_hierarchical_shared_private(
        solved, merits, shared, total_float_budget=budget,
    )
    spectra = []
    for merit in merits:
        spectra.append(torch.linalg.eigvalsh(merit).flip(0).clamp_min(0))
    expected = independent.allocate_equal_storage_independent_ranks(
        spectra, n_output_bases=1, input_dim=4, output_dim=4, shared_rank=2,
    )
    assert fit.allocation.private_ranks == expected.ranks_by_site
    assert fit.price.map_float_count == expected.independent_float_count
    assert fit.explained_shared_merit == 0.0
    # This is equality of the deployed coefficient maps, not merely of rank prices.
    for site, (solved_cross, merit, rank) in enumerate(zip(
        solved, merits, expected.ranks_by_site, strict=True,
    )):
        _, vectors = torch.linalg.eigh(merit)
        basis = vectors[:, -rank:].flip(1) if rank else vectors[:, :0]
        expected_map = solved_cross @ basis @ basis.T
        assert torch.allclose(
            hybrid.coefficient_maps(fit)[site], expected_map, atol=1e-11, rtol=1e-11,
        )


def test_zero_private_reduces_exactly_to_global_rrr():
    solved, merits = _problem()
    shared = hybrid.global_shared_basis(merits, 2)
    budget = 4 * (3 + 1) * 2
    fit = hybrid.fit_hierarchical_shared_private(
        solved, merits, shared, total_float_budget=budget,
    )
    assert fit.allocation.private_ranks == (0, 0, 0)
    assert fit.explained_private_merit == 0.0
    for site, coefficient in enumerate(hybrid.coefficient_maps(fit)):
        assert torch.allclose(coefficient, solved[site] @ shared @ shared.T, atol=1e-12)


def test_exact_literal_price_and_multiply_receipt():
    price = hybrid.hierarchical_price(3, 4, 2, (0, 1, 2))
    assert price.shared_input_float_count == 24
    assert price.shared_output_float_count == 8
    assert price.private_float_count == 24
    assert price.map_float_count == 56
    assert price.map_float_bytes == 224
    assert price.dense_multiplies_by_site == (16, 24, 32)
    assert price.dense_multiplies_per_uncovered_token == 72


def test_inexact_or_impossible_storage_is_rejected():
    spectra = tuple(torch.arange(4, 0, -1, dtype=torch.float64) for _ in range(3))
    with pytest.raises(ValueError, match="cannot be expressed"):
        hybrid.allocate_private_ranks(
            spectra, dimension=4, shared_rank=0, total_float_budget=33,
        )
    with pytest.raises(ValueError, match="exceeds"):
        hybrid.allocate_private_ranks(
            spectra, dimension=4, shared_rank=0, total_float_budget=8 * 13,
        )


def test_site_permutation_only_permutes_private_allocation():
    spectra = (
        torch.tensor([10., 9., 1., 0.], dtype=torch.float64),
        torch.tensor([8., 2., 1., 0.], dtype=torch.float64),
        torch.tensor([7., 6., 1., 0.], dtype=torch.float64),
    )
    first = hybrid.allocate_private_ranks(
        spectra, dimension=4, shared_rank=0, total_float_budget=8 * 5,
    )
    order = (2, 0, 1)
    second = hybrid.allocate_private_ranks(
        tuple(spectra[index] for index in order), dimension=4, shared_rank=0,
        total_float_budget=8 * 5,
    )
    restored = [0, 0, 0]
    for new, old in enumerate(order):
        restored[old] = second.private_ranks[new]
    assert tuple(restored) == first.private_ranks
    assert second.selected_residual_merit == pytest.approx(first.selected_residual_merit)


def test_merit_scale_preserves_projectors_and_rank_allocation():
    solved, merits = _problem(seed=12)
    shared = hybrid.global_shared_basis(merits, 1)
    budget = 4 * 4 + 8 * 4
    first = hybrid.fit_hierarchical_shared_private(
        solved, merits, shared, total_float_budget=budget,
    )
    scaled_merits = tuple(17.0 * value for value in merits)
    second = hybrid.fit_hierarchical_shared_private(
        solved, scaled_merits, shared, total_float_budget=budget,
    )
    assert first.allocation.private_ranks == second.allocation.private_ranks
    assert second.explained_private_merit == pytest.approx(17 * first.explained_private_merit)
    for left, right in zip(first.private_bases, second.private_bases, strict=True):
        assert torch.allclose(_projector(left), _projector(right), atol=1e-10)


def test_factor_receipt_hashes_without_program_authority():
    solved, merits = _problem()
    shared = hybrid.global_shared_basis(merits, 1)
    fit = hybrid.fit_hierarchical_shared_private(
        solved, merits, shared, total_float_budget=4 * 4 + 8 * 3,
    )
    receipt = hybrid.factor_hash_receipt(fit)
    assert receipt["serialized_program_authority"] is False
    assert len(receipt["sha256"]) == 64
    assert len(receipt["private_basis_sha256s"]) == 3


def test_frozen_bilin18_storage_grid_is_exact():
    value = hybrid.canonical_bilin18_storage_points()
    assert value["budgets"] == {
        "global_q512": 21_823_488,
        "typed_q512": 22_413_312,
        "independent_q512": 42_467_328,
    }
    assert value["private_rank_slots"]["global_q512"] == {
        "0": 9472, "64": 8288, "128": 7104, "256": 4736, "512": 0,
    }
    assert value["private_rank_slots"]["typed_q512"]["0"] == 9728
    assert value["private_rank_slots"]["independent_q512"]["0"] == 18432
    assert value["common_table_float_count"] == 224_736_768
    assert value["full_program_float_counts"] == {
        "global_q512": 246_560_256,
        "typed_q512": 247_150_080,
        "independent_q512": 267_204_096,
    }


def test_validation_rejects_nonorthogonal_shared_basis_and_indefinite_residual():
    solved, merits = _problem()
    bad = torch.ones((4, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="orthonormal"):
        hybrid.fit_hierarchical_shared_private(solved, merits, bad, total_float_budget=64)
    indefinite = list(merits)
    indefinite[0] = -torch.eye(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="indefinite"):
        hybrid.residual_eigensystems(tuple(indefinite), torch.empty((4, 0), dtype=torch.float64))
