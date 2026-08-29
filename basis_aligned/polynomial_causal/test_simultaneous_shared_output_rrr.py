import math

import pytest
import torch

from simultaneous_shared_output_rrr import (
    allocate_equal_storage_independent_ranks,
    canonical_price_receipt,
    fit_grouped_output_bases,
    fit_shared_output_basis,
    grouped_map_price,
    map_price,
    penalized_objective_from_statistics,
)


def _synthetic(seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    n_sites, n_rows, input_dim, output_dim, true_rank = 4, 80, 5, 7, 2
    raw_basis = torch.randn(output_dim, true_rank, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(raw_basis, mode="reduced")
    xs, ys, grams, crosses, y2s = [], [], [], [], []
    for _ in range(n_sites):
        x = torch.randn(n_rows, input_dim, generator=generator, dtype=torch.float64)
        a = torch.randn(input_dim, true_rank, generator=generator, dtype=torch.float64)
        y = x @ a @ basis.T
        xs.append(x)
        ys.append(y)
        grams.append(x.T @ x)
        crosses.append(x.T @ y)
        y2s.append(float(y.square().sum()))
    return xs, ys, grams, crosses, y2s, basis


def test_shared_basis_recovers_exact_common_output_subspace():
    xs, ys, grams, crosses, _, true_basis = _synthetic()
    fit = fit_shared_output_basis(grams, crosses, rank=2, ridge=1e-10)
    assert fit["basis_orthogonality_max_abs"] < 1e-12
    assert fit["projector_idempotence_max_abs"] < 1e-12
    assert torch.allclose(fit["projector"], true_basis @ true_basis.T, atol=1e-10, rtol=1e-10)
    for x, y, coefficient in zip(xs, ys, fit["coefficient_maps"], strict=True):
        assert torch.allclose(x @ coefficient, y, atol=2e-10, rtol=2e-10)


def test_top_eigenspace_beats_random_subspace_on_registered_objective():
    _, _, grams, crosses, y2s, _ = _synthetic()
    ridge = 1e-3
    fit = fit_shared_output_basis(grams, crosses, rank=2, ridge=ridge)
    optimum = penalized_objective_from_statistics(
        grams, crosses, y2s, fit["basis"], fit["input_maps"], ridge
    )

    generator = torch.Generator().manual_seed(91)
    random_basis, _ = torch.linalg.qr(
        torch.randn(7, 2, generator=generator, dtype=torch.float64), mode="reduced"
    )
    random_maps = []
    identity = torch.eye(5, dtype=torch.float64)
    for gram, cross in zip(grams, crosses, strict=True):
        random_maps.append(torch.linalg.solve(gram + ridge * identity, cross @ random_basis))
    random_objective = penalized_objective_from_statistics(
        grams, crosses, y2s, random_basis, random_maps, ridge
    )
    assert optimum < random_objective - 1.0


def test_full_output_rank_replays_independent_ridge_maps():
    _, _, grams, crosses, _, _ = _synthetic()
    ridge = 0.2
    fit = fit_shared_output_basis(grams, crosses, rank=7, ridge=ridge)
    identity = torch.eye(5, dtype=torch.float64)
    for gram, cross, coefficient in zip(
        grams, crosses, fit["coefficient_maps"], strict=True
    ):
        independent = torch.linalg.solve(gram + ridge * identity, cross)
        assert torch.allclose(coefficient, independent, atol=2e-12, rtol=2e-12)


def test_basis_rotation_is_a_gauge_not_a_different_program():
    _, _, grams, crosses, _, _ = _synthetic()
    ridge = 1e-3
    fit = fit_shared_output_basis(grams, crosses, rank=2, ridge=ridge)
    rotation, _ = torch.linalg.qr(
        torch.tensor([[1.0, 2.0], [-3.0, 0.5]], dtype=torch.float64)
    )
    rotated_basis = fit["basis"] @ rotation
    rotated_input_maps = [input_map @ rotation for input_map in fit["input_maps"]]
    for original, rotated in zip(
        fit["coefficient_maps"],
        [input_map @ rotated_basis.T for input_map in rotated_input_maps],
        strict=True,
    ):
        assert torch.allclose(original, rotated, atol=2e-12, rtol=2e-12)


def test_current_price_implication_is_exact():
    receipt = canonical_price_receipt()
    assert receipt["separate_float_count"] == 42_467_328
    assert receipt["shared_output_float_count"] == 21_823_488
    assert receipt["saved_float_count"] == 20_643_840
    assert receipt["separate_float_bytes"] == 169_869_312
    assert receipt["shared_output_float_bytes"] == 87_293_952
    assert math.isclose(receipt["saved_fraction"], 0.4861111111111111)
    assert receipt["multiplies_per_site"] == 1_179_648


def test_grouped_prices_interpolate_between_one_shared_and_independent():
    global_shared = grouped_map_price(36, 1, 1152, 1152, 512)
    attention_mlp = grouped_map_price(36, 2, 1152, 1152, 512)
    independent = grouped_map_price(36, 36, 1152, 1152, 512)
    assert global_shared.grouped_float_count == 21_823_488
    assert attention_mlp.grouped_float_count == 22_413_312
    assert math.isclose(attention_mlp.saved_fraction, 0.4722222222222222)
    assert independent.grouped_float_count == independent.separate_float_count
    assert independent.saved_float_count == 0


def test_equal_storage_allocator_uses_exact_global_budget_and_strongest_marginals():
    # Heterogeneous known answer: the best exact allocation is ranks (3, 1, 2),
    # not the balanced (2, 2, 2).  With n=3, g=1, d=p=4, q=3 the grouped
    # grammar costs 48 floats, exactly six independent rank slots.
    spectra = [
        torch.tensor([10.0, 9.0, 8.0, 0.1], dtype=torch.float64),
        torch.tensor([7.0, 0.6, 0.5, 0.4], dtype=torch.float64),
        torch.tensor([6.0, 5.0, 0.3, 0.2], dtype=torch.float64),
    ]
    allocation = allocate_equal_storage_independent_ranks(
        spectra,
        n_output_bases=1,
        input_dim=4,
        output_dim=4,
        shared_rank=3,
    )
    assert allocation.grouped_float_budget == 48
    assert allocation.independent_float_count == 48
    assert allocation.total_rank_slots == 6
    assert allocation.ranks_by_site == (3, 1, 2)
    assert math.isclose(allocation.selected_marginal_merit, 45.0)


@pytest.mark.parametrize(
    ("groups", "expected_slots"),
    [(1, 9472), (2, 9728)],
)
def test_rank512_bilin18_equal_storage_slot_counts_are_exact(groups, expected_slots):
    spectra = [torch.linspace(1152.0, 1.0, 1152, dtype=torch.float64) for _ in range(36)]
    allocation = allocate_equal_storage_independent_ranks(
        spectra,
        n_output_bases=groups,
        input_dim=1152,
        output_dim=1152,
        shared_rank=512,
    )
    assert allocation.total_rank_slots == expected_slots
    assert allocation.independent_float_count == allocation.grouped_float_budget
    assert max(allocation.ranks_by_site) - min(allocation.ranks_by_site) <= 1


def test_equal_storage_allocator_rejects_nonprefix_spectra_and_inexact_budget():
    with pytest.raises(ValueError, match="nonincreasing"):
        allocate_equal_storage_independent_ranks(
            [torch.tensor([2.0, 3.0], dtype=torch.float64)],
            n_output_bases=1,
            input_dim=2,
            output_dim=2,
            shared_rank=1,
        )
    with pytest.raises(ValueError, match="cannot be matched"):
        allocate_equal_storage_independent_ranks(
            [torch.tensor([2.0, 1.0], dtype=torch.float64)] * 2,
            n_output_bases=1,
            input_dim=2,
            output_dim=3,
            shared_rank=1,
        )


def test_one_group_exactly_replays_global_shared_fit():
    _, _, grams, crosses, _, _ = _synthetic()
    shared = fit_shared_output_basis(grams, crosses, rank=2, ridge=1e-3)
    grouped = fit_grouped_output_bases(
        grams, crosses, groups=["all"] * len(grams), rank=2, ridge=1e-3
    )
    assert math.isclose(
        grouped["explained_penalized_fit"],
        shared["explained_penalized_fit"],
        rel_tol=1e-12,
    )
    for expected, observed in zip(
        shared["coefficient_maps"], grouped["coefficient_maps"], strict=True
    ):
        assert torch.allclose(expected, observed, atol=2e-12, rtol=2e-12)


def test_one_group_per_site_replays_independent_reduced_rank_fits():
    _, _, grams, crosses, _, _ = _synthetic()
    grouped = fit_grouped_output_bases(
        grams, crosses, groups=list(range(len(grams))), rank=2, ridge=1e-3
    )
    for index, observed in enumerate(grouped["coefficient_maps"]):
        independent = fit_shared_output_basis(
            [grams[index]], [crosses[index]], rank=2, ridge=1e-3
        )
        assert torch.allclose(
            observed, independent["coefficient_maps"][0], atol=2e-12, rtol=2e-12
        )


def test_grouped_fit_rejects_missing_and_unhashable_labels():
    _, _, grams, crosses, _, _ = _synthetic()
    with pytest.raises(ValueError, match="one label per site"):
        fit_grouped_output_bases(grams, crosses, groups=["short"], rank=2, ridge=1e-3)
    with pytest.raises(ValueError, match="hashable"):
        fit_grouped_output_bases(
            grams, crosses, groups=[[0]] * len(grams), rank=2, ridge=1e-3
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_sites": 0, "input_dim": 4, "output_dim": 4, "rank": 2},
        {"n_sites": 2, "input_dim": 4, "output_dim": 4, "rank": 5},
    ],
)
def test_price_rejects_invalid_grammar(kwargs):
    with pytest.raises(ValueError):
        map_price(**kwargs)


def test_statistics_reject_nonsymmetric_and_nonpositive_inputs():
    gram = torch.eye(3, dtype=torch.float64)
    cross = torch.ones(3, 2, dtype=torch.float64)
    bad = gram.clone()
    bad[0, 1] = 0.2
    with pytest.raises(ValueError, match="not symmetric"):
        fit_shared_output_basis([bad], [cross], rank=1, ridge=1e-3)
    with pytest.raises(ValueError, match="strictly positive"):
        fit_shared_output_basis([gram], [cross], rank=1, ridge=0.0)
