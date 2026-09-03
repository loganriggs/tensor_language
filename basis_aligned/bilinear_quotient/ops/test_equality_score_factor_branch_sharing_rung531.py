#!/usr/bin/env python3

import torch

import equality_score_factor_branch_sharing_rung531 as rung531


def _populate_statistics(target_factors: torch.Tensor):
    statistics = rung531.empty_statistics()
    target_products = torch.stack([
        target_factors[2 * index] * target_factors[2 * index + 1]
        for index in range(len(rung531.TERMS))
    ])
    selected = torch.ones(target_factors.shape[1:], dtype=torch.bool)
    for segment in range(len(rung531.SEGMENTS)):
        rung531._add_gram_statistics(
            statistics, segment, 0, target_factors, target_factors,
            target_products, target_products, selected)
        permuted = torch.roll(target_factors, 17 + segment, dims=-1)
        permuted_products = torch.stack([
            permuted[2 * index] * permuted[2 * index + 1]
            for index in range(len(rung531.TERMS))
        ])
        rung531._add_gram_statistics(
            statistics, segment, 1, permuted, target_factors,
            permuted_products, target_products, selected)
        non_equality = torch.roll(target_factors, 31 + segment, dims=-1)
        non_equality_products = torch.stack([
            non_equality[2 * index] * non_equality[2 * index + 1]
            for index in range(len(rung531.TERMS))
        ])
        rung531._add_gram_statistics(
            statistics, segment, 2, non_equality, non_equality,
            non_equality_products, non_equality_products, selected)
    return statistics


def _independent_factors(seed=531, entries=257):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(8, 1, 1, entries, generator=generator, dtype=torch.float64)


def test_direct_both_factor_candidate_is_recovered_with_product_gauge():
    factors = _independent_factors()
    factors[2] = 2.0 * factors[0]
    factors[3] = -3.0 * factors[1]
    reports, both, one, gauge = rung531.analyze(_populate_statistics(factors))
    name = "L5H5->L7H3"
    assert name in both
    assert name not in one
    assert name in gauge
    assert reports[name]["selected_assignment"] == "direct"
    assert reports[name]["scale_product_relative_difference"] < 1e-12


def test_swapped_both_factor_candidate_is_recovered():
    factors = _independent_factors(seed=532)
    factors[4] = -4.0 * factors[1]
    factors[5] = 0.5 * factors[0]
    reports, both, _one, gauge = rung531.analyze(_populate_statistics(factors))
    name = "L5H5->L8H3"
    assert name in both
    assert name in gauge
    assert reports[name]["selected_assignment"] == "swapped"


def test_exactly_one_factor_candidate_does_not_claim_product_gauge_for_unknown_pair():
    factors = _independent_factors(seed=533)
    factors[2] = 1.75 * factors[0]
    reports, both, one, gauge = rung531.analyze(_populate_statistics(factors))
    name = "L5H5->L7H3"
    assert name not in both
    assert name in one
    assert name not in gauge
    assert reports[name]["exactly_one_factor_shared"]


def test_key_prefix_reversal_is_a_bijection_inside_every_causal_prefix():
    values = torch.arange(2 * 4 * 4, dtype=torch.float64).view(2, 4, 4)
    reversed_values = rung531._key_prefix_reverse(values)
    for batch in range(2):
        for query in range(4):
            observed = reversed_values[batch, query, :query + 1]
            expected = values[batch, query, :query + 1].flip(0)
            assert torch.equal(observed, expected)


def test_registered_shape_and_price_are_exact():
    assert len(rung531.PAIRS) == 12
    assert rung531.FORWARDS == 125
    assert tuple(rung531.SEGMENTS.values()) == ((0, 250), (250, 375), (375, 500))
    assert set(rung531.KNOWN_PRODUCT_PAIRS) <= set(rung531.PAIR_NAMES)
