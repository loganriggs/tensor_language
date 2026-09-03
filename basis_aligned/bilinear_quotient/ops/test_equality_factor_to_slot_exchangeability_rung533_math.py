#!/usr/bin/env python3

import torch

import equality_factor_to_slot_exchangeability_rung533_math as rung533


def test_all_four_parent_mappings_work_but_two_matched_controls_are_missing():
    table = rung533.analyze_parent()
    assert set(table) == set(rung533.SCALES)
    assert all(row["base_contexts_passing"] == 8 for row in table.values())
    assert table["source_second_to_target_first"]["beats_matched_control_by_0p15_contexts"] == 8
    assert table["source_first_to_target_second"]["beats_matched_control_by_0p15_contexts"] == 8
    assert table["source_first_to_target_first"]["matched_key_control_present_in_parent"] is False
    assert table["source_second_to_target_second"]["matched_key_control_present_in_parent"] is False


def test_each_mapping_has_a_scale_matched_nontrivial_key_control():
    generator = torch.Generator().manual_seed(7)
    tensors = [torch.randn(2, 9, 9, generator=generator) for _ in range(4)]
    for mapping in rung533.SCALES:
        ordinary = rung533.substitution(mapping, *tensors)
        permuted = rung533.substitution(mapping, *tensors, permuted=True)
        assert ordinary.shape == permuted.shape == (2, 9, 9)
        assert not torch.equal(ordinary, permuted)


def test_key_permutation_reverses_only_each_valid_prefix():
    value = torch.arange(25).view(1, 5, 5).float()
    observed = rung533.key_prefix_reverse(value)
    for query in range(5):
        assert torch.equal(observed[0, query, :query + 1], value[0, query, :query + 1].flip(0))
        assert torch.equal(observed[0, query, query + 1:], value[0, query, query + 1:])
