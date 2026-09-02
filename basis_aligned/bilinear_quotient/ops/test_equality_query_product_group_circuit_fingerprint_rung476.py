import torch

import equality_query_product_group_circuit_fingerprint_rung476 as subject


def test_group_partition_is_exact_and_controls_are_matched():
    selection = {}
    for site in subject.SITES:
        selection[site] = {
            "selected_indices": [1, 3],
            "amplitude_control_indices": [2, 4],
            "random_control_indices": [5, 6],
        }
    old = subject.HIDDEN
    try:
        subject.HIDDEN = 8
        groups = subject.make_groups(selection)
    finally:
        subject.HIDDEN = old
    for site in subject.SITES:
        assert groups["selected"][site] == [1, 3]
        assert groups["complement"][site] == [0, 2, 4, 5, 6, 7]
        assert len(groups["amplitude"][site]) == len(groups["selected"][site])
        assert len(groups["random"][site]) == len(groups["selected"][site])


def test_expected_count_formulas():
    assert subject.FORWARDS_PER_BATCH == 37
    assert subject.PATCH_CALLS_PER_BATCH == 36
    assert subject.EXPECTED_FORWARDS == 9250
    assert subject.EXPECTED_PATCH_CALLS == 9000


def test_frozen_kinds_are_semantic_controls_not_rank_arms():
    assert subject.KINDS == ("selected", "complement", "amplitude", "random")
    assert subject.HIDDEN == 4608
