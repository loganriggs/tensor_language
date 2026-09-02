import torch

import equality_query_subtractive_factorial_rung474 as subject


def test_subset_order_and_indices():
    assert subject.SUBSET_NAMES == (
        "m8", "m9", "m12", "m8+m9", "m8+m12", "m9+m12", "m8+m9+m12",
    )
    assert subject.SINGLE_INDICES == (0, 1, 2)
    assert subject.PAIR_INDICES == (3, 4, 5)
    assert subject.UNION_INDEX == 6


def test_frozen_subtraction_reproduces_baseline_for_intact_source():
    source = torch.tensor([1., -2., 4.])
    absent = torch.tensor([-.5, 1., 3.])
    delta = source.float() - absent.float()
    reconstructed = (source.float() - delta).to(source.dtype)
    assert torch.equal(reconstructed, absent)


def test_expected_call_formulas():
    assert subject.FORWARDS_PER_BATCH == 35
    assert subject.EXPECTED_FORWARDS == subject.EXPECTED_BATCHES * 35
    assert subject.EXPECTED_PATCH_CALLS_PER_BATCH == 54
