#!/usr/bin/env python3

import pytest
import torch

import equality_score_factor_branch_sharing_rung531_math as math531


def _factors():
    first = torch.tensor([1.0, -2.0, 3.0, 0.5, -1.5])
    second = torch.tensor([-0.5, 4.0, 1.0, -3.0, 2.5])
    return first, second


def test_direct_scalar_gauges_reconstruct_both_branches_and_product():
    first, second = _factors()
    selected = math531.choose_assignment(first, second, 2.0 * first, -3.0 * second)
    assert selected["selected"] == "direct"
    report = selected["reports"]["direct"]
    assert report["target_first_scale"] == pytest.approx(2.0)
    assert report["target_second_scale"] == pytest.approx(-3.0)
    assert report["branch_scale_product"] == pytest.approx(-6.0)
    assert report["independent_product_scale"] == pytest.approx(-6.0)
    assert report["first"]["relative_rmse"] == pytest.approx(0.0)
    assert report["second"]["relative_rmse"] == pytest.approx(0.0)
    assert report["product"]["relative_rmse"] == pytest.approx(0.0)


def test_swapped_assignment_is_selected_and_frozen_scales_transfer():
    first, second = _factors()
    discovery = math531.choose_assignment(first, second, -4.0 * second, 0.5 * first)
    assert discovery["selected"] == "swapped"
    fit = discovery["reports"]["swapped"]
    heldout = math531.evaluate_frozen_assignment(
        1.5 * first,
        -2.0 * second,
        7.0 * second,
        0.60 * first,
        assignment="swapped",
        target_first_scale=fit["target_first_scale"],
        target_second_scale=fit["target_second_scale"],
    )
    # The held-out source gauges changed, so frozen scalar transfer must report the mismatch.
    assert heldout["first"]["relative_rmse"] > 0.0
    assert heldout["second"]["relative_rmse"] > 0.0


def test_exact_tie_prefers_direct():
    first = torch.tensor([1.0, -2.0, 3.0])
    selected = math531.choose_assignment(first, first, 2.0 * first, -3.0 * first)
    assert selected["selected"] == "direct"


def test_shape_zero_and_nonfinite_inputs_fail_closed():
    first, second = _factors()
    with pytest.raises(ValueError):
        math531.fit_scalar(first, second[:-1])
    with pytest.raises(ValueError):
        math531.fit_scalar(torch.zeros_like(first), second)
    bad = first.clone()
    bad[0] = float("nan")
    with pytest.raises(ValueError):
        math531.fit_scalar(bad, second)
