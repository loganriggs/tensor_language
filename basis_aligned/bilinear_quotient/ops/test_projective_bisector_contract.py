import pytest
import torch

from projective_bisector_contract import ProjectiveBisectorError, projective_minimax_center


def test_orthogonal_axes_have_half_overlap_bisector():
    result = projective_minimax_center(torch, torch.tensor([1., 0.]), torch.tensor([0., 1.]))
    assert result["exact_worst_squared_overlap"] == pytest.approx(0.5)
    assert result["observed_worst_squared_overlap"] == pytest.approx(0.5)
    assert result["closure_abs_error"] < 1e-6


def test_projective_center_is_sign_gauge_invariant():
    left, right = torch.tensor([1., 2., 0.]), torch.tensor([2., 1., 1.])
    positive = projective_minimax_center(torch, left, right)
    negative = projective_minimax_center(torch, left, -right)
    assert positive["exact_worst_squared_overlap"] == pytest.approx(
        negative["exact_worst_squared_overlap"])
    assert torch.allclose(positive["center"], negative["center"])


def test_identical_lines_have_unit_overlap():
    result = projective_minimax_center(torch, torch.tensor([3., 4.]), torch.tensor([6., 8.]))
    assert result["exact_worst_squared_overlap"] == pytest.approx(1.0)
    assert result["closure_abs_error"] < 1e-6


def test_nonfinite_or_zero_axis_fails_closed():
    with pytest.raises(ProjectiveBisectorError):
        projective_minimax_center(torch, torch.zeros(2), torch.ones(2))
    with pytest.raises(ProjectiveBisectorError):
        projective_minimax_center(torch, torch.tensor([float("nan"), 0.]), torch.ones(2))
