from __future__ import annotations

import pytest
import torch

import cut_cross_interpolation_diagnostic as diagnostic


def test_exact_rank_two_matrix_is_recovered_by_rank_two_cross() -> None:
    generator = torch.Generator().manual_seed(20260828)
    left = torch.randn((7, 2), generator=generator, dtype=torch.float64)
    right = torch.randn((2, 7), generator=generator, dtype=torch.float64)
    matrix = left @ right
    result = diagnostic.maximum_volume_cross(matrix, 2)
    assert result["nre"] < 1e-12
    assert result["max_abs_error"] < 1e-12


def test_anchored_inner_removes_additive_row_and_column_terms() -> None:
    row = torch.arange(8, dtype=torch.float64)[:, None]
    column = torch.arange(8, dtype=torch.float64)[None, :]
    assert torch.equal(diagnostic.anchored_inner((row + column).flatten()), torch.zeros(
        (7, 7), dtype=torch.float64
    ))


def test_best_rank_residual_reports_diffuse_support() -> None:
    matrix = torch.eye(4, dtype=torch.float64)
    result = diagnostic.best_rank_diagnostic(matrix, 1)
    assert result["effective_residual_support_cells"] == pytest.approx(3.0)
    assert 0.0 < result["nre"] < 1.0


def test_invalid_requests_fail_closed() -> None:
    with pytest.raises(ValueError):
        diagnostic.maximum_volume_cross(torch.ones((2, 2), dtype=torch.float32), 1)
    with pytest.raises(ValueError):
        diagnostic.anchored_inner(torch.ones(63, dtype=torch.float64))
