from __future__ import annotations

import pytest
import torch

import cut_cross_bootstrap_stability as stability


def test_repeated_exact_rank_two_matrix_has_stable_exact_cross() -> None:
    generator = torch.Generator().manual_seed(20260828)
    matrix = torch.randn((5, 2), generator=generator, dtype=torch.float64) @ torch.randn(
        (2, 5), generator=generator, dtype=torch.float64
    )
    draws = matrix.unsqueeze(0).repeat(12, 1, 1)
    result = stability.pivot_stability(draws, point_matrix=matrix, rank=2)
    assert result["point_pivot_selection_frequency"] == 1.0
    assert result["unique_winning_pivots"] == 1
    assert result["frozen_point_pivot_cross_nre"]["q95"] < 1e-12


def test_perturbed_draws_report_literal_selection_frequency() -> None:
    first = torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64))
    second = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    draws = torch.stack((first, first, second))
    result = stability.pivot_stability(draws, point_matrix=first, rank=1)
    assert result["point_pivot_selection_frequency"] == pytest.approx(2 / 3)
    assert result["unique_winning_pivots"] == 2


def test_malformed_pivot_request_fails_closed() -> None:
    with pytest.raises(ValueError):
        stability.pivot_stability(
            torch.ones((2, 3, 4), dtype=torch.float64),
            point_matrix=torch.ones((3, 4), dtype=torch.float64), rank=1,
        )
