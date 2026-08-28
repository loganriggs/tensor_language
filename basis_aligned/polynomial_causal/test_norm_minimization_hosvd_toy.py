import math

import torch

from norm_minimization_hosvd_toy import (
    cp_scalar_gauge_toy,
    gl_edge_gauge_toy,
    minimum_norm_edge_factors,
)


def test_cp_balance_minimizes_displayed_norm_but_cannot_move_hosvd() -> None:
    result = cp_scalar_gauge_toy()
    assert result.parameter_norm_after < 1e-4 * result.parameter_norm_before
    assert result.folded_tensor_relative_drift < 1e-11
    assert result.hosvd_spectrum_relative_drift < 1e-11
    assert result.balanced_log_defect < 1e-12


def test_gl_balance_reaches_exact_nuclear_norm_minimum() -> None:
    result = gl_edge_gauge_toy()
    assert result.parameter_norm_after < 1e-4 * result.parameter_norm_before
    assert math.isclose(
        result.parameter_norm_after, result.exact_minimum_norm, rel_tol=1e-12, abs_tol=1e-12
    )
    assert result.contraction_relative_drift < 1e-11
    assert result.contraction_spectrum_relative_drift < 1e-11
    assert result.balanced_gram_relative_defect < 1e-11
    assert result.bond_rank_before == result.bond_rank_after


def test_gl_balance_removes_dormant_bond_rank_in_orbit_closure() -> None:
    result = gl_edge_gauge_toy(dormant=True)
    assert result.bond_rank_before == 5
    assert result.bond_rank_after == 3
    assert result.parameter_norm_after < result.parameter_norm_before
    assert result.contraction_relative_drift < 1e-12
    assert result.contraction_spectrum_relative_drift < 1e-12


def test_orthogonal_gauge_is_the_residual_minimum_norm_ambiguity() -> None:
    generator = torch.Generator().manual_seed(7)
    left = torch.randn(9, 4, generator=generator, dtype=torch.float64)
    right = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    balanced_left, balanced_right, _ = minimum_norm_edge_factors(left, right)
    q, _ = torch.linalg.qr(
        torch.randn(4, 4, generator=generator, dtype=torch.float64)
    )
    rotated_left = balanced_left @ q
    rotated_right = q.T @ balanced_right
    assert torch.allclose(rotated_left @ rotated_right, balanced_left @ balanced_right)
    assert torch.allclose(
        rotated_left.square().sum() + rotated_right.square().sum(),
        balanced_left.square().sum() + balanced_right.square().sum(),
    )
