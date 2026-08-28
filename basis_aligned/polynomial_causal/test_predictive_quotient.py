from __future__ import annotations

import pytest
import torch

import predictive_quotient as quotient


def test_balanced_spectrum_and_tail_certificate_match_known_diagonal_problem() -> None:
    covariance = torch.diag(torch.tensor([4.0, 1.0, 0.25], dtype=torch.float64))
    observability = torch.diag(torch.tensor([1.0, 9.0, 4.0], dtype=torch.float64))
    result = quotient.solve_predictive_quotient(covariance, observability)
    torch.testing.assert_close(
        result.eigenvalues, torch.tensor([9.0, 4.0, 1.0], dtype=torch.float64),
        rtol=1e-14, atol=1e-14,
    )
    assert result.support_rank == 3
    assert result.discarded_quadratic_response(0) == pytest.approx(14.0)
    assert result.discarded_quadratic_response(1) == pytest.approx(5.0)
    assert result.discarded_quadratic_response(2) == pytest.approx(1.0)
    assert result.rank_for_fraction(0.90) == 2
    expected = torch.diag(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(result.projector(1), expected, rtol=0, atol=1e-14)


def test_predictive_quotient_is_invariant_under_orthogonal_code_gauge() -> None:
    generator = torch.Generator().manual_seed(20260828)
    left = torch.randn(7, 7, generator=generator, dtype=torch.float64)
    right = torch.randn(7, 7, generator=generator, dtype=torch.float64)
    covariance = left @ left.T + 0.2 * torch.eye(7, dtype=torch.float64)
    observability = right @ right.T
    gauge, _ = torch.linalg.qr(
        torch.randn(7, 7, generator=generator, dtype=torch.float64),
    )
    original = quotient.solve_predictive_quotient(covariance, observability)
    moved = quotient.solve_predictive_quotient(
        gauge.T @ covariance @ gauge, gauge.T @ observability @ gauge,
    )
    torch.testing.assert_close(
        original.eigenvalues, moved.eigenvalues, rtol=2e-12, atol=2e-12,
    )
    for rank in (0, 1, 3, 7):
        assert original.discarded_quadratic_response(rank) == pytest.approx(
            moved.discarded_quadratic_response(rank), rel=2e-12, abs=2e-12,
        )
        torch.testing.assert_close(
            moved.projector(rank), gauge.T @ original.projector(rank) @ gauge,
            rtol=2e-11, atol=2e-11,
        )


def test_vjp_observability_and_empirical_covariance_recover_exact_objects() -> None:
    codes = torch.tensor([
        [-1.0, 0.0], [1.0, 0.0], [0.0, -2.0], [0.0, 2.0],
    ])
    covariance = quotient.covariance_from_codes(codes)
    centered = codes.double() - codes.double().mean(0)
    torch.testing.assert_close(covariance, centered.T @ centered / 3)

    gradients = torch.tensor([
        [[1.0, 0.0], [0.0, 2.0]],
        [[1.0, 0.0], [0.0, 2.0]],
    ])
    observability = quotient.observability_from_vjp_sketches(gradients)
    torch.testing.assert_close(
        observability, torch.diag(torch.tensor([0.5, 2.0], dtype=torch.float64)),
    )


def test_null_observability_and_invalid_inputs_fail_or_certify_exactly() -> None:
    covariance = torch.eye(4, dtype=torch.float64)
    observability = torch.diag(torch.tensor([3.0, 1.0, 0.0, 0.0]))
    result = quotient.solve_predictive_quotient(covariance, observability)
    assert result.rank_for_fraction(1.0) == 2
    assert result.discarded_quadratic_response(2) == 0
    deltas = torch.tensor([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])
    torch.testing.assert_close(
        result.quadratic_response(deltas), torch.tensor([3.0, 0.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        quotient.solve_predictive_quotient(
            covariance, torch.diag(torch.tensor([1.0, 1.0, 1.0, -0.1])),
        )
    with pytest.raises(ValueError, match="symmetric"):
        quotient.solve_predictive_quotient(
            covariance, torch.triu(torch.ones(4, 4, dtype=torch.float64)),
        )


def test_singular_covariance_keeps_supported_and_unobserved_null_spaces_distinct() -> None:
    covariance = torch.diag(torch.tensor([4.0, 1.0, 0.0, 0.0]))
    # The second supported direction and both unsupported directions all have zero
    # response.  The solver must nevertheless retain the complete covariance support.
    observability = torch.diag(torch.tensor([3.0, 0.0, 7.0, 5.0]))
    result = quotient.solve_predictive_quotient(covariance, observability)
    assert result.support_rank == 2
    torch.testing.assert_close(
        result.eigenvalues,
        torch.tensor([12.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        rtol=0, atol=1e-14,
    )
    torch.testing.assert_close(
        result.projector(2),
        torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)),
        rtol=0, atol=1e-14,
    )
