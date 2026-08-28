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

    four_dimensional = gradients[:, :, None, :].expand(-1, -1, 3, -1)
    torch.testing.assert_close(
        quotient.observability_from_vjp_sketches(four_dimensional), observability,
    )


def test_categorical_probe_ids_are_reproducible_and_obey_degenerate_distribution() -> None:
    logits = torch.full((2, 5, 4), -100.0)
    logits[..., 2] = 100.0
    first = quotient.categorical_fisher_probe_ids(
        logits, (11, 29, 47), score_start=1, score_stop=5,
    )
    second = quotient.categorical_fisher_probe_ids(
        logits, (11, 29, 47), score_start=1, score_stop=5,
    )
    assert first.shape == (3, 2, 4)
    assert first.dtype == torch.long and first.device.type == "cpu"
    assert torch.equal(first, second)
    assert torch.equal(first, torch.full_like(first, 2))


def test_fisher_vjp_matches_exact_categorical_score_gradient() -> None:
    codes = torch.tensor([
        [[0.2, -0.4], [0.7, 0.1], [-0.3, 0.8]],
    ], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([
        [0.4, -0.2, 0.7],
        [-0.5, 0.9, 0.1],
    ], dtype=torch.float64)
    logits = codes @ weight
    targets = torch.tensor([
        [[0, 1]],
        [[2, 0]],
    ], dtype=torch.long)
    sketches = quotient.fisher_vjp_sketches(
        codes, logits, targets, score_start=1, score_stop=3,
    )
    probabilities = torch.softmax(logits.detach()[:, 1:3].float(), dim=-1).double()
    expected = []
    for probe in targets:
        rows = []
        for position in range(2):
            y = int(probe[0, position])
            rows.append(weight[:, y] - weight @ probabilities[0, position])
        expected.append(torch.stack(rows).unsqueeze(0))
    torch.testing.assert_close(
        sketches, torch.stack(expected), rtol=2e-7, atol=2e-7,
    )


def test_exhaustive_uniform_probes_recover_exact_softmax_fisher_and_merge() -> None:
    codes = torch.zeros(1, 1, 2, dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([
        [0.4, -0.2, 0.7],
        [-0.5, 0.9, 0.1],
    ], dtype=torch.float64)
    logits = codes @ weight
    targets = torch.arange(3, dtype=torch.long).view(3, 1, 1)
    sketches = quotient.fisher_vjp_sketches(
        codes, logits, targets, score_start=0, score_stop=1,
    )
    observed = quotient.observability_from_vjp_sketches(sketches)
    fisher = torch.diag(torch.full((3,), 1 / 3, dtype=torch.float64)) - torch.full(
        (3, 3), 1 / 9, dtype=torch.float64,
    )
    torch.testing.assert_close(observed, weight @ fisher @ weight.T, rtol=2e-7, atol=2e-7)

    left_sum, left_count = quotient.vjp_outer_product_sum(sketches[:1])
    right_sum, right_count = quotient.vjp_outer_product_sum(sketches[1:])
    torch.testing.assert_close(
        (left_sum + right_sum) / (left_count + right_count), observed,
        rtol=0, atol=1e-14,
    )


def test_fisher_vjp_fails_when_code_is_not_on_the_logit_graph() -> None:
    codes = torch.randn(1, 3, 2, requires_grad=True)
    unrelated = torch.randn(1, 3, 4, requires_grad=True)
    targets = torch.zeros(1, 1, 2, dtype=torch.long)
    with pytest.raises(RuntimeError, match="not have been used"):
        quotient.fisher_vjp_sketches(
            codes, unrelated, targets, score_start=1, score_stop=3,
        )

    posthoc_codes = torch.randn(1, 192, 2, requires_grad=True)
    posthoc_logits = posthoc_codes @ torch.randn(2, 4)
    posthoc_targets = torch.zeros(1, 1, 192, dtype=torch.long)
    with pytest.raises(ValueError, match="outside the code/logit trajectory"):
        # The production/default contract requires the complete 256-position graph;
        # an already sliced 192-position alias cannot impersonate it.
        quotient.fisher_vjp_sketches(
            posthoc_codes, posthoc_logits, posthoc_targets,
        )


def test_causal_vjp_sums_future_fisher_and_independent_probe_cross_terms_cancel() -> None:
    codes = torch.zeros(1, 2, 1, dtype=torch.float64, requires_grad=True)
    z0, z1 = codes[:, 0, 0], codes[:, 1, 0]
    first = torch.stack((2.0 * z0, torch.zeros_like(z0)), dim=-1)
    second = torch.stack((3.0 * z0 + 5.0 * z1, torch.zeros_like(z0)), dim=-1)
    logits = torch.stack((first, second), dim=1)
    # Exhaust all independent uniform categorical target pairs.
    targets = torch.tensor([
        [[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]],
    ], dtype=torch.long)
    sketches = quotient.fisher_vjp_sketches(
        codes, logits, targets, score_start=0, score_stop=2,
    )
    position_energy = sketches.square().mean(dim=0)[0, :, 0]
    # z0 affects both outputs: .25 * (2^2 + 3^2). z1 affects only output 1.
    torch.testing.assert_close(
        position_energy, torch.tensor([3.25, 6.25], dtype=torch.float64),
        rtol=2e-7, atol=2e-7,
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

    accepted_roundoff = quotient.solve_predictive_quotient(
        torch.eye(2, dtype=torch.float64),
        torch.diag(torch.tensor([-5e-11, 1.0], dtype=torch.float64)),
        psd_rtol=1e-10,
    )
    assert float(accepted_roundoff.quadratic_response(
        torch.tensor([1.0, 0.0], dtype=torch.float64)
    )) >= 0.0
    assert float(torch.linalg.eigvalsh(accepted_roundoff.observability)[0]) >= 0.0


def test_rank_gap_rule_and_split_stability_use_whitened_subspaces() -> None:
    covariance = torch.diag(torch.tensor([4.0, 1.0, 0.25], dtype=torch.float64))
    left = quotient.solve_predictive_quotient(
        covariance, torch.diag(torch.tensor([20.0, 16.0, 16.0])),
    )
    # Balanced responses are [80, 16, 4], so rank 2 retains 96% with gap 4.
    assert quotient.selected_rank_with_gap(left) == 2
    no_gap = quotient.solve_predictive_quotient(
        covariance, torch.diag(torch.tensor([20.0, 16.0, 60.0])),
    )
    assert quotient.selected_rank_with_gap(no_gap) is None

    right = quotient.solve_predictive_quotient(
        covariance, torch.diag(torch.tensor([19.0, 17.0, 16.0])),
    )
    comparison = quotient.compare_predictive_quotients(
        left, right, comparison_rank=2,
    )
    assert comparison.left_fraction_rank == 2
    assert comparison.right_fraction_rank == 2
    assert comparison.relative_trace_difference == pytest.approx(3 / 98.5)
    assert comparison.normalized_chordal_distance == pytest.approx(0.0, abs=1e-14)
