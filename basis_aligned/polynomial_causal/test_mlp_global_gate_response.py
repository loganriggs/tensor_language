from __future__ import annotations

import torch
import pytest

import mlp_global_gate_response as gate


def test_trajectory_complete_contraction_equals_shared_alpha_autograd() -> None:
    generator = torch.Generator().manual_seed(17)
    contexts, positions, width, hidden, probes = 2, 4, 3, 5, 3
    state = torch.randn(contexts, positions, width, generator=generator, dtype=torch.float64)
    left = torch.randn(hidden, width, generator=generator, dtype=torch.float64)
    right = torch.randn(hidden, width, generator=generator, dtype=torch.float64)
    down = torch.randn(width, hidden, generator=generator, dtype=torch.float64)
    gradients = torch.randn(
        probes, contexts, positions, width, generator=generator, dtype=torch.float64,
    )
    products = (state @ left.T) * (state @ right.T)
    expected = gate.trajectory_complete_response(products, gradients, down)
    assert expected.shape == (contexts, probes, hidden)
    separated = torch.empty_like(expected)
    for probe in range(probes):
        for context in range(contexts):
            alpha = torch.ones(hidden, dtype=torch.float64, requires_grad=True)
            write = (products[context] * alpha) @ down.T
            score = (write * gradients[probe, context]).sum()
            separated[context, probe] = torch.autograd.grad(score, alpha)[0]
    assert torch.allclose(expected, separated, atol=1e-12, rtol=1e-12)
    local_only = torch.einsum(
        "cn,pco,on->cpn", products[:, 0], gradients[:, :, 0], down,
    )
    assert not torch.allclose(expected, local_only)


def test_contraction_flows_into_selector_without_context_probe_axis_swap() -> None:
    generator = torch.Generator().manual_seed(19)
    products = torch.randn(3, 4, 7, generator=generator, dtype=torch.float64)
    gradients = torch.randn(5, 3, 4, 2, generator=generator, dtype=torch.float64)
    down = torch.randn(2, 7, generator=generator, dtype=torch.float64)
    first = gate.trajectory_complete_response(products, gradients, down)
    second = first + 1e-6 * torch.randn(
        first.shape, generator=generator, dtype=torch.float64,
    )
    report = gate.paired_selector_report(
        first, second, budgets=(2,), target_rank=2, random_seed=20260828,
    )
    assert first.shape == (3, 5, 7)
    assert report["shape"] == [3, 5, 7]


def test_response_is_invariant_to_gate_scale_gauge_and_equivariant_to_permutation() -> None:
    generator = torch.Generator().manual_seed(23)
    state = torch.randn(2, 3, 4, generator=generator, dtype=torch.float64)
    left = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    right = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    down = torch.randn(4, 6, generator=generator, dtype=torch.float64)
    gradients = torch.randn(2, 2, 3, 4, generator=generator, dtype=torch.float64)
    products = (state @ left.T) * (state @ right.T)
    base = gate.trajectory_complete_response(products, gradients, down)
    scales = torch.linspace(0.5, 2.0, 6, dtype=torch.float64)
    gauged_products = (state @ (left * scales[:, None]).T) * (state @ right.T)
    gauged = gate.trajectory_complete_response(gauged_products, gradients, down / scales)
    assert torch.allclose(base, gauged, atol=1e-11, rtol=1e-11)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    permuted = gate.trajectory_complete_response(
        products[:, :, permutation], gradients, down[:, permutation],
    )
    assert torch.allclose(permuted, base[:, :, permutation])


def test_ridge_selector_finds_stable_response_span() -> None:
    generator = torch.Generator().manual_seed(31)
    core = torch.randn(4, 5, 3, generator=generator, dtype=torch.float64)
    mixing = torch.tensor([
        [1.0, 0.0, 0.0, 0.7, 0.0, 0.2, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.6, 0.0, 0.2, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.2],
    ], dtype=torch.float64)
    first = core @ mixing + 1e-4 * torch.randn(
        4, 5, 8, generator=generator, dtype=torch.float64,
    )
    second = first + 1e-4 * torch.randn(first.shape, generator=generator, dtype=torch.float64)
    report = gate.paired_selector_report(
        first, second, budgets=(3, 5), target_rank=3, random_seed=20260828,
    )
    assert report["status"] == "paired_gate_response_selector_complete"
    assert report["rows"]["3"]["ridge"]["jaccard"] == 1.0
    assert report["rows"]["3"]["ridge"]["first_to_second_capture"] > 0.999999
    assert report["rows"]["3"]["ridge"]["first_to_second_cross_fit_css_relative_error"] < 1e-3
    assert report["rows"]["3"]["ridge"]["first_to_second_all_on_relative_error"] < 1e-3
    assert "finite-removal" in report["claim_boundary"]


def test_ridge_selector_uses_positive_tail_below_numerical_rank_tolerance() -> None:
    response = torch.diag(torch.tensor(
        [1.0, 1e-16, 0.0, 0.0], dtype=torch.float64,
    )).reshape(1, 4, 4)
    scores = gate.ridge_leverage_scores(response, 1)
    assert torch.allclose(
        scores, torch.tensor([1.0, 0.5, 0.0, 0.0], dtype=torch.float64),
        atol=1e-14, rtol=1e-14,
    )
    for scale in (1e-6, 1e6):
        assert torch.allclose(
            gate.ridge_leverage_scores(response * scale, 1), scores,
            atol=1e-14, rtol=1e-14,
        )


def test_selector_validation_is_fail_closed() -> None:
    response = torch.ones(2, 3, 4)
    try:
        gate.trajectory_complete_response(response, torch.ones(1, 2, 2, 3), torch.ones(3, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("position mismatch was accepted")
    try:
        gate.paired_selector_report(
            response, torch.ones(2, 2, 4), budgets=(2,), target_rank=1, random_seed=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unpaired response halves were accepted")


def test_cross_fit_css_does_not_refit_the_evaluation_half() -> None:
    fit = torch.tensor([[
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]], dtype=torch.float64)
    evaluate = torch.tensor([[
        [1.0, 0.0, -1.0],
        [0.0, 1.0, 2.0],
    ]], dtype=torch.float64)
    assert gate.projection_capture(evaluate, (0, 1)) == 1.0
    transferred = gate.cross_fit_css_relative_error(fit, evaluate, (0, 1))
    assert transferred > 0.8


def test_candidate_path_moves_omitted_gates_toward_zero() -> None:
    scale = gate.candidate_path_scale(
        5, (1, 3), torch.tensor([1.0, 0.5], dtype=torch.float64), 0.1,
    )
    torch.testing.assert_close(
        scale, torch.tensor([0.9, 1.0, 0.9, 0.95, 0.9], dtype=torch.float64),
    )
    endpoint = gate.candidate_path_scale(
        5, (1, 3), torch.tensor([1.0, 0.5], dtype=torch.float64), 1.0,
    )
    torch.testing.assert_close(
        endpoint, torch.tensor([0.0, 1.0, 0.0, 0.5, 0.0], dtype=torch.float64),
    )


def test_categorical_fisher_quadratic_predicts_small_kl() -> None:
    logits = torch.tensor([0.3, -0.7, 1.1, 0.2], dtype=torch.float64)
    direction = torch.tensor([0.5, -0.2, 0.1, -0.4], dtype=torch.float64)
    probability = torch.softmax(logits, dim=0)
    centered = direction - torch.dot(probability, direction)
    fisher = torch.dot(probability, centered.square())
    epsilon = 1e-4
    changed = torch.softmax(logits + epsilon * direction, dim=0)
    kl = torch.dot(probability, torch.log(probability / changed))
    predicted = 0.5 * epsilon**2 * fisher
    assert abs(float(kl - predicted)) / float(predicted) < 2e-4


def test_regularized_solver_is_fixed_float64_and_rejects_ill_conditioning() -> None:
    design = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    target = torch.tensor([1.0, -1.0, 0.5])
    solution, receipt = gate.regularized_svd_solution(design, target)
    assert solution.dtype == torch.float64
    assert receipt["relative_singular_cutoff"] == gate.SVD_RELATIVE_CUTOFF
    assert receipt["relative_tikhonov_ridge"] == gate.TIKHONOV_RELATIVE_RIDGE
    with pytest.raises(ValueError, match="condition number"):
        gate.regularized_svd_solution(
            torch.tensor([[1.0, 1.0], [0.0, 1e-8]], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
        )


def test_factor_product_canonicalization_is_scale_sign_gauge_invariant() -> None:
    generator = torch.Generator().manual_seed(913)
    products = torch.randn(3, 5, 7, generator=generator, dtype=torch.float64)
    down = torch.randn(11, 7, generator=generator, dtype=torch.float64)
    first_h, first_d, first_receipt = gate.canonicalize_factor_product_gates(
        products, down,
    )
    gauge = torch.tensor([2.0, -4.0, 0.5, -0.25, 8.0, -2.0, 0.125])
    second_h, second_d, second_receipt = gate.canonicalize_factor_product_gates(
        products * gauge, down / gauge,
    )
    assert torch.equal(first_h, second_h)
    assert torch.equal(first_d, second_d)
    assert first_receipt == second_receipt
    pivots = torch.tensor(first_receipt["pivot_indices"])
    assert bool((first_d.gather(0, pivots[None, :]).squeeze(0) > 0).all())


def test_canonical_factor_derangement_is_gauge_invariant_and_permutation_equivariant() -> None:
    generator = torch.Generator().manual_seed(8128)
    products = torch.randn(2, 4, 7, generator=generator, dtype=torch.float64)
    down = torch.randn(5, 7, generator=generator, dtype=torch.float64)
    gauge = torch.tensor([-4.0, 0.5, 2.0, -0.25, 8.0, 0.125, -2.0])
    baseline = gate.canonical_factor_product_derangement(products, down, 2026082806)
    replay = gate.canonical_factor_product_derangement(
        products * gauge, down / gauge, 2026082806,
    )
    assert baseline == replay
    assert sorted(baseline) == list(range(7))
    assert all(source != target for source, target in enumerate(baseline))

    relabel = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    relabeled = gate.canonical_factor_product_derangement(
        products[:, :, relabel], down[:, relabel], 2026082806,
    )
    inverse = torch.empty_like(relabel)
    inverse[relabel] = torch.arange(len(relabel))
    expected = tuple(int(inverse[baseline[int(relabel[index])]]) for index in range(7))
    assert relabeled == expected
