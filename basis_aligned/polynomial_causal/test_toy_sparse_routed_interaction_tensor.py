from __future__ import annotations

import math

import torch

import toy_sparse_routed_interaction_tensor as toy


def test_fold_gauges_scale_interaction_and_router_controls() -> None:
    result = toy.algebraic_checks()
    assert result["fold_exact"]
    assert result["permutation_rescaling_gauge_exact"]
    assert result["input_leg_swap_exact"]
    assert math.isclose(result["scaled_cosine"], 1.0, abs_tol=1e-12)
    assert result["scaled_relative_error"] > 1.0
    assert math.isclose(result["optimal_scale"], 0.4, abs_tol=1e-12)
    assert result["scale_corrected_error"] < 1e-20
    assert result["mobius_recovery_error"] < 1e-12
    assert result["null_interaction_norm"] < 1e-12
    assert math.isclose(result["identical_bank_similarity"], 1.0, abs_tol=1e-12)
    assert result["wrong_router_relative_mse"] > 0.1


def test_gaussian_metric_matches_monte_carlo() -> None:
    generator = torch.Generator().manual_seed(13)
    target = torch.randn(3, 4, 4, generator=generator, dtype=torch.float64)
    candidate = torch.randn(3, 4, 4, generator=generator, dtype=torch.float64)
    target, candidate = toy.symmetrize(target), toy.symmetrize(candidate)
    x = torch.randn(500_000, 4, generator=generator, dtype=torch.float64)
    exact = toy.gaussian_inner(target, candidate)
    samples = (
        toy.tensor_forward(target, x) * toy.tensor_forward(candidate, x)
    ).sum(-1)
    monte_carlo = samples.mean()
    standard_error = samples.std(unbiased=True) / math.sqrt(len(samples))
    # Use a sampling-error criterion rather than fixed relative error: the cross-inner
    # product can be close to zero through cancellation even when individual terms are
    # large, making relative Monte Carlo error ill-conditioned.
    assert abs(exact - monte_carlo) < 5.0 * standard_error


def test_hybrid_adam_recovers_planted_interaction() -> None:
    for seed in range(3):
        result = toy.run_planted_optimization(seed, steps=1200)
        assert result.passed, result
