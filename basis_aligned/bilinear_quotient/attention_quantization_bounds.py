"""Certified finite-input perturbation bounds for RMS-normalized product attention."""

from __future__ import annotations

import math

import torch


def rmsnorm_perturbation_bound(value: torch.Tensor, error: torch.Tensor,
                               epsilon: float) -> float:
    """Bound ||N_eps(value+error)-N_eps(value)|| by segment Lipschitzness.

    Reverse triangle inequality lower-bounds every norm on the straight segment by
    max(||value||-||error||, 0).  This can be loose but is deterministic and exact
    for the declared finite input.
    """
    if value.shape != error.shape or value.ndim != 1:
        raise ValueError("value and error must be same-shaped vectors")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    dimension = value.numel()
    error_norm = float(torch.linalg.vector_norm(error))
    minimum_norm = max(float(torch.linalg.vector_norm(value))-error_norm, 0.0)
    minimum_denominator = math.sqrt(minimum_norm**2/dimension+epsilon)
    return error_norm/minimum_denominator


def product_attention_error_bound(projected, projection_errors, epsilon: float) -> float:
    """Bound one product score's absolute error after Q/K RMSNorm and orthogonal RoPE."""
    names = ("q", "k", "q2", "k2")
    if set(projected) != set(names) or set(projection_errors) != set(names):
        raise ValueError("four Q/K/Q2/K2 vectors are required")
    dimensions = {projected[name].numel() for name in names}
    if len(dimensions) != 1:
        raise ValueError("projection dimensions disagree")
    root_dimension = math.sqrt(next(iter(dimensions)))
    normalized_errors = {
        name: rmsnorm_perturbation_bound(projected[name], projection_errors[name],
                                         epsilon)
        for name in names
    }
    first_branch = (normalized_errors["q"]+normalized_errors["k"])/root_dimension
    second_branch = (normalized_errors["q2"]+normalized_errors["k2"])/root_dimension
    # |a'b'-ab| <= |a'-a||b'| + |a||b'-b| and normalized dot scores
    # have magnitude at most one. Cap branch bounds at the exact diameter two.
    return min(first_branch, 2.0)+min(second_branch, 2.0)


def allocate_dyadic_steps(sensitivities, total_bit_levels: int,
                          minimum_level: int = 0):
    """Greedy separable allocation minimizing sum sensitivity_i * 2^(-level_i).

    Each added level halves that head's first-order error proxy and has one equal
    abstract cost unit. Deterministic tie-breaking uses the sorted head key.
    """
    if total_bit_levels < minimum_level*len(sensitivities):
        raise ValueError("budget below mandatory minimum levels")
    if any(value < 0 or not math.isfinite(value) for value in sensitivities.values()):
        raise ValueError("sensitivities must be finite and nonnegative")
    levels = {key: minimum_level for key in sorted(sensitivities)}
    remaining = total_bit_levels-sum(levels.values())
    for _ in range(remaining):
        # Marginal reduction is sensitivity * 2^(-level-1).
        winner = min(levels, key=lambda key: (
            -sensitivities[key]*2**(-levels[key]-1), key))
        levels[winner] += 1
    return levels
