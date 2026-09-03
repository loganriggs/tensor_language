#!/usr/bin/env python3
"""Pure contraction and scoring math for rung 526."""

from __future__ import annotations

import math
from typing import Mapping

import torch

import mlp0_token_context_operator_quotient_rung525_math as r525


RAW_COSINE_CEILING = 0.50
RANDOM_CONTROLS = 16


def circuit_signature(
    token_left: torch.Tensor,
    token_right: torch.Tensor,
    context_left: torch.Tensor,
    context_right: torch.Tensor,
    downstream_hidden: torch.Tensor,
    *,
    gain: float = 1.0,
) -> torch.Tensor:
    """Contract sum_i G[c,i]^T K_t da_i for all tokens and circuits.

    ``downstream_hidden[c,i]`` is ``D0.T @ G[c,i]``. Shapes are token
    factors ``[T,H]``, context factors ``[N,H]``, and downstream factors
    ``[C,N,H]``. The returned downstream signature is ``[T,C]``.
    """
    if token_left.ndim != 2 or token_right.shape != token_left.shape:
        raise ValueError("token factor shapes differ")
    if context_left.ndim != 2 or context_right.shape != context_left.shape:
        raise ValueError("context factor shapes differ")
    if downstream_hidden.ndim != 3:
        raise ValueError("downstream factors must be [circuit,position,hidden]")
    if downstream_hidden.shape[1:] != context_left.shape:
        raise ValueError("downstream and context factor shapes differ")
    if token_left.shape[1] != context_left.shape[1]:
        raise ValueError("hidden dimensions differ")
    if not math.isfinite(gain):
        raise ValueError("gain must be finite")
    right_accumulator = torch.einsum(
        "nih,ih->nh", downstream_hidden, context_right
    )
    left_accumulator = torch.einsum(
        "nih,ih->nh", downstream_hidden, context_left
    )
    return gain * (
        token_left @ right_accumulator.mT
        + token_right @ left_accumulator.mT
    )


def explicit_circuit_signature(
    token_left: torch.Tensor,
    token_right: torch.Tensor,
    context_left: torch.Tensor,
    context_right: torch.Tensor,
    downstream_hidden: torch.Tensor,
    *,
    gain: float = 1.0,
) -> torch.Tensor:
    """Reference implementation that explicitly forms every hidden product."""
    values = []
    for token in range(len(token_left)):
        interaction = (
            token_left[token][None] * context_right
            + context_left * token_right[token][None]
        )
        values.append(gain * torch.einsum("nih,ih->n", downstream_hidden, interaction))
    return torch.stack(values)


def derange_coordinates(values: torch.Tensor) -> torch.Tensor:
    """Apply a deterministic, token-specific circuit-coordinate permutation."""
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("expected at least two circuit coordinates")
    width = values.shape[1]
    token = torch.arange(len(values), device=values.device, dtype=torch.int64)
    coordinate = torch.arange(width, device=values.device, dtype=torch.int64)
    shift = (token * 17 + 3) % width
    indices = (coordinate[None] + shift[:, None]) % width
    return values.gather(1, indices)


def _median_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return float(numerator.median() / denominator.median().clamp_min(1e-30))


def score_discovery(
    *,
    distance_d0: torch.Tensor,
    distance_d1: torch.Tensor,
    raw_d1: torch.Tensor,
    random_d1: torch.Tensor,
    scrambled_d1: torch.Tensor,
    candidate_donors: torch.Tensor,
    circuit_half_d1: tuple[torch.Tensor, torch.Tensor],
    rung525_donors: torch.Tensor,
) -> dict[str, object]:
    """Apply frozen B/D and discovery strong-null rules."""
    n = len(distance_d1)
    one_dimensional = (
        distance_d0, raw_d1, scrambled_d1, candidate_donors,
        circuit_half_d1[0], circuit_half_d1[1], rung525_donors,
    )
    if any(value.shape != (n,) for value in one_dimensional):
        raise ValueError("discovery score vectors differ")
    if random_d1.shape != (n, RANDOM_CONTROLS):
        raise ValueError("random controls changed shape")
    random_q05 = torch.quantile(random_d1, 0.05, dim=1)
    exceptional = float((distance_d1 < random_q05).float().mean())
    ratios = {
        "candidate_over_raw": _median_ratio(distance_d1, raw_d1),
        "candidate_over_random": _median_ratio(distance_d1, random_d1),
        "candidate_over_scrambled": _median_ratio(distance_d1, scrambled_d1),
    }
    rho = r525.spearman(distance_d0, distance_d1)
    groups = r525.repeated_group_counts(candidate_donors)
    half_medians = [float(value.median()) for value in circuit_half_d1]
    half_difference = abs(half_medians[0] - half_medians[1]) / max(
        min(half_medians), 1e-30
    )
    changed_fraction = float((candidate_donors != rung525_donors).float().mean())
    pred_b = bool(
        ratios["candidate_over_raw"] <= 0.75
        and ratios["candidate_over_random"] <= 0.35
        and ratios["candidate_over_scrambled"] <= 0.75
        and exceptional >= 0.05
        and rho >= 0.40
    )
    pred_d = bool(
        groups["repeated_donors"] >= 100
        and groups["receivers_in_repeated_groups"] >= 1_000
        and half_difference <= 0.25
        and changed_fraction >= 0.80
    )
    strong_null = bool(
        ratios["candidate_over_raw"] >= 0.95 or exceptional < 0.01
    )
    return {
        "median_d1_candidate_distance": float(distance_d1.median()),
        "median_d1_raw_distance": float(raw_d1.median()),
        "median_d1_random_distance": float(random_d1.median()),
        "median_d1_scrambled_distance": float(scrambled_d1.median()),
        **ratios,
        "exceptional_vs_far_random_fraction": exceptional,
        "d0_d1_candidate_distance_spearman": rho,
        "groups": groups,
        "circuit_half_d1_medians": half_medians,
        "circuit_half_median_relative_difference": half_difference,
        "fraction_different_from_rung525_donor": changed_fraction,
        "prediction_b_document_transfer": pred_b,
        "prediction_d_reusable_changed_groups": pred_d,
        "strong_null": strong_null,
    }


def score_validation_half(
    *,
    candidate: torch.Tensor,
    raw: torch.Tensor,
    random: torch.Tensor,
    scrambled: torch.Tensor,
) -> dict[str, object]:
    n = len(candidate)
    if any(value.shape != (n,) for value in (raw, scrambled)):
        raise ValueError("validation score vectors differ")
    if random.shape != (n, RANDOM_CONTROLS):
        raise ValueError("validation random controls changed shape")
    q05 = torch.quantile(random, 0.05, dim=1)
    exceptional = float((candidate < q05).float().mean())
    ratios = {
        "candidate_over_raw": _median_ratio(candidate, raw),
        "candidate_over_random": _median_ratio(candidate, random),
        "candidate_over_scrambled": _median_ratio(candidate, scrambled),
    }
    passes = bool(
        ratios["candidate_over_raw"] <= 0.85
        and ratios["candidate_over_random"] <= 0.50
        and ratios["candidate_over_scrambled"] <= 0.85
        and exceptional >= 0.05
    )
    return {
        "median_candidate_distance": float(candidate.median()),
        **ratios,
        "exceptional_vs_far_random_fraction": exceptional,
        "passes": passes,
    }


__all__ = [
    "RANDOM_CONTROLS", "RAW_COSINE_CEILING", "circuit_signature",
    "derange_coordinates", "explicit_circuit_signature", "score_discovery",
    "score_validation_half",
]
