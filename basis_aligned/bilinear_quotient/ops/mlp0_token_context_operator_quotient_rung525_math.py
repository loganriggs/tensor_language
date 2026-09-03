#!/usr/bin/env python3
"""Pure contraction, search, and scoring math for rung 525."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn.functional as F


PROBES = 256
HALF_PROBES = 128
RAW_COSINE_CEILING = 0.50


def operator_sketch(
    token_left: torch.Tensor,
    token_right: torch.Tensor,
    context_left: torch.Tensor,
    context_right: torch.Tensor,
    output_hidden: torch.Tensor,
    *,
    gain: float = 1.0,
) -> torch.Tensor:
    """Compute q_j^T K_t a_j for every token t and paired probe j.

    ``output_hidden[j]`` is ``Down.T @ q_j``. The result avoids constructing
    any 1152-by-1152 token operator.
    """
    if token_left.ndim != 2 or token_right.shape != token_left.shape:
        raise ValueError("token factor shapes differ")
    if context_left.ndim != 2 or context_right.shape != context_left.shape:
        raise ValueError("context factor shapes differ")
    if output_hidden.shape != context_left.shape:
        raise ValueError("output and context probe shapes differ")
    if token_left.shape[1] != context_left.shape[1]:
        raise ValueError("hidden dimensions differ")
    if not math.isfinite(gain):
        raise ValueError("gain must be finite")
    weighted_right = output_hidden * context_right
    weighted_left = output_hidden * context_left
    return gain * (token_left @ weighted_right.mT + token_right @ weighted_left.mT)


@dataclass(frozen=True)
class StandardizedSketch:
    values: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor


def standardize_from_donors(sketch: torch.Tensor, donor_indices: torch.Tensor) -> StandardizedSketch:
    if sketch.ndim != 2 or donor_indices.ndim != 1 or donor_indices.dtype != torch.int64:
        raise ValueError("invalid sketch or donor indices")
    donors = sketch[donor_indices].float()
    mean = donors.mean(0)
    scale = donors.std(0, unbiased=False).clamp_min(1e-8)
    values = (sketch.float() - mean) / scale
    if not bool(torch.isfinite(values).all()):
        raise ValueError("standardized sketch is non-finite")
    return StandardizedSketch(values=values, mean=mean, scale=scale)


def pair_distances(
    values: torch.Tensor, receiver_indices: torch.Tensor, donor_for_receiver: torch.Tensor
) -> torch.Tensor:
    if receiver_indices.shape != donor_for_receiver.shape:
        raise ValueError("receiver and donor maps differ")
    difference = values[receiver_indices] - values[donor_for_receiver]
    return difference.square().mean(1)


def nearest_far_donors(
    values: torch.Tensor,
    raw: torch.Tensor,
    receiver_indices: torch.Tensor,
    donor_indices: torch.Tensor,
    *,
    raw_cosine_ceiling: float = RAW_COSINE_CEILING,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find minimum sketch-distance donors under a raw-cosine exclusion."""
    if values.ndim != 2 or raw.ndim != 2 or values.shape[0] != raw.shape[0]:
        raise ValueError("value/raw population shapes differ")
    if receiver_indices.dtype != torch.int64 or donor_indices.dtype != torch.int64:
        raise ValueError("indices must be int64")
    if not math.isfinite(raw_cosine_ceiling) or not -1 <= raw_cosine_ceiling <= 1:
        raise ValueError("invalid cosine ceiling")
    donor_values = values[donor_indices]
    donor_norms = donor_values.square().sum(1)
    donor_raw = F.normalize(raw[donor_indices].float(), dim=1, eps=1e-12)
    chosen, distances, cosines = [], [], []
    for start in range(0, len(receiver_indices), chunk_size):
        ids = receiver_indices[start:start + chunk_size]
        receiver_values = values[ids]
        distance = (
            receiver_values.square().sum(1, keepdim=True)
            + donor_norms[None]
            - 2 * receiver_values @ donor_values.mT
        ) / values.shape[1]
        raw_cosine = F.normalize(raw[ids].float(), dim=1, eps=1e-12) @ donor_raw.mT
        eligible = raw_cosine <= raw_cosine_ceiling
        if not bool(eligible.any(1).all()):
            raise ValueError("one or more receivers have no eligible far donor")
        distance.masked_fill_(~eligible, torch.inf)
        minimum, position = distance.min(1)
        chosen.append(donor_indices[position])
        distances.append(minimum)
        cosines.append(raw_cosine.gather(1, position[:, None])[:, 0])
    return torch.cat(chosen), torch.cat(distances), torch.cat(cosines)


def nearest_raw_donors(
    raw: torch.Tensor, receiver_indices: torch.Tensor, donor_indices: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    donor_raw = F.normalize(raw[donor_indices].float(), dim=1, eps=1e-12)
    chosen, cosines = [], []
    for start in range(0, len(receiver_indices), chunk_size):
        ids = receiver_indices[start:start + chunk_size]
        similarity = F.normalize(raw[ids].float(), dim=1, eps=1e-12) @ donor_raw.mT
        maximum, position = similarity.max(1)
        chosen.append(donor_indices[position])
        cosines.append(maximum)
    return torch.cat(chosen), torch.cat(cosines)


def repeated_group_counts(donors: torch.Tensor) -> dict[str, int]:
    _unique, counts = donors.unique(return_counts=True)
    repeated = counts >= 2
    return {
        "distinct_donors": int(len(counts)),
        "repeated_donors": int(repeated.sum()),
        "receivers_in_repeated_groups": int(counts[repeated].sum()),
        "largest_group": int(counts.max()) if len(counts) else 0,
    }


def rankdata(values: torch.Tensor) -> torch.Tensor:
    """Deterministic average-free ranks; ties receive stable ordinal ranks."""
    order = torch.argsort(values, stable=True)
    result = torch.empty_like(values, dtype=torch.float64)
    result[order] = torch.arange(len(values), dtype=torch.float64, device=values.device)
    return result


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("invalid Spearman inputs")
    a = rankdata(left)
    b = rankdata(right)
    a = a - a.mean()
    b = b - b.mean()
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))


def score_real(
    *,
    candidate_a_distance: torch.Tensor,
    candidate_b_distance: torch.Tensor,
    raw_b_distance: torch.Tensor,
    random_b_distances: torch.Tensor,
    deranged_b_distance: torch.Tensor,
    candidate_donors: torch.Tensor,
    half_a_b_distances: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, object]:
    """Apply rung 525's frozen B/C and strong-null rules."""
    n = len(candidate_b_distance)
    if any(len(value) != n for value in (
        candidate_a_distance, raw_b_distance, deranged_b_distance,
        half_a_b_distances[0], half_a_b_distances[1], candidate_donors,
    )):
        raise ValueError("candidate score vectors differ")
    if random_b_distances.shape != (n, 16):
        raise ValueError("expected sixteen far-random controls per receiver")
    med_candidate = float(candidate_b_distance.median())
    med_raw = float(raw_b_distance.median())
    med_random = float(random_b_distances.median())
    med_deranged = float(deranged_b_distance.median())
    random_q05 = torch.quantile(random_b_distances, 0.05, dim=1)
    exceptional_fraction = float((candidate_b_distance < random_q05).float().mean())
    groups = repeated_group_counts(candidate_donors)
    half_medians = [float(value.median()) for value in half_a_b_distances]
    half_difference = abs(half_medians[0] - half_medians[1]) / max(
        min(half_medians), 1e-30
    )
    rho = spearman(candidate_a_distance, candidate_b_distance)
    prediction_b = bool(
        med_candidate <= 0.75 * med_raw
        and med_candidate <= 0.25 * med_random
        and med_candidate <= 0.75 * med_deranged
        and exceptional_fraction >= 0.05
    )
    prediction_c = bool(
        groups["repeated_donors"] >= 100
        and groups["receivers_in_repeated_groups"] >= 1_000
        and rho >= 0.50
        and half_difference <= 0.20
    )
    strong_null = bool(
        med_candidate >= 0.95 * med_raw or exceptional_fraction < 0.01
    )
    return {
        "median_bank_b_candidate_distance": med_candidate,
        "median_bank_b_raw_control_distance": med_raw,
        "median_bank_b_random_control_distance": med_random,
        "median_bank_b_deranged_control_distance": med_deranged,
        "candidate_over_raw": med_candidate / max(med_raw, 1e-30),
        "candidate_over_random": med_candidate / max(med_random, 1e-30),
        "candidate_over_deranged": med_candidate / max(med_deranged, 1e-30),
        "exceptional_vs_far_random_fraction": exceptional_fraction,
        "bank_a_to_bank_b_distance_spearman": rho,
        "half_search_bank_b_medians": half_medians,
        "half_search_median_relative_difference": half_difference,
        "groups": groups,
        "prediction_b_operator_transfer": prediction_b,
        "prediction_c_repeated_groups": prediction_c,
        "strong_null": strong_null,
        "physical_successor_licensed": prediction_b and prediction_c and not strong_null,
    }


__all__ = [
    "HALF_PROBES", "PROBES", "RAW_COSINE_CEILING", "StandardizedSketch",
    "nearest_far_donors", "nearest_raw_donors", "operator_sketch", "pair_distances",
    "rankdata", "repeated_group_counts", "score_real", "spearman",
    "standardize_from_donors",
]
