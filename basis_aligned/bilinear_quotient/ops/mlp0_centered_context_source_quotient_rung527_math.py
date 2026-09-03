#!/usr/bin/env python3
"""Exact algebra and frozen pair detector for rung 527."""

from __future__ import annotations

import itertools
import math
from typing import Iterable

import torch
import torch.nn.functional as F


GROUPS = ("SELF", "PREVIOUS", "NEAR", "DISTANT_SAME", "DISTANT_OTHER")
N_GROUPS = len(GROUPS)
N_LINEAR = N_GROUPS
N_QUADRATIC = N_GROUPS + math.comb(N_GROUPS, 2)
N_TERMS = N_LINEAR + N_QUADRATIC

MATERIAL_RMS = 0.0005
SCALE_MIN = 0.25
SCALE_MAX = 4.0
DISCOVERY_COSINE = (0.90, 0.80)
DISCOVERY_RESIDUAL = (0.35, 0.50)
CONFIRMATION_COSINE = 0.75
CONFIRMATION_RESIDUAL = 0.55
MAX_CANDIDATES = 8


def term_specs(groups: Iterable[str] = GROUPS) -> tuple[dict, ...]:
    """Return the frozen five linear, five self, and ten cross term identities."""
    groups = tuple(groups)
    if len(groups) != N_GROUPS or len(set(groups)) != N_GROUPS:
        raise ValueError("rung 527 requires five distinct source groups")
    rows = [
        {"name": f"LINEAR::{name}", "operation": "linear", "sources": (index,)}
        for index, name in enumerate(groups)
    ]
    rows.extend(
        {"name": f"SELF::{name}", "operation": "self", "sources": (index, index)}
        for index, name in enumerate(groups)
    )
    rows.extend(
        {"name": f"CROSS::{groups[left]}::{groups[right]}", "operation": "cross",
         "sources": (left, right)}
        for left, right in itertools.combinations(range(N_GROUPS), 2)
    )
    if len(rows) != N_TERMS:
        raise RuntimeError("term vocabulary does not contain exactly 20 terms")
    return tuple(rows)


TERM_SPECS = term_specs()
TERM_NAMES = tuple(row["name"] for row in TERM_SPECS)


def _project(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(value, weight)


def uncentered_terms(
    deltas: torch.Tensor,
    mean_state: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    down: torch.Tensor,
    gain: float | torch.Tensor,
) -> torch.Tensor:
    """Return five linear then fifteen uncentered quadratic output terms.

    ``deltas`` has shape ``[5, ..., input_width]``.  All tensors are expected to
    use a common floating dtype; callers choose float64 for identity tests and
    float32 for the real checkpoint.
    """
    if deltas.ndim < 2 or deltas.shape[0] != N_GROUPS:
        raise ValueError("deltas must have leading source dimension five")
    if deltas.shape[-1] != mean_state.shape[-1]:
        raise ValueError("mean state and source writes use different widths")
    if left.shape != right.shape or left.shape[1] != deltas.shape[-1]:
        raise ValueError("Left/Right shapes do not match source width")
    if down.shape[1] != left.shape[0]:
        raise ValueError("Down shape does not match product width")

    left_s = _project(deltas, left)
    right_s = _project(deltas, right)
    left_m = _project(mean_state, left)
    right_m = _project(mean_state, right)
    outputs = []
    for source in range(N_GROUPS):
        hidden = left_s[source] * right_m + left_m * right_s[source]
        outputs.append(_project(hidden, down))
    for source in range(N_GROUPS):
        outputs.append(_project(left_s[source] * right_s[source], down))
    for first, second in itertools.combinations(range(N_GROUPS), 2):
        hidden = (left_s[first] * right_s[second]
                  + left_s[second] * right_s[first])
        outputs.append(_project(hidden, down))
    return torch.stack(outputs) * torch.as_tensor(
        gain, dtype=deltas.dtype, device=deltas.device)


def center_terms(raw_terms: torch.Tensor, quadratic_means: torch.Tensor) -> torch.Tensor:
    """Subtract each FIT quadratic expectation from its own semantic term."""
    if raw_terms.shape[0] != N_TERMS:
        raise ValueError("raw term axis must contain 20 terms")
    if quadratic_means.shape != (N_QUADRATIC, raw_terms.shape[-1]):
        raise ValueError("quadratic means must have shape [15, output_width]")
    result = raw_terms.clone()
    view = (N_QUADRATIC,) + (1,) * (raw_terms.ndim - 2) + (raw_terms.shape[-1],)
    result[N_LINEAR:] -= quadratic_means.reshape(view)
    return result


def complete_semantic_context(
    deltas: torch.Tensor,
    mean_state: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    down: torch.Tensor,
    gain: float | torch.Tensor,
    complete_quadratic_mean: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the centered context function from the summed source write."""
    delta = deltas.sum(0)
    left_delta = _project(delta, left)
    right_delta = _project(delta, right)
    left_mean = _project(mean_state, left)
    right_mean = _project(mean_state, right)
    hidden = (left_delta * right_mean + left_mean * right_delta
              + left_delta * right_delta)
    return (torch.as_tensor(gain, dtype=delta.dtype, device=delta.device)
            * _project(hidden, down) - complete_quadratic_mean)


def safe_cosine(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    actual = torch.as_tensor(actual, dtype=torch.float64).reshape(-1)
    predicted = torch.as_tensor(predicted, dtype=torch.float64).reshape(-1)
    denominator = actual.norm() * predicted.norm()
    return float(actual.dot(predicted) / denominator.clamp_min(1e-30))


def relative_residual(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    actual = torch.as_tensor(actual, dtype=torch.float64).reshape(-1)
    predicted = torch.as_tensor(predicted, dtype=torch.float64).reshape(-1)
    return float((actual - predicted).norm() / actual.norm().clamp_min(1e-30))


def _pair_window(actual: torch.Tensor, donor: torch.Tensor, beta: float) -> dict:
    reciprocal = 1.0 / beta
    return {
        "left_from_right_cosine": safe_cosine(actual, beta * donor),
        "left_from_right_relative_residual": relative_residual(actual, beta * donor),
        "right_from_left_cosine": safe_cosine(donor, reciprocal * actual),
        "right_from_left_relative_residual": relative_residual(donor, reciprocal * actual),
    }


def pair_metrics(effects: torch.Tensor, left: int, right: int, beta: float) -> dict:
    """Score a frozen scalar on a tensor shaped [term, half, circuit]."""
    effects = torch.as_tensor(effects, dtype=torch.float64)
    if effects.ndim != 3 or effects.shape[0] != N_TERMS or effects.shape[1] != 2:
        raise ValueError("effects must have shape [20,2,circuit]")
    pooled = effects.flatten(1)
    row = {
        "left": left,
        "right": right,
        "left_name": TERM_NAMES[left],
        "right_name": TERM_NAMES[right],
        "beta_left_from_right": float(beta),
        "left_rms": float(pooled[left].square().mean().sqrt()),
        "right_rms": float(pooled[right].square().mean().sqrt()),
        "scale_holds": bool(SCALE_MIN <= abs(beta) <= SCALE_MAX),
        "windows": {},
    }
    for half in range(2):
        row["windows"][f"half{half}"] = _pair_window(
            effects[left, half], effects[right, half], beta)
    row["material"] = bool(row["left_rms"] >= MATERIAL_RMS
                           and row["right_rms"] >= MATERIAL_RMS)
    return row


def _window_holds(row: dict, half: int, cosine: float, residual: float) -> bool:
    window = row["windows"][f"half{half}"]
    return bool(
        min(window["left_from_right_cosine"], window["right_from_left_cosine"])
        >= cosine
        and max(window["left_from_right_relative_residual"],
                window["right_from_left_relative_residual"]) <= residual
    )


def discover_pairs(effects: torch.Tensor) -> tuple[list[dict], dict]:
    effects = torch.as_tensor(effects, dtype=torch.float64)
    if effects.ndim != 3 or effects.shape[0] != N_TERMS or effects.shape[1] != 2:
        raise ValueError("effects must have shape [20,2,circuit]")
    passing = []
    all_rows = []
    for left, right in itertools.combinations(range(N_TERMS), 2):
        denominator = effects[right, 0].square().sum().clamp_min(1e-30)
        beta = float(effects[right, 0].dot(effects[left, 0]) / denominator)
        if abs(beta) < 1e-30:
            continue
        row = pair_metrics(effects, left, right, beta)
        row["holds"] = bool(
            row["material"] and row["scale_holds"]
            and _window_holds(row, 0, DISCOVERY_COSINE[0], DISCOVERY_RESIDUAL[0])
            and _window_holds(row, 1, DISCOVERY_COSINE[1], DISCOVERY_RESIDUAL[1])
        )
        all_rows.append(row)
        if row["holds"]:
            passing.append(row)
    return passing, {
        "candidate_count": len(passing),
        "small_relation": bool(1 <= len(passing) <= MAX_CANDIDATES),
        "pairs_evaluated": len(all_rows),
    }


def confirmation_pairs(effects: torch.Tensor, candidates: list[dict],
                       pooled: torch.Tensor | None = None) -> tuple[list[dict], dict]:
    effects = torch.as_tensor(effects, dtype=torch.float64)
    if pooled is not None:
        pooled = torch.as_tensor(pooled, dtype=torch.float64)
        if pooled.shape != (N_TERMS, effects.shape[-1]):
            raise ValueError("pooled effects must have shape [20,circuit]")
    passing = []
    checks = {}
    for candidate in candidates:
        row = pair_metrics(
            effects, candidate["left"], candidate["right"],
            candidate["beta_left_from_right"])
        row["holds"] = bool(
            row["material"] and row["scale_holds"]
            and all(_window_holds(
                row, half, CONFIRMATION_COSINE, CONFIRMATION_RESIDUAL)
                    for half in range(2))
        )
        if pooled is not None:
            row["windows"]["pooled"] = _pair_window(
                pooled[row["left"]], pooled[row["right"]],
                row["beta_left_from_right"])
            pooled_window = row["windows"]["pooled"]
            row["holds"] = bool(
                row["holds"]
                and min(pooled_window["left_from_right_cosine"],
                        pooled_window["right_from_left_cosine"])
                    >= CONFIRMATION_COSINE
                and max(pooled_window["left_from_right_relative_residual"],
                        pooled_window["right_from_left_relative_residual"])
                    <= CONFIRMATION_RESIDUAL)
        key = f"{row['left_name']} <-> {row['right_name']}"
        checks[key] = row
        if row["holds"]:
            passing.append(candidate)
    return passing, checks


def permutation_control_counts(effects: torch.Tensor, seeds: Iterable[int]) -> list[int]:
    effects = torch.as_tensor(effects, dtype=torch.float64)
    counts = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(int(seed))
        shuffled = effects.clone()
        for term in range(N_TERMS):
            order = torch.randperm(effects.shape[-1], generator=generator)
            shuffled[term] = effects[term, :, order]
        candidates, _summary = discover_pairs(shuffled)
        counts.append(len(candidates))
    return counts


def planted_algebra(seed: int = 527_100) -> dict:
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    samples, input_width, product_width, output_width = 257, 7, 11, 5
    sources = torch.randn(N_GROUPS, samples, input_width, generator=generator, dtype=dtype)
    deltas = sources - sources.mean(1, keepdim=True)
    mean_state = torch.randn(input_width, generator=generator, dtype=dtype)
    left = torch.randn(product_width, input_width, generator=generator, dtype=dtype)
    right = torch.randn(product_width, input_width, generator=generator, dtype=dtype)
    down = torch.randn(output_width, product_width, generator=generator, dtype=dtype)
    gain = 0.71
    raw = uncentered_terms(deltas, mean_state, left, right, down, gain)
    quadratic_means = raw[N_LINEAR:].mean(1)
    terms = center_terms(raw, quadratic_means)
    full_quadratic_mean = quadratic_means.sum(0)
    direct = complete_semantic_context(
        deltas, mean_state, left, right, down, gain, full_quadratic_mean)
    reconstructed = terms.sum(0)
    numerical = torch.randn_like(direct, generator=generator) * 1e-4
    deployed = direct + numerical
    return {
        "expectation_sum_max_abs": float(
            (raw[N_LINEAR:].mean(1).sum(0) - full_quadratic_mean).abs().max()),
        "semantic_reconstruction_max_abs": float((reconstructed - direct).abs().max()),
        "deployed_with_remainder_max_abs": float(
            (reconstructed + numerical - deployed).abs().max()),
    }


def planted_pair_problem(seed: int = 527_200) -> tuple[torch.Tensor, set[tuple[int, int]]]:
    generator = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(32, N_TERMS, generator=generator,
                                        dtype=torch.float64)).Q.T
    effects = torch.stack((basis, basis + 0.002 * torch.randn(
        N_TERMS, 32, generator=generator, dtype=torch.float64)), dim=1) * 0.02
    effects[0] = 2.0 * effects[1]
    effects[2] = -0.5 * effects[3]
    return effects, {(0, 1), (2, 3)}


def planted_suite() -> dict:
    algebra = planted_algebra()
    effects, expected = planted_pair_problem()
    pairs, summary = discover_pairs(effects)
    observed = {(row["left"], row["right"]) for row in pairs}
    control_counts = permutation_control_counts(effects, range(527_300, 527_316))
    return {
        "algebra": algebra,
        "pair_summary": summary,
        "expected_pairs": sorted(expected),
        "observed_pairs": sorted(observed),
        "candidate_exact": observed == expected,
        "permutation_control_counts": control_counts,
        "permutations_destroy_pairs": max(control_counts, default=0) == 0,
        "holds": bool(
            max(algebra.values()) <= 1e-10
            and observed == expected
            and max(control_counts, default=0) == 0),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(planted_suite(), indent=2))
