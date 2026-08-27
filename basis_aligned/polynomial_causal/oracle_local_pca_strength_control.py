"""Pure contracts for the dual-strength local-PCA oracle control.

The GPU runner calibrates intervention strengths and supplies paired per-row CE.
This module keeps scale selection and the registered decision rule independently
testable, including the conservative discovery/heldout minimum statistic.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence


SCALE_LOW = 0.1
SCALE_HIGH = 10.0
BISECTION_STEPS = 14
RELATIVE_MATCH_TOLERANCE = 0.01
SCALE_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)


def match_monotone_scale(
    target: float,
    metric_at_scale: Callable[[float], float],
    *,
    low: float = SCALE_LOW,
    high: float = SCALE_HIGH,
    steps: int = BISECTION_STEPS,
    relative_tolerance: float = RELATIVE_MATCH_TOLERANCE,
    scale_grid: Sequence[float] = SCALE_GRID,
) -> dict[str, Any]:
    """Match a positive target with fail-closed bounded monotone bisection.

    Every sampled metric is retained.  The routine rejects an unbracketed target,
    nonfinite measurements, or an observed decrease larger than numerical slack.
    It returns the sampled point closest to the target and requires a one-percent
    relative match by default.
    """
    target = float(target)
    low, high = float(low), float(high)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("strength target must be finite and positive")
    if (not math.isfinite(low) or not math.isfinite(high)
            or not 0.0 < low < high):
        raise ValueError("scale bounds must be finite, positive, and ordered")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("bisection steps must be a positive integer")
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative tolerance must be finite and positive")

    sampled: dict[float, float] = {}

    def measure(scale: float) -> float:
        if scale not in sampled:
            value = float(metric_at_scale(scale))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"nonfinite or negative strength metric at scale {scale}")
            sampled[scale] = value
        return sampled[scale]

    grid = [float(value) for value in scale_grid]
    if (not grid or grid[0] != low or grid[-1] != high
            or any(not math.isfinite(value) for value in grid)
            or any(left >= right for left, right in zip(grid, grid[1:]))):
        raise ValueError("scale grid must be finite, ordered, and span the bounds")
    grid_values = [measure(scale) for scale in grid]
    slack = max(1e-12, relative_tolerance * target * 0.05)
    if any(right + slack < left for left, right in zip(grid_values, grid_values[1:])):
        raise ValueError("strength metric decreases across registered scale grid")
    if target < grid_values[0] - slack or target > grid_values[-1] + slack:
        raise ValueError(
            f"strength target is not bracketed: target={target} "
            f"range=[{grid_values[0]},{grid_values[-1]}]"
        )

    bracket = next(
        ((left, right) for left, right, left_value, right_value
         in zip(grid, grid[1:], grid_values, grid_values[1:])
         if left_value - slack <= target <= right_value + slack),
        None,
    )
    if bracket is None:
        raise ValueError("strength target was not bracketed by adjacent grid points")
    left, right = bracket
    for _ in range(steps):
        middle = math.sqrt(left * right)
        value = measure(middle)
        if value < target:
            left = middle
        else:
            right = middle

    ordered = sorted(sampled.items())
    if any(right_value + slack < left_value
           for (_, left_value), (_, right_value) in zip(ordered, ordered[1:])):
        raise ValueError("strength metric was not monotone on sampled scales")
    scale, matched = min(
        ordered,
        key=lambda row: (abs(row[1] - target), abs(math.log(row[0]))),
    )
    relative_error = abs(matched - target) / target
    if relative_error > relative_tolerance:
        raise ValueError(
            f"strength match missed tolerance: relative_error={relative_error}"
        )
    return {
        "scale": scale,
        "target": target,
        "matched": matched,
        "relative_error": relative_error,
        "bounds": [low, high],
        "steps": steps,
        "relative_tolerance": relative_tolerance,
        "scale_grid": grid,
        "samples": [
            {"scale": sample_scale, "metric": value}
            for sample_scale, value in ordered
        ],
    }


def paired_gain(baseline_rows: Sequence[float], arm_rows: Sequence[float]) -> float:
    baseline = [float(value) for value in baseline_rows]
    arm = [float(value) for value in arm_rows]
    if not baseline or len(baseline) != len(arm):
        raise ValueError("paired CE rows must have equal positive length")
    if any(not math.isfinite(value) for value in baseline + arm):
        raise ValueError("paired CE rows must be finite")
    return sum(base - score for base, score in zip(baseline, arm)) / len(baseline)


def analyze_strength_control(
    candidate_gains: Mapping[str, Mapping[str, float | Sequence[float]]],
    null_gains: Mapping[str, Mapping[str, float | Sequence[float]]],
    *,
    full_heldout_gain: float,
    bootstrap_ci95: Sequence[float],
) -> dict[str, Any]:
    """Apply the preregistered two-split exact-null local-interface gate."""
    splits = ("discovery", "heldout")
    candidate = {split: float(candidate_gains[split]["mean"]) for split in splits}
    null_names = sorted(null_gains)
    if len(null_names) != 20:
        raise ValueError("strength control requires exactly twenty null identities")
    null = {
        name: {split: float(null_gains[name][split]["mean"]) for split in splits}
        for name in null_names
    }
    ci = [float(value) for value in bootstrap_ci95]
    full_heldout_gain = float(full_heldout_gain)
    values = list(candidate.values()) + [value for row in null.values() for value in row.values()]
    if (len(ci) != 2 or any(not math.isfinite(value) for value in ci)
            or not math.isfinite(full_heldout_gain)
            or any(not math.isfinite(value) for value in values)):
        raise ValueError("strength-control decision inputs must be finite")

    candidate_statistic = min(candidate.values())
    null_statistics = {name: min(row.values()) for name, row in null.items()}
    at_least_candidate = sum(
        value >= candidate_statistic for value in null_statistics.values()
    )
    exact_p = (1 + at_least_candidate) / 21
    fraction = (
        candidate["heldout"] / full_heldout_gain
        if full_heldout_gain > 0.0 else None
    )
    decision = {
        "candidate_positive_both_splits": all(value > 0.0 for value in candidate.values()),
        "heldout_bootstrap_ci95_lower_gt_zero": ci[0] > 0.0,
        "candidate_beats_all_twenty_nulls_joint_split": at_least_candidate == 0,
        "heldout_fraction_of_full_oracle_ge_0p40": (
            fraction is not None and fraction >= 0.40
        ),
    }
    decision["passes"] = all(decision.values())
    return {
        "candidate_split_gains": candidate,
        "candidate_joint_split_statistic": candidate_statistic,
        "null_joint_split_statistics": null_statistics,
        "nulls_at_least_candidate": at_least_candidate,
        "exact_one_sided_p": exact_p,
        "heldout_bootstrap_ci95": ci,
        "full_oracle_heldout_gain": full_heldout_gain,
        "heldout_fraction_of_full_oracle_gain": fraction,
        "decision": decision,
    }
