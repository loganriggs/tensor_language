#!/usr/bin/env python3
"""Pure selection contract for construction-robust causal projectors.

The optimizer may guide on smooth surrogates, but checkpoint licensing uses observed
per-environment causal metrics.  This module deliberately knows nothing about a model,
projector parameterization, or task authority so it can be reused without duplicating
selection logic in each DAS runner.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence


class MultiEnvironmentContractError(ValueError):
    """The checkpoint inventory cannot support the registered decision."""


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise MultiEnvironmentContractError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise MultiEnvironmentContractError(f"{label} is not finite")
    return number


def score_checkpoint(
    checkpoint: Mapping[str, object], *, target_panels: Sequence[str],
    control_panels: Sequence[str], projection_min: float, direction_min: float,
    mean_control_weight: float = 0.25,
) -> dict[str, object]:
    """Return fail-closed target feasibility and a worst-control lexicographic key."""
    targets = checkpoint.get("targets")
    controls = checkpoint.get("controls")
    if not isinstance(targets, Mapping) or not isinstance(controls, Mapping):
        raise MultiEnvironmentContractError("checkpoint needs target and control mappings")
    if not target_panels or not control_panels:
        raise MultiEnvironmentContractError("target and control panels must be nonempty")
    projection_min = _finite(projection_min, "projection_min")
    direction_min = _finite(direction_min, "direction_min")
    mean_control_weight = _finite(mean_control_weight, "mean_control_weight")
    if mean_control_weight < 0:
        raise MultiEnvironmentContractError("mean_control_weight must be nonnegative")

    target_rows = []
    for panel in target_panels:
        metric = targets.get(panel)
        if not isinstance(metric, Mapping):
            raise MultiEnvironmentContractError(f"missing target panel {panel}")
        projection = _finite(metric.get("signed_projection"), f"{panel}.signed_projection")
        direction = _finite(metric.get("direction_fraction"), f"{panel}.direction_fraction")
        violation = max(0.0, projection_min - projection) + max(0.0, direction_min - direction)
        target_rows.append((panel, projection, direction, violation))

    control_rows = []
    for panel in control_panels:
        metric = controls.get(panel)
        if not isinstance(metric, Mapping):
            raise MultiEnvironmentContractError(f"missing control panel {panel}")
        median = _finite(metric.get("median_kl"), f"{panel}.median_kl")
        mean = _finite(metric.get("mean_kl"), f"{panel}.mean_kl")
        maximum = _finite(metric.get("max_kl"), f"{panel}.max_kl")
        flips = metric.get("top1_flip_count")
        if isinstance(flips, bool) or not isinstance(flips, int) or flips < 0:
            raise MultiEnvironmentContractError(f"{panel}.top1_flip_count is invalid")
        control_rows.append((panel, median, mean, maximum, flips))

    worst_violation = max(row[3] for row in target_rows)
    worst_median = max(row[1] for row in control_rows)
    worst_mean = max(row[2] for row in control_rows)
    worst_maximum = max(row[3] for row in control_rows)
    total_flips = sum(row[4] for row in control_rows)
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise MultiEnvironmentContractError("step is invalid")
    return {
        "feasible": worst_violation == 0.0,
        "worst_target_violation": worst_violation,
        "minimum_target_projection": min(row[1] for row in target_rows),
        "minimum_target_direction_fraction": min(row[2] for row in target_rows),
        "worst_control_median_kl": worst_median,
        "worst_control_mean_kl": worst_mean,
        "worst_control_max_kl": worst_maximum,
        "total_control_flips": total_flips,
        "control_objective": worst_median + mean_control_weight * worst_mean,
        "step": step,
    }


def select_checkpoint(
    checkpoints: Sequence[Mapping[str, object]], *, target_panels: Sequence[str],
    control_panels: Sequence[str], projection_min: float, direction_min: float,
    mean_control_weight: float = 0.25,
) -> tuple[Mapping[str, object], dict[str, object]]:
    """Choose a checkpoint without letting control quality sacrifice target feasibility."""
    if not checkpoints:
        raise MultiEnvironmentContractError("checkpoint inventory is empty")
    scored = [
        (checkpoint, score_checkpoint(
            checkpoint, target_panels=target_panels, control_panels=control_panels,
            projection_min=projection_min, direction_min=direction_min,
            mean_control_weight=mean_control_weight,
        ))
        for checkpoint in checkpoints
    ]
    feasible = [item for item in scored if item[1]["feasible"]]
    pool = feasible or scored
    if feasible:
        key = lambda item: (
            item[1]["control_objective"], item[1]["worst_control_max_kl"],
            item[1]["total_control_flips"], item[1]["step"],
        )
    else:
        key = lambda item: (
            item[1]["worst_target_violation"], item[1]["control_objective"],
            item[1]["worst_control_max_kl"], item[1]["total_control_flips"],
            item[1]["step"],
        )
    return min(pool, key=key)
