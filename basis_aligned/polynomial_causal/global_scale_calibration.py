"""Pure contract for a foldable one-scalar compiler calibration.

This module selects scales on calibration metrics only and evaluates already-selected
scales on sealed roles.  It never loads model, rows, or result artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


SCALES = (0.35, 0.50, 0.65, 0.80, 1.00, 1.25)
BASELINE_SCALE = 1.0
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ScaleMetrics:
    scale: float
    target_ce: float
    teacher_kl: float
    top1_accuracy: float
    teacher_top1_agreement: float
    scored_tokens: int


def _validate_bank(bank: Sequence[ScaleMetrics]) -> dict[float, ScaleMetrics]:
    output: dict[float, ScaleMetrics] = {}
    for item in bank:
        if not isinstance(item, ScaleMetrics) or item.scale not in SCALES or (
            item.scale in output
        ) or not isinstance(item.scored_tokens, int) or isinstance(
            item.scored_tokens, bool
        ) or item.scored_tokens <= 0:
            raise ValueError("scale metric bank has invalid schema")
        numeric = (
            item.target_ce, item.teacher_kl, item.top1_accuracy,
            item.teacher_top1_agreement,
        )
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not (
            math.isfinite(value)
        ) for value in numeric) or item.target_ce < 0 or item.teacher_kl < 0 or any(
            value < 0 or value > 1 for value in numeric[2:]
        ):
            raise ValueError("scale metric bank contains invalid values")
        output[item.scale] = item
    if tuple(sorted(output)) != tuple(sorted(SCALES)) or len({
        item.scored_tokens for item in output.values()
    }) != 1:
        raise ValueError("scale metric bank is incomplete or scores different positions")
    return output


def _select(bank: Sequence[ScaleMetrics], field: str) -> float:
    values = _validate_bank(bank)
    best = min(getattr(item, field) for item in values.values())
    tied = [scale for scale, item in values.items() if getattr(item, field) <= (
        best + TIE_TOLERANCE
    )]
    # Prefer the least intervention; a remaining symmetric tie prefers the smaller
    # scale so bytes are deterministic.  Evaluation data never enters this rule.
    return min(tied, key=lambda scale: (abs(scale - BASELINE_SCALE), scale))


def select_calibration_scales(bank: Sequence[ScaleMetrics]) -> dict[str, float]:
    """Return separate task-predictive and teacher-faithful choices."""
    return {
        "target_ce_selected_scale": _select(bank, "target_ce"),
        "teacher_kl_selected_scale": _select(bank, "teacher_kl"),
    }


def _improvements(selected: ScaleMetrics, baseline: ScaleMetrics) -> dict[str, float]:
    return {
        "target_ce_improvement": baseline.target_ce - selected.target_ce,
        "teacher_kl_improvement": baseline.teacher_kl - selected.teacher_kl,
        "top1_accuracy_change": selected.top1_accuracy - baseline.top1_accuracy,
        "teacher_top1_agreement_change": (
            selected.teacher_top1_agreement - baseline.teacher_top1_agreement
        ),
    }


def evaluate_sealed_roles(
    selected: Mapping[str, float], roles: Mapping[str, Sequence[ScaleMetrics]],
) -> dict[str, object]:
    """Apply registered rolewise CE/KL gates without reselection."""
    if set(selected) != {"target_ce_selected_scale", "teacher_kl_selected_scale"} or (
        not roles
    ):
        raise ValueError("selected scales or sealed roles have invalid schema")
    for value in selected.values():
        if value not in SCALES:
            raise ValueError("selected scale is outside the frozen bank")
    ledger: dict[str, dict[str, dict[str, float]]] = {}
    for role, raw_bank in roles.items():
        if not isinstance(role, str) or not role:
            raise ValueError("sealed role name is invalid")
        bank = _validate_bank(raw_bank)
        baseline = bank[BASELINE_SCALE]
        ledger[role] = {
            purpose: _improvements(bank[scale], baseline)
            for purpose, scale in selected.items()
        }
    predictive = all(
        row["target_ce_selected_scale"]["target_ce_improvement"] >= 0.005 and
        row["target_ce_selected_scale"]["teacher_kl_improvement"] >= -0.010
        for row in ledger.values()
    )
    faithful = all(
        row["teacher_kl_selected_scale"]["teacher_kl_improvement"] >= 0.005 and
        row["teacher_kl_selected_scale"]["target_ce_improvement"] >= -0.010
        for row in ledger.values()
    )
    return {
        "selected_scales": dict(selected),
        "role_ledger": ledger,
        "predictive_scale_pass": predictive,
        "teacher_faithful_scale_pass": faithful,
        "any_registered_scale_pass": predictive or faithful,
        "selection_reopened_on_sealed_roles": False,
        "literal_price": {
            "stored_scalar_values": 1,
            "extra_deployed_float_values": 0,
            "extra_runtime_multiplies": 0,
            "folded_into_existing_tables_and_output_factors": True,
        },
    }

