#!/usr/bin/env python3
"""Pure FIT/VALIDATION math for the preregistered rung-523 optimizer repair."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch


ROW_HIGH_LR = "row_specific_lr_0.03"
ROW_LOW_LR = "row_specific_lr_0.003"
FIXED_HIGH_LR = "fixed_target_map_lr_0.03"
FIXED_LOW_LR = "fixed_target_map_lr_0.003"
PROSPECTIVE_ARMS = (ROW_LOW_LR, FIXED_HIGH_LR, FIXED_LOW_LR)
ADOPTION_ORDER = (FIXED_HIGH_LR, ROW_LOW_LR, FIXED_LOW_LR)
EXPECTED_FITS_PER_ARM = 15
UPDATES_PER_FIT = 200
SPIKE_THRESHOLD = 100.0
EXTREME_THRESHOLD = 1_000.0
MAXIMUM_SPIKES_PER_ARM = 3


def fixed_target_map_scales(
    full_by_map: torch.Tensor,
    member_mask: torch.Tensor,
    split_rows: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return one FIT-only mean-square full-response scale per donor map.

    ``full_by_map`` is ``[map, local FIT row, token]``. ``member_mask`` is the
    global ``[row, token]`` mask and ``split_rows`` maps local rows back to it.
    """
    if full_by_map.ndim != 3 or not full_by_map.is_floating_point():
        raise ValueError("full_by_map must be a floating [map,row,token] tensor")
    if member_mask.ndim != 2 or member_mask.dtype != torch.bool:
        raise ValueError("member_mask must be a boolean [global row,token] tensor")
    if split_rows.ndim != 1 or split_rows.dtype != torch.int64:
        raise ValueError("split_rows must be a one-dimensional int64 tensor")
    if full_by_map.shape[1] != split_rows.numel():
        raise ValueError("full response and split-row axes differ")
    if full_by_map.shape[2] != member_mask.shape[1]:
        raise ValueError("full response and member-mask token axes differ")
    if split_rows.numel() == 0 or int(split_rows.min()) < 0 or int(split_rows.max()) >= member_mask.shape[0]:
        raise ValueError("split_rows reference rows outside the member mask")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    local_mask = member_mask[split_rows]
    if not bool(local_mask.any()):
        raise ValueError("target has no eligible FIT member positions")
    if not bool(torch.isfinite(full_by_map).all()):
        raise ValueError("full response contains a non-finite value")
    scales = torch.stack([
        values[local_mask].square().mean() + epsilon for values in full_by_map
    ])
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError("fixed target/map scale is not finite and positive")
    return scales


def normalized_target_loss(
    full_member: torch.Tensor,
    projected_member: torch.Tensor,
    projected_control: torch.Tensor,
    *,
    denominator: torch.Tensor | float,
    control_coefficient: float = 24.0,
) -> torch.Tensor:
    """Compute member mismatch plus control response under an explicit scale."""
    for name, value in (
        ("full_member", full_member),
        ("projected_member", projected_member),
        ("projected_control", projected_control),
    ):
        if value.ndim != 1 or value.numel() == 0 or not value.is_floating_point():
            raise ValueError(f"{name} must be a nonempty floating vector")
        if not bool(torch.isfinite(value).all().detach().cpu()):
            raise ValueError(f"{name} contains a non-finite value")
    if full_member.shape != projected_member.shape:
        raise ValueError("full and projected member vectors differ in shape")
    scale = torch.as_tensor(
        denominator, dtype=projected_member.dtype, device=projected_member.device
    )
    if scale.ndim != 0 or not bool(torch.isfinite(scale).detach().cpu()) or float(scale) <= 0:
        raise ValueError("denominator must be a finite positive scalar")
    if not math.isfinite(control_coefficient) or control_coefficient < 0:
        raise ValueError("control coefficient must be finite and nonnegative")
    member = (projected_member - full_member).square().mean() / scale
    control = projected_control.square().mean() / scale
    return member + control_coefficient * control


@dataclass(frozen=True)
class FitHealth:
    losses: Sequence[float]
    initial_common_validation: float
    final_common_validation: float
    orthonormality_error: float
    projector_distance: float
    finite_gradients: bool = True
    model_gradients_absent: bool = True


def score_candidate_cell(fits: Sequence[FitHealth]) -> dict[str, object]:
    """Apply every preregistered rung-523 health rule to one 15-fit cell."""
    if len(fits) != EXPECTED_FITS_PER_ARM:
        raise ValueError(f"candidate cell must contain exactly {EXPECTED_FITS_PER_ARM} fits")
    scored = []
    pooled_losses: list[float] = []
    for index, fit in enumerate(fits):
        losses = [float(value) for value in fit.losses]
        if len(losses) != UPDATES_PER_FIT:
            raise ValueError("each fit must contain exactly 200 losses")
        finite_losses = all(math.isfinite(value) for value in losses)
        initial_mean = sum(losses[:20]) / 20
        final_mean = sum(losses[-20:]) / 20
        failures = []
        if not finite_losses:
            failures.append("nonfinite_loss")
        if not fit.finite_gradients:
            failures.append("nonfinite_gradient")
        if not fit.model_gradients_absent:
            failures.append("model_parameter_gradient")
        if not math.isfinite(fit.orthonormality_error) or fit.orthonormality_error > 1e-5:
            failures.append("orthonormality")
        if not math.isfinite(fit.projector_distance) or fit.projector_distance <= 0.02:
            failures.append("projector_did_not_move")
        if not finite_losses or final_mean >= initial_mean:
            failures.append("final_window_not_below_initial_window")
        if (
            not math.isfinite(fit.initial_common_validation)
            or not math.isfinite(fit.final_common_validation)
            or fit.final_common_validation >= fit.initial_common_validation
        ):
            failures.append("common_validation_not_better_than_initialization")
        pooled_losses.extend(losses)
        scored.append({
            "fit_index": index,
            "passes_per_fit_rules": not failures,
            "failures": failures,
            "initial_window_mean": initial_mean,
            "final_window_mean": final_mean,
            "initial_common_validation": fit.initial_common_validation,
            "final_common_validation": fit.final_common_validation,
            "orthonormality_error": fit.orthonormality_error,
            "projector_distance": fit.projector_distance,
        })
    spike_count = sum(value > SPIKE_THRESHOLD for value in pooled_losses)
    extreme_count = sum(value > EXTREME_THRESHOLD for value in pooled_losses)
    cell_failures = []
    if not all(value["passes_per_fit_rules"] for value in scored):
        cell_failures.append("one_or_more_fits_failed")
    if spike_count > MAXIMUM_SPIKES_PER_ARM:
        cell_failures.append("more_than_three_losses_above_100")
    if extreme_count:
        cell_failures.append("one_or_more_losses_above_1000")
    return {
        "fit_count": len(fits),
        "update_count": len(pooled_losses),
        "spike_count_strictly_above_100": spike_count,
        "extreme_count_strictly_above_1000": extreme_count,
        "passing_fit_count": sum(value["passes_per_fit_rules"] for value in scored),
        "passes": not cell_failures,
        "failures": cell_failures,
        "fits": scored,
    }


def adoption_decision(cells: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Apply the frozen minimal-change decision table to prospective cells."""
    if set(cells) != set(PROSPECTIVE_ARMS):
        raise ValueError("prospective candidate census changed")
    passed = {name: bool(cells[name].get("passes")) for name in PROSPECTIVE_ARMS}
    if passed[FIXED_HIGH_LR]:
        adopted = FIXED_HIGH_LR
        if passed[ROW_LOW_LR]:
            diagnosis = "both_single_changes_sufficient_scale_not_uniquely_identified"
        else:
            diagnosis = "fixed_scale_sufficient"
    elif passed[ROW_LOW_LR]:
        adopted = ROW_LOW_LR
        diagnosis = "lower_learning_rate_sufficient"
    elif passed[FIXED_LOW_LR]:
        adopted = FIXED_LOW_LR
        diagnosis = "both_changes_required"
    else:
        adopted = None
        diagnosis = "raw_adam_through_qr_closed"
    return {
        "candidate_passes": passed,
        "adopted_arm": adopted,
        "diagnosis": diagnosis,
        "licenses_sealed_rung522_repeat": adopted is not None,
        "is_circuit_evidence": False,
    }


__all__ = [
    "ADOPTION_ORDER", "EXPECTED_FITS_PER_ARM", "EXTREME_THRESHOLD",
    "FIXED_HIGH_LR", "FIXED_LOW_LR", "FitHealth", "MAXIMUM_SPIKES_PER_ARM",
    "PROSPECTIVE_ARMS", "ROW_HIGH_LR", "ROW_LOW_LR", "SPIKE_THRESHOLD",
    "UPDATES_PER_FIT", "adoption_decision", "fixed_target_map_scales",
    "normalized_target_loss", "score_candidate_cell",
]
