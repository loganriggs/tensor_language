#!/usr/bin/env python3
"""Donor-free, target-guided actuator extension of aspectual program v9."""

from __future__ import annotations

import math

import torch

import aspectual_anchor_transparent_path_program_v9 as v9


PROGRAM_ID = "aspectual_anchor.has_vs_had.transparent_path_program_v10"
RANK1_BASIS_SHA256 = v9.RANK1_BASIS_SHA256
ACTUATOR_BUDGET = 7833.8336181640625
ACTUATOR_GRID_POINTS = 257

ProgramInputError = v9.ProgramInputError
carrier_amplitude = v9.carrier_amplitude
rank1_carrier_projection = v9.rank1_carrier_projection
compiled_sparse_suffix_delta = v9.compiled_sparse_suffix_delta
exact_final_logits = v9.exact_final_logits
exact_scored_pair = v9.exact_scored_pair


def exact_selected_margin(state, lm_head, *, target_id: int, foil_id: int):
    """Score one target/foil margin through the checkpoint's exact normalized soft-capped head."""
    if not isinstance(state, torch.Tensor) or state.ndim not in (1, 2):
        raise ProgramInputError("state must be a rank-one vector or rank-two batch")
    if not isinstance(target_id, int) or not isinstance(foil_id, int) or target_id == foil_id:
        raise ProgramInputError("target_id and foil_id must be distinct integers")
    if not hasattr(lm_head, "weight") or getattr(lm_head, "bias", None) is not None:
        raise ProgramInputError("lm_head must expose an unbiased weight matrix")
    if lm_head.weight.ndim != 2 or lm_head.weight.shape[1] != state.shape[-1]:
        raise ProgramInputError("lm_head input width must equal state width")
    if min(target_id, foil_id) < 0 or max(target_id, foil_id) >= lm_head.weight.shape[0]:
        raise ProgramInputError("target_id or foil_id out of vocabulary range")
    normalized = torch.nn.functional.rms_norm(state, (state.shape[-1],))
    target_raw = normalized @ lm_head.weight[target_id]
    foil_raw = normalized @ lm_head.weight[foil_id]
    return 30.0 * torch.tanh(target_raw / 30.0) - 30.0 * torch.tanh(foil_raw / 30.0)


def donor_free_margin_reflection(
    base_resid18,
    basis,
    lm_head,
    *,
    target_id: int,
    foil_id: int,
    budget: float = ACTUATOR_BUDGET,
    grid_points: int = ACTUATOR_GRID_POINTS,
):
    """Reflect a requested margin along the carrier without consuming a donor activation.

    The selected alpha minimizes ``|margin(x + alpha*q) + margin(x)|`` on the frozen grid.
    It depends only on the base state, unit carrier, target/foil readout, and one calibrated budget.
    """
    if not isinstance(base_resid18, torch.Tensor) or base_resid18.ndim != 1:
        raise ProgramInputError("base_resid18 must be a rank-one tensor")
    q = basis.reshape(-1) if isinstance(basis, torch.Tensor) else None
    if q is None or q.shape != base_resid18.shape:
        raise ProgramInputError("basis must flatten to the base state shape")
    if abs(float(q.float().norm()) - 1.0) > 1.0e-4:
        raise ProgramInputError("basis must be unit norm")
    if not isinstance(budget, (int, float)) or not math.isfinite(float(budget)) or float(budget) <= 0.0:
        raise ProgramInputError("budget must be a finite positive scalar")
    if not isinstance(grid_points, int) or grid_points < 3 or grid_points % 2 != 1:
        raise ProgramInputError("grid_points must be an odd integer of at least three")
    grid = torch.linspace(-float(budget), float(budget), grid_points, device=base_resid18.device, dtype=base_resid18.dtype)
    candidates = base_resid18[None, :] + grid[:, None] * q[None, :]
    base_margin = exact_selected_margin(base_resid18, lm_head, target_id=target_id, foil_id=foil_id)
    margins = exact_selected_margin(candidates, lm_head, target_id=target_id, foil_id=foil_id)
    index = int(torch.argmin(torch.abs(margins + base_margin)))
    alpha = grid[index]
    patched = candidates[index]
    target_logit, foil_logit = exact_scored_pair(patched, lm_head, answer_id=target_id, foil_id=foil_id)
    return {
        "patched_resid18": patched,
        "alpha": alpha,
        "grid_index": index,
        "base_target_margin": base_margin,
        "patched_target_margin": target_logit - foil_logit,
        "target_logit": target_logit,
        "foil_logit": foil_logit,
    }


def program_manifest() -> dict[str, object]:
    """Return v9 scope plus the donor-free selective actuator evidence."""
    manifest = dict(v9.program_manifest())
    manifest.update({
        "program_id": PROGRAM_ID,
        "donor_free_actuator": {
            "site": "resid:18",
            "basis_sha256": RANK1_BASIS_SHA256,
            "budget": ACTUATOR_BUDGET,
            "grid_points": ACTUATOR_GRID_POINTS,
            "calibrated_scalars": 1,
            "confirmation_donor_activation_required": False,
            "required_runtime_inputs": ("base_resid18", "rank1_basis", "lm_head", "target_id", "foil_id"),
        },
        "donor_free_confirmation": {
            "A_recovery_range": (1.0857816560749587, 1.1713244502901692),
            "P_margin_reflection_range": (0.9952393909938151, 0.9986408064580191),
            "C_normalized_unrelated_effect": 0.011031758729177964,
        },
        "stored_fit_scalars": 1153,
    })
    return manifest
