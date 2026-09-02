#!/usr/bin/env python3
"""RUNG489 -- distinguish native-state dominance from T/I-specific midpoint sharing."""

# BQGATE: EXPERIMENT
# pred_a exact native-state plus curvature instrument
# pred_b common native-state reader predicts every branch response
# pred_c T-I midpoint donors beat C and native-state controls
# pred_d the mechanism classification is stable across discovery halves
# pred_e the frozen mechanism class validates on new intervention outcomes

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import mlp1_finite_secant_factor_interchange_rung487 as base
import mlp1_finite_secant_factor_interchange_rung488 as parent


PREREG = POLY / "MLP1_NATIVE_STATE_DOMINANCE_RUNG489_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/mlp1_finite_secant_factor_interchange_rung488.py"
PARENT_RESULT = ROOT / "mlp1_finite_secant_factor_interchange_rung488_results.json"
OUT = ROOT / "mlp1_native_state_dominance_rung489_results.json"
HASHES = {
    PREREG: "3b3b12d0e7dc828fb5b2b3cb0b733b9ec83e87e0df8a7bb41a7f25e5a2bf3a15",
    PARENT_SOURCE: "05d63815f5196027930bbe19f9c31cc83031837e246217b9b2b4a13afee71a84",
    PARENT_RESULT: "ed0ce441fc6629275edf7ded6eb8b26b70d1b80fa0157fa46872c8292b038ebb",
}
BRANCHES = ("T", "C", "I")
MODES = ("native", "mid_T", "mid_C", "mid_I", "curvature")
WRITE_MODES = MODES[:-1]
DISCOVERY_RANGE = (0, 500)
VALIDATION_RANGE = (500, 1000)
SPLIT = 250
BATCH = 4
POSITION_SHIFTS = base.POSITION_SHIFTS
U = 2.0 ** -8
POLARIZATION_BF16_BAR = 8 * U * U
OWN_WRITE_BF16_BAR = 4 * U * U


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(PARENT_RESULT.read_text())
    if receipt.get("rung") != 488 \
            or receipt.get("pred_a_exact_lawful_instrument") is not True \
            or receipt.get("pred_b_stable_own_finite_responses") is not True \
            or receipt.get("pred_c_exact_frozen_factor_interchange_edge") is not True \
            or receipt.get("pred_d_stable_factor_sharing_graph") is not True \
            or receipt.get("pred_e_heldout_documents") is not True \
            or receipt.get("validation_licensed_and_opened") is not True \
            or receipt.get("strong_null") is not False \
            or receipt.get("selected_edges") != parent.FROZEN_EDGES \
            or receipt.get("next_step") \
            != "cross_document_T_I_shared_live_state_extraction_and_selective_swap":
        raise RuntimeError("rung488 did not license the specificity falsifier")
    rows, _positive, fit_rows, metadata = parent.validate_inputs()
    return rows, fit_rows, metadata


def _accumulate(stats, half, target_index, mode_index, control_index,
                prediction, target):
    prediction = prediction.double()
    target = target.double()
    stats[half, target_index, mode_index, control_index, 0] += float(
        (prediction * target).sum())
    stats[half, target_index, mode_index, control_index, 1] += float(
        prediction.square().sum())
    stats[half, target_index, mode_index, control_index, 2] += float(
        target.square().sum())


def _cosines(stats):
    return stats[..., 0] / (stats[..., 1] * stats[..., 2]).sqrt().clamp_min(1e-30)


def collect_phase(model, rows, reference, start_doc, stop_doc):
    arm_batches = []
    absent_batches = []
    native_batches = []
    write_stats = torch.zeros(
        2, len(BRANCHES), len(WRITE_MODES), 1 + len(POSITION_SHIFTS), 3,
        dtype=torch.float64)
    calls = {
        "native_forwards": 0, "absent_forwards": 0, "physical_forwards": 0,
        "native_attention": 0, "native_mlp": 0,
        "absent_attention": 0, "absent_mlp": 0, "site0_removal": 0,
        "physical_attention": 0, "physical_mlp": 0,
        "D_injections": 0, "A_injections": 0, "M_injections": 0,
    }
    errors = {
        "native_prefix_D_relative_squared_max": 0.0,
        "native_prefix_A_relative_squared_max": 0.0,
        "native_prefix_M_relative_squared_max": 0.0,
        "mlp0_state_max_abs": 0.0,
        "native_plus_curvature_float32_relative_squared_max": 0.0,
        "native_plus_curvature_bf16_relative_squared_max": 0.0,
        "own_native_write_relative_squared_max": 0.0,
        "analytical_num": 0.0, "analytical_den": 0.0,
        "deployed_num": 0.0, "deployed_den": 0.0,
    }
    device = next(model.parameters()).device
    mlp1 = model.transformer.h[1].mlp
    for start in range(start_doc, stop_doc, BATCH):
        stop = min(start + BATCH, stop_doc)
        tokens = rows[start:stop, :-1].to(device)
        targets = rows[start:stop, 1:].to(device)
        native_logits, native, native_calls = base._native_forward(
            model, tokens, reference)
        calls["native_forwards"] += 1
        calls["native_attention"] += native_calls["attention"]
        calls["native_mlp"] += native_calls["mlp"]
        native_batches.append(base.factorial_parent._per_token_ce(native_logits, targets))
        for name, error in native["prefix_errors"].items():
            key = f"native_prefix_{name}_relative_squared_max"
            errors[key] = max(errors[key], error)
        for key in ("analytical_num", "analytical_den", "deployed_num", "deployed_den"):
            errors[key] += native["identity"][key]

        absent = {}
        absent_ce = {}
        for branch in BRANCHES:
            logits, capture, audit = base._absent_forward(
                model, tokens, native, native["branches"][branch])
            absent[branch] = capture
            absent_ce[branch] = base.factorial_parent._per_token_ce(logits, targets)
            calls["absent_forwards"] += 1
            calls["absent_attention"] += audit["attention"]
            calls["absent_mlp"] += audit["mlp"]
            calls["site0_removal"] += audit["site0_removal"]
            errors["mlp0_state_max_abs"] = max(
                errors["mlp0_state_max_abs"], capture["mlp0_state_error"])
        absent_batches.append(torch.stack([absent_ce[name] for name in BRANCHES]))

        midpoints = {
            branch: (native["z"].float() + absent[branch]["z"].float()) / 2
            for branch in BRANCHES
        }
        target_arms = []
        for target_index, branch in enumerate(BRANCHES):
            delta = native["z"].float() - absent[branch]["z"].float()
            native_term = base._secant(mlp1, delta, native["z"].float())
            curvature = -0.5 * base._secant(mlp1, delta, delta)
            writes = {
                "native": native_term,
                **{f"mid_{donor}": base._secant(mlp1, delta, midpoints[donor])
                   for donor in BRANCHES},
                "curvature": curvature,
            }
            own = writes[f"mid_{branch}"]
            errors["native_plus_curvature_float32_relative_squared_max"] = max(
                errors["native_plus_curvature_float32_relative_squared_max"],
                base._relative_squared(native_term + curvature, own))
            deployed = native["M"] - absent[branch]["M"]
            combined = (native_term + curvature).to(deployed.dtype)
            errors["native_plus_curvature_bf16_relative_squared_max"] = max(
                errors["native_plus_curvature_bf16_relative_squared_max"],
                base._relative_squared(combined, deployed))
            own_write = (absent[branch]["M"].float() + own.float()).to(native["M"].dtype)
            errors["own_native_write_relative_squared_max"] = max(
                errors["own_native_write_relative_squared_max"],
                base._relative_squared(own_write, native["M"]))

            arms = []
            for mode in MODES:
                logits, audit = base._physical_forward(
                    model, tokens, absent[branch], writes[mode])
                calls["physical_forwards"] += 1
                calls["physical_attention"] += audit["attention"]
                calls["physical_mlp"] += audit["mlp"]
                for name in ("D", "A", "M"):
                    calls[f"{name}_injections"] += audit[name]
                arms.append(base.factorial_parent._per_token_ce(logits, targets))
            target_arms.append(torch.stack(arms, dim=-1))

            boundary = start_doc + SPLIT
            for local_start, local_stop, half in (
                    (0, max(0, min(stop, boundary) - start), 0),
                    (max(0, boundary - start), stop - start, 1)):
                if local_stop <= local_start:
                    continue
                sl = slice(local_start, local_stop)
                for mode_index, mode in enumerate(WRITE_MODES):
                    _accumulate(write_stats, half, target_index, mode_index, 0,
                                writes[mode][sl], own[sl])
                    factor = native["z"].float() if mode == "native" \
                        else midpoints[mode[-1]]
                    for control_index, shift in enumerate(POSITION_SHIFTS, start=1):
                        shifted = base._secant(
                            mlp1, delta[sl], torch.roll(factor[sl], shift, dims=1))
                        _accumulate(write_stats, half, target_index, mode_index,
                                    control_index, shifted, own[sl])
        arm_batches.append(torch.stack(target_arms))

    batches = math.ceil((stop_doc - start_doc) / BATCH)
    physical_per_batch = len(BRANCHES) * len(MODES)
    expected = {
        "native_forwards": batches,
        "absent_forwards": len(BRANCHES) * batches,
        "physical_forwards": physical_per_batch * batches,
        "native_attention": 18 * batches, "native_mlp": 18 * batches,
        "absent_attention": 18 * len(BRANCHES) * batches,
        "absent_mlp": 18 * len(BRANCHES) * batches,
        "site0_removal": len(BRANCHES) * batches,
        "physical_attention": 18 * physical_per_batch * batches,
        "physical_mlp": 18 * physical_per_batch * batches,
        "D_injections": physical_per_batch * batches,
        "A_injections": physical_per_batch * batches,
        "M_injections": physical_per_batch * batches,
    }
    instrument = {
        "calls": calls, "expected_calls": expected, "calls_exact": calls == expected,
        **{key: value for key, value in errors.items()
           if key not in ("analytical_num", "analytical_den", "deployed_num", "deployed_den")},
        "analytical_branch_identity_relative_squared": errors["analytical_num"]
            / max(errors["analytical_den"], 1e-30),
        "deployed_branch_identity_relative_squared": errors["deployed_num"]
            / max(errors["deployed_den"], 1e-30),
        "documents": stop_doc - start_doc,
    }
    return {
        "arms": torch.cat(arm_batches, dim=1),
        "absent": torch.cat(absent_batches, dim=1),
        "native": torch.cat(native_batches, dim=0),
        "write_cosines": _cosines(write_stats),
        "instrument": instrument,
    }


def _cell(collected, benefits, half, target_index, mode_index, docs):
    branch = BRANCHES[target_index]
    own_index = MODES.index(f"mid_{branch}")
    target = benefits[target_index, docs, ..., own_index]
    predictor = benefits[target_index, docs, ..., mode_index]
    report = base._effect_report(predictor, target)
    if MODES[mode_index] in WRITE_MODES:
        write_index = WRITE_MODES.index(MODES[mode_index])
        write_cosine = float(collected["write_cosines"][half, target_index, write_index, 0])
        controls = collected["write_cosines"][half, target_index, write_index, 1:]
        shift_q95 = float(torch.quantile(controls, .95, interpolation="higher"))
    else:
        write_cosine = shift_q95 = None
    return {
        **report,
        "write_cosine": write_cosine,
        "shift_q95": shift_q95,
    }


def analyze_phase(collected, frozen_class=None):
    arms = collected["arms"].double()
    absent = collected["absent"].double()
    benefits = absent[..., None] - arms
    halves = (slice(0, SPLIT), slice(SPLIT, 2 * SPLIT))
    reports = {}
    half_classes = []
    half_native_passes = []
    half_specific_passes = []
    for half, docs in enumerate(halves):
        target_reports = {}
        native_pass_all = True
        for target_index, branch in enumerate(BRANCHES):
            mode_reports = {
                mode: _cell(collected, benefits, half, target_index, mode_index, docs)
                for mode_index, mode in enumerate(MODES)
            }
            native_report = mode_reports["native"]
            native_holds = bool(
                native_report["cosine"] >= .90
                and native_report["best_scalar_adjusted_relative_error"] <= .45
                and native_report["write_cosine"] >= native_report["shift_q95"] + .15)
            native_pass_all &= native_holds
            own = benefits[target_index, docs, ..., MODES.index(f"mid_{branch}")]
            native_effect = benefits[target_index, docs, ..., MODES.index("native")]
            curvature_effect = benefits[target_index, docs, ..., MODES.index("curvature")]
            interaction = own - native_effect - curvature_effect
            scale = float(own.square().mean().sqrt())
            mode_reports["native"]["native_reader_holds"] = native_holds
            mode_reports["physical_decomposition"] = {
                "own_rms_nat": scale,
                "native_rms_over_own": float(native_effect.square().mean().sqrt()) / max(scale, 1e-30),
                "curvature_rms_over_own": float(curvature_effect.square().mean().sqrt()) / max(scale, 1e-30),
                "interaction_rms_over_own": float(interaction.square().mean().sqrt()) / max(scale, 1e-30),
                "closure_max_abs_nat": float(
                    (own - native_effect - curvature_effect - interaction).abs().max()),
            }
            target_reports[branch] = mode_reports

        specific_cells = {}
        for branch, desired in (("T", "mid_I"), ("I", "mid_T")):
            desired_report = target_reports[branch][desired]
            controls = [target_reports[branch]["mid_C"], target_reports[branch]["native"]]
            holds = bool(
                desired_report["cosine"] >= .80
                and desired_report["best_scalar_adjusted_relative_error"] <= .50
                and desired_report["cosine"] >= max(c["cosine"] for c in controls) + .03
                and desired_report["best_scalar_adjusted_relative_error"]
                <= min(c["best_scalar_adjusted_relative_error"] for c in controls) - .05)
            specific_cells[branch] = {
                "desired_mode": desired,
                "best_control_cosine": max(c["cosine"] for c in controls),
                "best_control_error": min(
                    c["best_scalar_adjusted_relative_error"] for c in controls),
                "holds": holds,
            }
        specific = all(report["holds"] for report in specific_cells.values())
        classification = "ti_specific" if specific else \
            "common_native" if native_pass_all else "neither"
        reports[f"half{half}"] = {
            "targets": target_reports,
            "specificity_cells": specific_cells,
            "native_pass_all": bool(native_pass_all),
            "ti_specific_pass_all": bool(specific),
            "classification": classification,
        }
        half_classes.append(classification)
        half_native_passes.append(bool(native_pass_all))
        half_specific_passes.append(bool(specific))
    stable_class = half_classes[0] if half_classes[0] == half_classes[1] \
        and half_classes[0] != "neither" else None
    validation_holds = frozen_class is None or all(
        classification == frozen_class for classification in half_classes)
    all_live = bool((benefits.square().mean(dim=(1, 2)) > 0).all())
    return {
        "half_reports": reports,
        "half_classifications": half_classes,
        "stable_classification": stable_class,
        "pred_b_common_native_reader": bool(all(half_native_passes)),
        "pred_c_T_I_specific_midpoint": bool(all(half_specific_passes)),
        "pred_d_stable_nonnull_classification": stable_class is not None,
        "frozen_classification": frozen_class,
        "frozen_classification_holds": bool(validation_holds),
        "all_physical_effects_live": all_live,
    }


def instrument_valid(instrument, analysis):
    return bool(
        instrument["calls_exact"]
        and instrument["native_prefix_D_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_A_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_M_relative_squared_max"] <= 1e-12
        and instrument["mlp0_state_max_abs"] == 0.0
        and instrument["native_plus_curvature_float32_relative_squared_max"] <= 1e-8
        and instrument["native_plus_curvature_bf16_relative_squared_max"]
        <= POLARIZATION_BF16_BAR
        and instrument["own_native_write_relative_squared_max"] <= OWN_WRITE_BF16_BAR
        and instrument["analytical_branch_identity_relative_squared"] <= 1e-8
        and instrument["deployed_branch_identity_relative_squared"] <= 1e-5
        and analysis["all_physical_effects_live"])


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(BRANCHES) == 3 and len(MODES) == 5
        assert POLARIZATION_BF16_BAR == 0.0001220703125
        assert OWN_WRITE_BF16_BAR == 0.00006103515625
        print(json.dumps({
            "status": "dry_run_passed", "rung": 489,
            "model_loaded": False, "outcomes_opened": False,
            "validation_outcomes_opened": False, "final_or_sealed_opened": False,
            "discovery_forwards": (500 // BATCH) * (1 + 3 + 15),
            "conditional_validation_forwards": (500 // BATCH) * (1 + 3 + 15),
            "modes": list(MODES),
            "registered_predictions": ["pred_a", "pred_b", "pred_c", "pred_d", "pred_e"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung489 output namespace already exists")
    rows, fit_rows, metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    reference = base.component_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))
    discovery = collect_phase(model, rows, reference, *DISCOVERY_RANGE)
    discovery_analysis = analyze_phase(discovery)
    pred_a = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
                  and instrument_valid(discovery["instrument"], discovery_analysis))
    pred_b = discovery_analysis["pred_b_common_native_reader"]
    pred_c = discovery_analysis["pred_c_T_I_specific_midpoint"]
    pred_d = discovery_analysis["pred_d_stable_nonnull_classification"]
    frozen_class = discovery_analysis["stable_classification"]
    validation_licensed = bool(pred_a and pred_d)
    validation = validation_analysis = None
    pred_e = False
    if validation_licensed:
        validation = collect_phase(model, rows, reference, *VALIDATION_RANGE)
        validation_analysis = analyze_phase(validation, frozen_class=frozen_class)
        pred_e = bool(
            instrument_valid(validation["instrument"], validation_analysis)
            and validation_analysis["pred_d_stable_nonnull_classification"]
            and validation_analysis["frozen_classification_holds"])
    strong_null = bool(not pred_a or not pred_d or not pred_e)
    result = {
        "status": "complete", "rung": 489,
        "claim_level": "heldout_native_state_vs_T_I_specific_reader_classification",
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(BRANCHES), "modes": list(MODES),
        "position_shift_offsets": list(POSITION_SHIFTS),
        "precision_bars": {
            "bf16_unit_roundoff": U,
            "native_plus_curvature_relative_squared": POLARIZATION_BF16_BAR,
            "own_write_relative_squared": OWN_WRITE_BF16_BAR,
        },
        "discovery": {
            "documents": list(DISCOVERY_RANGE), "split": SPLIT,
            "instrument": discovery["instrument"], "analysis": discovery_analysis,
            "native_ce_mean": float(discovery["native"].mean()),
        },
        "validation": None if validation is None else {
            "documents": list(VALIDATION_RANGE), "split": 750,
            "instrument": validation["instrument"], "analysis": validation_analysis,
            "native_ce_mean": float(validation["native"].mean()),
        },
        "selected_classification": frozen_class,
        'pred_a_exact_lawful_instrument': pred_a,
        'pred_b_common_native_state_reader': pred_b,
        'pred_c_T_I_specific_midpoint_reader': pred_c,
        'pred_d_stable_nonnull_classification': pred_d,
        'pred_e_heldout_intervention_outcomes': pred_e,
        "validation_licensed_and_opened": validation_licensed,
        "strong_null": strong_null,
        "final_or_sealed_opened": False,
        "execution_price": {
            "discovery_full_model_forwards": sum(
                discovery["instrument"]["calls"][key] for key in
                ("native_forwards", "absent_forwards", "physical_forwards")),
            "validation_full_model_forwards": 0 if validation is None else sum(
                validation["instrument"]["calls"][key] for key in
                ("native_forwards", "absent_forwards", "physical_forwards")),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "branch_change_semantics_under_shared_native_state_reader"
            if pred_e and frozen_class == "common_native" else
            "T_I_reader_extraction_and_selective_swap"
            if pred_e and frozen_class == "ti_specific" else
            "within_branch_integrated_secant_response_reader"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 489,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "selected_classification": frozen_class,
        "strong_null": strong_null,
        "validation_opened": validation_licensed,
        "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
