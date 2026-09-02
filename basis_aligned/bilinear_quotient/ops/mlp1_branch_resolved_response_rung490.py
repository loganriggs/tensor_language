#!/usr/bin/env python3
"""RUNG490 -- held-out T/I-versus-C finite-response signature."""

# BQGATE: EXPERIMENT
# pred_a exact validation instrument and frozen parent
# pred_b T and I native-state responses pass while C fails
# pred_c T-I versus C response gaps are material
# pred_d curvature and nonlinear response ratios retain T>I>C ordering

from __future__ import annotations

import json
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
import mlp1_native_state_dominance_rung489 as parent


PREREG = POLY / "MLP1_BRANCH_RESOLVED_RESPONSE_RUNG490_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/mlp1_native_state_dominance_rung489.py"
PARENT_RESULT = ROOT / "mlp1_native_state_dominance_rung489_results.json"
OUT = ROOT / "mlp1_branch_resolved_response_rung490_results.json"
HASHES = {
    PREREG: "d1ccc22299155a148b1558435abbcba4b3c6b6c6f35eea3039ad4c5d1a71eadb",
    PARENT_SOURCE: "589e4f9bbbc161962af2853f7228ecd3b7ff2adb88dc694ca52022140a58a343",
    PARENT_RESULT: "dc589b9d585f56534bab814eee5a253bfda37038a58f3b2e5863982e37fc5b93",
}


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or parent.sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(PARENT_RESULT.read_text())
    if receipt.get("rung") != 489 \
            or receipt.get("pred_a_exact_lawful_instrument") is not True \
            or receipt.get("pred_b_common_native_state_reader") is not False \
            or receipt.get("pred_c_T_I_specific_midpoint_reader") is not False \
            or receipt.get("pred_d_stable_nonnull_classification") is not False \
            or receipt.get("pred_e_heldout_intervention_outcomes") is not False \
            or receipt.get("validation_licensed_and_opened") is not False \
            or receipt.get("strong_null") is not True \
            or receipt.get("selected_classification") is not None \
            or receipt["discovery"]["analysis"]["half_classifications"] \
            != ["neither", "neither"] \
            or receipt.get("next_step") != "within_branch_integrated_secant_response_reader":
        raise RuntimeError("rung489 did not license the branch-resolved validation")
    return parent.validate_inputs()


def score(validation, analysis):
    quarter_reports = []
    pred_b = pred_c = pred_d = True
    for half_name in ("half0", "half1"):
        targets = analysis["half_reports"][half_name]["targets"]
        native = {branch: targets[branch]["native"] for branch in parent.BRANCHES}
        ti_pass = all(native[branch]["native_reader_holds"] for branch in ("T", "I"))
        c_effect_fails = bool(
            native["C"]["cosine"] < .90
            or native["C"]["best_scalar_adjusted_relative_error"] > .45)
        write_controls_pass = all(
            native[branch]["write_cosine"] >= native[branch]["shift_q95"] + .15
            for branch in parent.BRANCHES)
        b_holds = bool(ti_pass and c_effect_fails and write_controls_pass)

        cosine_gap = min(native["T"]["cosine"], native["I"]["cosine"]) \
            - native["C"]["cosine"]
        error_gap = native["C"]["best_scalar_adjusted_relative_error"] \
            - max(native["T"]["best_scalar_adjusted_relative_error"],
                  native["I"]["best_scalar_adjusted_relative_error"])
        c_holds = bool(cosine_gap >= .05 and error_gap >= .10)

        curvature = {
            branch: targets[branch]["physical_decomposition"]["curvature_rms_over_own"]
            for branch in parent.BRANCHES
        }
        interaction = {
            branch: targets[branch]["physical_decomposition"]["interaction_rms_over_own"]
            for branch in parent.BRANCHES
        }
        curvature_order = curvature["T"] > curvature["I"] > curvature["C"]
        interaction_order = interaction["T"] > interaction["I"] > interaction["C"]
        d_holds = bool(curvature_order and interaction_order)
        pred_b &= b_holds
        pred_c &= c_holds
        pred_d &= d_holds
        quarter_reports.append({
            "quarter": half_name,
            "native_reports": native,
            "T_I_pass_C_fails_and_writes_controlled": b_holds,
            "min_T_I_minus_C_cosine": cosine_gap,
            "C_minus_max_T_I_error": error_gap,
            "material_contrast_holds": c_holds,
            "curvature_rms_over_own": curvature,
            "interaction_rms_over_own": interaction,
            "curvature_T_gt_I_gt_C": bool(curvature_order),
            "interaction_T_gt_I_gt_C": bool(interaction_order),
            "finite_correction_order_holds": d_holds,
        })
    return {
        "quarter_reports": quarter_reports,
        "pred_b_T_I_pass_C_fails": bool(pred_b),
        "pred_c_material_branch_contrast": bool(pred_c),
        "pred_d_finite_correction_order": bool(pred_d),
    }


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert parent.VALIDATION_RANGE == (500, 1000)
        assert parent.SPLIT == 250 and len(parent.MODES) == 5
        print(json.dumps({
            "status": "dry_run_passed", "rung": 490,
            "model_loaded": False, "outcomes_opened": False,
            "final_or_sealed_opened": False,
            "validation_forwards": (500 // parent.BATCH) * (1 + 3 + 15),
            "registered_predictions": ["pred_a", "pred_b", "pred_c", "pred_d"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung490 output namespace already exists")
    rows, fit_rows, metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    reference = base.component_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))
    validation = parent.collect_phase(
        model, rows, reference, *parent.VALIDATION_RANGE)
    analysis = parent.analyze_phase(validation)
    scored = score(validation, analysis)
    pred_a = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
                  and parent.instrument_valid(validation["instrument"], analysis))
    pred_b = scored["pred_b_T_I_pass_C_fails"]
    pred_c = scored["pred_c_material_branch_contrast"]
    pred_d = scored["pred_d_finite_correction_order"]
    strong_null = bool(not (pred_a and pred_b and pred_c and pred_d))
    result = {
        "status": "complete", "rung": 490,
        "claim_level": "heldout_branch_resolved_finite_response_signature",
        "source_hashes": {str(path): parent.sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "documents": list(parent.VALIDATION_RANGE), "split": 750,
        "instrument": validation["instrument"],
        "parent_analysis": analysis,
        "analysis": scored,
        'pred_a_exact_lawful_instrument': pred_a,
        'pred_b_T_I_native_pass_C_fails': pred_b,
        'pred_c_material_T_I_versus_C_contrast': pred_c,
        'pred_d_finite_correction_order_T_I_C': pred_d,
        "strong_null": strong_null,
        "final_or_sealed_opened": False,
        "execution_price": {
            "full_model_forwards": sum(
                validation["instrument"]["calls"][key] for key in
                ("native_forwards", "absent_forwards", "physical_forwards")),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "branchwise_integrated_response_shared_T_I_native_term_separate_corrections"
            if not strong_null else "fully_separate_branchwise_integrated_response"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 490,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
