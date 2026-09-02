#!/usr/bin/env python3
"""RUNG499 -- prospective corrected copy-task calibration on unopened action outcomes."""

# BQGATE: EXPERIMENT

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
for path in (ROOT, ROOT / "ops", POLY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_matcher_causal_action_quotient_rung498 as parent
import rung498_copy_task_portability_diagnosis as diagnosis


PREREG = POLY / "EQUALITY_MATCHER_COPY_TASK_CALIBRATION_RUNG499_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/equality_matcher_causal_action_quotient_rung498.py"
PARENT_RESULT = ROOT / "equality_matcher_causal_action_quotient_rung498_results.json"
PARENT_BUNDLE = ROOT / "equality_matcher_causal_action_quotient_rung498_bundle.pt"
DIAGNOSIS_SOURCE = ROOT / "ops/rung498_copy_task_portability_diagnosis.py"
DIAGNOSIS_RESULT = ROOT / "rung498_copy_task_portability_diagnosis_results.json"
OUT = ROOT / "equality_matcher_copy_task_calibration_rung499_results.json"
BUNDLE = ROOT / "equality_matcher_copy_task_calibration_rung499_bundle.pt"
HASHES = {
    PREREG: "df674c3c2132e913ede845778422a6b9a361ba3d7f17f2ebbaa4598883a44044",
    PARENT_SOURCE: "3186d610b77e1684849a54af79e83ce3d7a6a4338e36b3ec27ce2d7cc8696e59",
    PARENT_RESULT: "206ab207fc8698016c76a611cc2dbc353428fc54772c10cd173e45c9cd774a55",
    PARENT_BUNDLE: "dcdee47c84649e4bd01e124d7a7758ca63acf461a7d04d09beb4e7bffe625588",
    DIAGNOSIS_SOURCE: "93ea53f87264eb0893affd9daf6b06a8099e4d243366c7c4c5f6453f521e5b51",
    DIAGNOSIS_RESULT: "6a733bf55bf7920825dfc0a5ccf367fbf3e419ae98b249a68ca59ac8748ab3a0",
}
BOUNDS = (500, 1000, 750)
SELECTED_TAGS = (
    "r.1.0", "r.1.1", "r.1.2", "r.1.3", "r.11.1.1", "r.11.1.2",
    "r.11.3.1", "r.3.0", "r.5.0.1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_task_masks(rows):
    masks = diagnosis.build_masks(rows)
    return {
        "all_positive": masks["copy_positive"],
        "near_positive": masks["copy_near"],
        "far_positive": masks["copy_far"],
        "one_predecessor_positive": masks["copy_one_predecessor"],
        "multiple_predecessor_positive": masks["copy_multiple_predecessors"],
        "off_target": ~masks["copy_positive"] & torch.cat((
            torch.zeros(len(rows), 64, dtype=torch.bool),
            torch.ones(len(rows), 192, dtype=torch.bool)), dim=1),
    }


def _wins(positive, control, cosine_margin=.10, residual_margin=.15):
    return bool(
        positive["cosine"] >= control["cosine"] + cosine_margin
        or positive["scaled_residual"] <= control["scaled_residual"] - residual_margin)


def score(collection, analysis):
    instrument = collection["diagnostics"]
    pred_a = bool(
        collection["replay"]["logit_max_abs"] == 0.0
        and collection["replay"]["logit_relative_squared"] <= 1e-12
        and instrument["factor_reconstruction_max"] <= 1e-10
        and instrument["minimum_nonzero_edit_rms"] > 0
        and instrument["zero_intended_edit_actions"] == 0
        and instrument["calls"] == {"native": 125, "analytical": 2375}
        and instrument["calls_exact"]
        and min(collection["support"]["all_positive"]) >= 3000
        and int(collection["circuit_support"].min()) >= 10)
    positive_cells = [analysis["L5H5"][background]["score_donor"][half]
                      for background in parent.BACKGROUNDS for half in range(2)]
    pred_b = all(
        .75 <= row["equality_recovery"] <= 1.30
        and row["task_effect"]["cosine"] >= .75
        and row["task_effect"]["scaled_residual"] <= .70
        and abs(row["off_target_hybrid_minus_native_nat"]) <= .01
        for row in positive_cells)
    comparisons = []
    old_total_recovery_checks = []
    for background in parent.BACKGROUNDS:
        for half in range(2):
            positive = analysis["L5H5"][background]["score_donor"][half]
            controls = (
                ("L7H3_score", analysis["L7H3"][background]["score_donor"][half]),
                ("L5H5_payload", analysis["L5H5"][background]["payload_donor"][half]),
            )
            for name, control in controls:
                task_win = _wins(positive["task_effect"], control["task_effect"])
                circuit_win = _wins(positive["circuit_effect"], control["circuit_effect"])
                old_recovery = bool(
                    abs(control["equality_recovery"] - 1)
                    >= abs(positive["equality_recovery"] - 1) + .20)
                comparisons.append({"background": background, "quarter": half,
                                    "control": name, "task_pattern_win": task_win,
                                    "circuit_pattern_win": circuit_win,
                                    "holds": task_win and circuit_win})
                old_total_recovery_checks.append({"background": background, "quarter": half,
                                                   "control": name, "holds": old_recovery})
    pred_c = all(row["holds"] for row in comparisons)
    closure = []
    for half in range(2):
        present = analysis["L5H5"]["early_present"]["score_donor"][half]
        absent = analysis["L5H5"]["early_absent"]["score_donor"][half]
        scales = [present["circuit_effect"]["positive_fit_scale"],
                  absent["circuit_effect"]["positive_fit_scale"]]
        drift = abs(scales[0] - scales[1]) / max(abs(scales[0]), abs(scales[1]), 1e-30)
        cross = parent._cosine(present["hybrid_circuit_fingerprint"],
                               absent["hybrid_circuit_fingerprint"])
        holds = bool(
            present["circuit_effect"]["cosine"] >= .60
            and absent["circuit_effect"]["cosine"] >= .60
            and present["circuit_effect"]["scaled_residual"] <= .80
            and absent["circuit_effect"]["scaled_residual"] <= .80
            and min(scales) > 0 and drift <= .50 and cross >= .65)
        closure.append({"quarter": half, "scale_drift": drift,
                        "hybrid_background_cosine": cross, "holds": holds})
    pred_d = all(row["holds"] for row in closure)
    return pred_a, pred_b, pred_c, pred_d, {
        "typed_pattern_control_comparisons": comparisons,
        "old_total_recovery_control_clause_reported_not_gating": old_total_recovery_checks,
        "background_closure": closure,
    }


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert BOUNDS == (500, 1000, 750) and len(SELECTED_TAGS) == 9
        assert _wins({"cosine": .8, "scaled_residual": .4},
                     {"cosine": .6, "scaled_residual": .7})
        print(json.dumps({"status": "dry_run_passed", "rung": 499,
                          "model_loaded": False, "document_0_500_outcomes_loaded": False,
                          "forwards": 2500,
                          "predictions": ["pred_a", "pred_b", "pred_c", "pred_d", "pred_e"]},
                         indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung499 output namespace already exists")
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    # Hash the parent receipts but deliberately do not deserialize their outcome-bearing JSON/PT files.
    rows, circuit_masks, _discovery_tags, validation_tags, scales, metadata = parent.validate_inputs()
    task_masks = build_task_masks(rows)
    expected_tags = tuple(tag for tag in validation_tags if min(
        int((circuit_masks[tag][kind].view(1000, 256)[lo:hi]
             & task_masks["all_positive"][lo:hi]).sum())
        for kind in ("member", "slice_control")
        for lo, hi in ((500, 750), (750, 1000))) >= 10)
    if expected_tags != SELECTED_TAGS:
        raise RuntimeError(f"support-only tag selection changed: {expected_tags}")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    collection = parent.collect_phase(
        model, rows, circuit_masks, list(SELECTED_TAGS), task_masks, scales, BOUNDS)
    analysis = parent.analyze_phase(
        collection, circuit_masks, list(SELECTED_TAGS), task_masks, BOUNDS)
    pred_a, pred_b, pred_c, pred_d, checks = score(collection, analysis)
    pred_a = bool(pred_a and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256)
    pred_e = bool(pred_a and pred_b and pred_c and pred_d)
    bundle = {
        "schema": "equality_matcher_copy_task_calibration_rung499_nll_v1",
        "validation_nll": collection["nll"],
        "document_0_500_outcomes_included": False,
        "raw_rows_tokens_logits_or_hidden_states_included": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 499,
        "claim_level": "prospective_known_positive_copy_task_calibration_not_discovery_or_compression",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata, "documents": [500, 1000],
        "quarters": [[500, 750], [750, 1000]], "selected_tags": list(SELECTED_TAGS),
        "actions": {"donors": list(parent.DONORS), "backgrounds": list(parent.BACKGROUNDS),
                    "states": list(parent.STATES), "frozen_scales": scales},
        "collection": parent._serial_collection(collection),
        "analysis": analysis, "checks": checks,
        'pred_a_exact_supported_untouched_instrument': pred_a,
        'pred_b_copy_score_relation_recovers': pred_b,
        'pred_c_task_and_circuit_patterns_reject_controls': pred_c,
        'pred_d_closed_under_early_service_removal': pred_d,
        'pred_e_action_assay_calibrated': pred_e,
        "strong_null": bool(not pred_a or not pred_b or not pred_c or not pred_d),
        "document_0_500_outcomes_loaded": False,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {"forwards": sum(collection["diagnostics"]["calls"].values()),
                            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                            "deployed_parameters_added": 0, "deployed_parameters_saved": 0},
        "runtime_s": time.time() - started,
        "next_step": (
            "register_four_score_directed_action_graph" if pred_e else
            "repair_instrument_only" if not pred_a else
            "abandon_copy_score_calibration" if not pred_b else
            "change_observation_set_before_search" if not pred_c else
            "measure_explicit_donor_recipient_interaction"),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 499,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": result["strong_null"], "execution_price": result["execution_price"],
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
