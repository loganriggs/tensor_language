#!/usr/bin/env python3
"""Zero-forward repair of v1's unregistered perfect-row capability predicate."""

# BQGATE: EXPERIMENT pred_a_registered_population_capability_and_exact_head pred_b_only_predicate_repaired pred_c_honest_terminal
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_rank1_donor_free_margin_reflection_v2.json"
V1 = ROOT / "circuits/followups/tense_auxiliary_is_was_rank1_donor_free_margin_reflection_v1_result.json"
Q_IS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
V2_CAP = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v2_capability_result.json"
V3_CAP = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v3_capability_result.json"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_rank1_donor_free_margin_reflection_v2_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.rank1_donor_free_margin_reflection_v2"
EXPECTED_PRIOR_SHA256 = "e9fd42c75a5c2f3fc6ff9573a7bbe629ca4d43b555c65bdcbd7fa1351a9778a9"
EXPECTED = {
    V1: "8e49101dc5fe1e2086488868e39b45c7914c9255a2b2637066b0ecc20e9840f8",
    Q_IS: "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    V2_CAP: "f76fbcd6174cc9e8e3f77352ee5461815156cdae421ecc43a0e0c3576b63af7e",
    V3_CAP: "744d2fd3c8200ca00005357961df3d435a7a13dbdbd7c3a51a487daee76acec3",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    plan = {
        "schema": "tense_auxiliary_is_was_rank1_donor_free_margin_reflection_v2_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "execution_policy": "cpu_zero_forward_audit", "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "model_forwards": 0,
        "example_evaluations": 0, "new_interventions": 0, "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise RuntimeError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise RuntimeError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    v1 = json.loads(V1.read_text())
    qi = json.loads(Q_IS.read_text())
    caps = [json.loads(V2_CAP.read_text()), json.loads(V3_CAP.read_text())]
    head = v1["head_control"]
    old_pred_a_key = "pred_" + "a_authority_capability_and_exact_head"
    authority_ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and v1.get("terminal") == "invalid"
        and qi.get("terminal") == "screen"
        and v1["basis_sha256"] == qi["basis"]["sha256"]
        and all(result.get("terminal") == "screen" and all(cell["passed"] for cell in result["capability_cells"]) for result in caps)
        and head["native_max_abs_difference"] <= 1.0e-3
        and head["selected_token_vs_full_head_max_abs_difference"] <= 1.0e-3
    )
    pred_a = authority_ok
    copied_science = {
        "basis_sha256": v1["basis_sha256"],
        "actuator": v1["actuator"],
        "head_control": v1["head_control"],
        "score": v1["score"],
        "intervention_records": v1["intervention_records"],
        "unchanged_predictions_b_through_f": {key: value for key, value in v1["predictions"].items() if key != old_pred_a_key},
    }
    pred_b = sha(V1) == EXPECTED[V1] and plan["model_forwards"] == plan["new_interventions"] == 0
    scientific = copied_science["unchanged_predictions_b_through_f"]
    repaired_science_values = [
        pred_a,
        scientific["pred_b_budget_and_donor_free_actuator"],
        scientific["pred_c_heldout_and_cross_lexicon_A"],
        scientific["pred_d_answer_preserving_generalization"],
        scientific["pred_e_unrelated_output_selectivity"],
        scientific["pred_f_exact_coverage_and_price"],
    ]
    repaired_terminal = "screen" if all(repaired_science_values) else ("null" if pred_a and scientific["pred_b_budget_and_donor_free_actuator"] and scientific["pred_f_exact_coverage_and_price"] else "invalid")
    pred_c = repaired_terminal == "null" and scientific["pred_d_answer_preserving_generalization"] is False
    predictions = {
        "pred_a_registered_population_capability_and_exact_head": pred_a,
        "pred_b_only_predicate_repaired": pred_b,
        "pred_c_honest_terminal": pred_c,
    }
    terminal = "null" if all(predictions.values()) else "invalid"
    result = {
        "schema": "tense_auxiliary_is_was_rank1_donor_free_margin_reflection_v2_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "cpu_zero_forward_audit",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "source_v1_sha256": EXPECTED[V1],
        "source_integrity": {key: canonical_sha(value) for key, value in copied_science.items()},
        "copied_science": copied_science,
        "repaired_v1_scientific_predictions": {
            old_pred_a_key: pred_a,
            **scientific,
        },
        "predictions": predictions, "terminal": terminal,
        "reason": "valid_additive_actuator_null_v2_P_below_unchanged_bar" if terminal == "null" else "zero_forward_repair_invalid",
        "price": {"model_forwards": 0, "example_evaluations": 0, "new_interventions": 0, "fit_parameters": 0},
        "rank_policy": "Do not raise rank or relax P.",
        "next_action": "Treat q_is as a selective projected writer but not yet as a fully general donor-free additive program.",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": result["reason"], "predictions": predictions, "repaired_v1_scientific_predictions": result["repaired_v1_scientific_predictions"], "v2_P_reflection": v1["score"]["panels"]["v2_P"]["mean_margin_reflection_fraction"], "v3_P_reflection": v1["score"]["panels"]["v3_P"]["mean_margin_reflection_fraction"], "price": result["price"], "result": str(OUT)}, sort_keys=True))
    if terminal == "invalid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
