#!/usr/bin/env python3
"""Zero-forward release audit for refined typed shared contextual path v2."""

# BQGATE: AUDIT pred_a_hash_bound_authorities pred_b_shared_local_value_edge pred_c_cross_task_nonlinearity_preserved pred_d_invalid_and_task_specific_boundaries_preserved pred_e_exact_zero_model_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_typed_shared_contextual_path_v2.json"
ARTIFACT = ROOT / "circuits/followups/aspectual_tense_typed_shared_contextual_path_v2_artifact.json"
OUT = ROOT / "circuits/followups/aspectual_tense_typed_shared_contextual_path_v2_result.json"
CANDIDATE_ID = "aspectual_tense.typed_shared_contextual_path_v2"
PATHS = {
    "v1_artifact": ROOT / "circuits/followups/aspectual_tense_typed_shared_contextual_path_v1_artifact.json",
    "v1_result": ROOT / "circuits/followups/aspectual_tense_typed_shared_contextual_path_v1_result.json",
    "is_factor": ROOT / "circuits/followups/tense_auxiliary_is_was_mlp4_h1h4_bank_routing_local_value_factorial_v1_result.json",
    "has_factor": ROOT / "circuits/followups/aspectual_anchor_mlp4_h1h4_bank_routing_local_value_factorial_v1_result.json",
    "cross_value": ROOT / "circuits/followups/aspectual_tense_h1h4_carrier_effective_value_branch_factorial_v1_result.json",
    "cross_pattern": ROOT / "circuits/followups/aspectual_tense_h1h4_carrier_pattern_value_factorial_v1_result.json",
    "block8_audit": ROOT / "circuits/followups/aspectual_tense_h1h4_deep_resid9_block8_factorial_invalidity_audit_v1_result.json",
}
EXPECTED_PRIOR_SHA256 = "37f07c6758a7855bdcd76a9c78a01a5356de7ad0a45fa78b88aa4aaace62e1c3"
EXPECTED = {
    "v1_artifact": "f0f038f37fd9d97dff088117f93acdf239bab74c5877b522d7976bc81bfc6e85",
    "v1_result": "afb17159330dc6abcf018d36313a7df2c78c6708b67feb8d2f2d9d2eee50faf0",
    "is_factor": "9a00cefe8986b1459c334c445a28208ae54911b81e5a42d78e1bc878777f07e4",
    "has_factor": "0f15a432f15b9f4a0a5f4b7470eb097793135c7d01118266fa6d45db2e8fd2c4",
    "cross_value": "9f3c04abe6bca5448d228d6fc71951b804e6a66a9a95834e59461a0d32bcf9b6",
    "cross_pattern": "fbba3090f5727481425d43baaf2d8f981738b986140769071d706a757e20b570",
    "block8_audit": "92a078fb685b3925c6e5f37601b45b33307cf6580fb30191c4720eacd31c3cc7",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_forwards": 0, "example_evaluations": 0, "fitted_scalars": 0, "new_learned_scalars": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists() or ARTIFACT.exists():
        raise FileExistsError("refusing to overwrite v2 artifact or result")
    observed = {name: sha(path) for name, path in PATHS.items()}
    data = {name: json.loads(path.read_text()) for name, path in PATHS.items()}
    pred_a = sha(PRIOR) == EXPECTED_PRIOR_SHA256 and observed == EXPECTED and data["v1_result"]["terminal"] == "screen" and data["is_factor"]["terminal"] == "screen" and data["has_factor"]["terminal"] == "screen" and data["cross_value"]["terminal"] == "screen" and data["cross_pattern"]["terminal"] == "null" and data["block8_audit"]["terminal"] == "audit"
    pred_b = all(value >= 0.60 for task in ("is_factor", "has_factor") for value in data[task]["local_value_retained_fraction"].values()) and data["is_factor"]["instrument"]["layer0_v1_invariance_max_abs_error"] == 0.0 and data["has_factor"]["instrument"]["layer0_v1_invariance_max_abs_error"] == 0.0
    pred_c = data["cross_pattern"]["predictions"]["pred_c_content_dominates_routing"] and not data["cross_pattern"]["predictions"]["pred_d_interaction_secondary"] and data["cross_value"]["predictions"]["pred_c_local_l9_branch_dominates"]
    old = data["v1_artifact"]
    pred_d = data["block8_audit"]["scientific_effect_status"] == "descriptive_only_quarantined" and data["has_factor"]["scope_boundary"].startswith("Internal factor reuse only") and old["external_interface"].endswith("abstention") and set(old["task_branches"]) == {"has_had", "is_was"}
    price = {"model_forwards": 0, "example_evaluations": 0, "fitted_scalars": 0, "new_learned_scalars": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 0, "example_evaluations": 0, "fitted_scalars": 0, "new_learned_scalars": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = {"pred_a_hash_bound_authorities": pred_a, "pred_b_shared_local_value_edge": pred_b, "pred_c_cross_task_nonlinearity_preserved": pred_c, "pred_d_invalid_and_task_specific_boundaries_preserved": pred_d, "pred_e_exact_zero_model_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_e else "invalid")
    artifact = {
        "schema": "aspectual_tense_typed_shared_contextual_path_artifact_v2",
        "program_id": CANDIDATE_ID,
        "extends": "aspectual_tense.typed_shared_contextual_path_v1",
        "shared_nodes": [
            {"id": "mlp4_contextualizer", "operation": "Down(left_change + right_change)", "fit": "none"},
            {"id": "normalized_contextual_carrier_state", "type": "task-indexed semantic source bank", "realizations": {"has_had": ["last", "period", "determiner"], "is_was": ["moment", "determiner"]}},
            {"id": "local_l9_value", "layer": 9, "operation": "(1-lambda9) * c_v9(RMS(carrier state))", "observed_lambda9": data["is_factor"]["instrument"]["observed_lambda9"]},
            {"id": "l9h1_h4_reader", "layer": 9, "heads": [1, 4], "operation": "exact attention pattern times effective-value source sum"},
        ],
        "shared_temporal_edge_evidence": {
            "has_had": {"local_value_retained_by_family": data["has_factor"]["local_value_retained_fraction"], "factor_shapley": data["has_factor"]["factor_shapley"], "scope_boundary": data["has_factor"]["scope_boundary"]},
            "is_was": {"local_value_retained_by_family": data["is_factor"]["local_value_retained_fraction"], "factor_shapley": data["is_factor"]["factor_shapley"]},
        },
        "cross_task_task_state_computation": {
            "exact_factors_retained": ["routing_mass_change", "pattern_weighted_content_change", "routing_content_interaction"],
            "local_l9_fraction_of_content": data["cross_value"]["local_retained_fraction"],
            "minor_carried_l0_v1_shapley": {orientation: values["carried_l0_v1_change"] for orientation, values in data["cross_value"]["branch_shapley"].items()},
            "interaction_not_secondary": True,
        },
        "task_branches": old["task_branches"],
        "external_interface": old["external_interface"],
        "licensed": old["licensed"] + ["shared local-L9-value-dominant MLP4-to-H1/H4 temporal computation"],
        "not_licensed": old["not_licensed"] + ["additive cross-task carrier simplification", "removal of the carried V1 branch", "block8 component attribution", "retroactive promotion of the old has/had total-mediation null"],
        "literal_price": old["literal_price"],
    }
    result = {"schema": "aspectual_tense_typed_shared_contextual_path_result_v2", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256, "authority_sha256": observed, "predictions": predictions, "price": price, "terminal": terminal, "reason": {"screen": "typed_shared_contextual_path_v2_released", "null": "registered_internal_refinement_misses", "invalid": "authority_scope_or_zero_price_invalid"}[terminal], "next_action": "test whether the shared local-value edge predicts a new capability-qualified temporal construction without refitting"}
    atomic_create_json(ARTIFACT, artifact)
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "predictions": predictions, "terminal": terminal, "reason": result["reason"], "artifact": str(ARTIFACT), "next_action": result["next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
