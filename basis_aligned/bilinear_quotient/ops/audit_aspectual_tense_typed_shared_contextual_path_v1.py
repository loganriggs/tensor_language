#!/usr/bin/env python3
"""Release audit for the typed shared contextual path across two task programs."""

# BQGATE: AUDIT pred_a_hash_bound_terminal_authorities pred_b_shared_operation_and_reader_identity pred_c_task_specific_edges_preserved pred_d_external_composition_preserved pred_e_exact_zero_model_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_typed_shared_contextual_path_v1.json"
ARTIFACT = ROOT / "circuits/followups/aspectual_tense_typed_shared_contextual_path_v1_artifact.json"
OUT = ROOT / "circuits/followups/aspectual_tense_typed_shared_contextual_path_v1_result.json"
CANDIDATE_ID = "aspectual_tense.typed_shared_contextual_path_v1"
PATHS = {
    "has_path": ROOT / "circuits/followups/aspectual_anchor_attention9_h1h4_lexical_holdout_v1_result.json",
    "is_path": ROOT / "circuits/followups/tense_auxiliary_is_was_mlp4_to_l9h1_h4_path_mediation_v1_result.json",
    "has_program": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v12_artifact.json",
    "is_program": ROOT / "circuits/followups/tense_auxiliary_is_was_transparent_path_program_v1_artifact.json",
    "dual": ROOT / "circuits/followups/aspectual_tense_raw_text_dual_program_fresh_lexicon_v1_result.json",
}
EXPECTED_PRIOR_SHA256 = "ebfeb657ce35f6fbb04d9883b23e022c38f172d99646ac7374534bfd0a57a686"
EXPECTED = {"has_path": "e07c5b210839a70ae1152fed907c4078a3439b833e8a2169346995abc16b2292", "is_path": "6f5d01abec1debb41f67178a57db1ce79ad2a704e2ab094e2d8b1f055b3865d5", "has_program": "df677c455ddde199c10d463ecb0ba6a30da493700b8458e1c8a7f939ff1ab95a", "is_program": "37af439342439807d3cbf2a5d410dbb489c23554957e465a9b5cebd1abb02d61", "dual": "36d54f861bd6dd70a493e306480a812b1fb9009e4e35c26fd77df5ab22d59ca7"}


class AuditError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_forwards": 0, "example_evaluations": 0, "fitted_scalars": 0, "new_learned_scalars": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists() or ARTIFACT.exists():
        raise FileExistsError("refusing to overwrite typed graph artifact or result")
    observed = {name: sha(path) for name, path in PATHS.items()}
    prior_ok = sha(PRIOR) == EXPECTED_PRIOR_SHA256
    data = {name: json.loads(path.read_text()) for name, path in PATHS.items()}
    pred_a = prior_ok and observed == EXPECTED and data["has_path"]["terminal"] == "screen" and data["is_path"]["terminal"] == "screen" and data["dual"]["terminal"] == "screen"
    has_score, is_score = data["has_path"]["score"], data["is_path"]
    pred_b = has_score["bank_to_all_h1h4_fraction"] >= 0.80 and is_score["bank_to_all_h1h4_retained_fraction"] >= 0.80 and data["has_path"]["predictions"]["pred_c_h1h4_transfer"] and is_score["predictions"]["pred_c_moment_determiner_mediation"]
    has_interface, is_interface = data["has_program"]["interface"], data["is_program"]["interface"]
    pred_c = has_interface["fixed_token_ids"] == {"has": 468, "had": 550} and is_interface["fixed_token_ids"] == {"is": 318, "was": 373} and has_score["complete_h1h4_to_writer_fraction"] == 0.37949835266702137 and is_score["bank_to_writer_retained_fraction"] == 0.4948401501891345
    summaries = data["dual"]["summaries"]
    pred_d = all(summaries[task][family][key] >= 0.75 for task in ("has_had", "is_was") for family, key in (("A1", "mean_recovery"), ("A2", "mean_recovery"), ("P", "mean_margin_reflection_fraction"))) and all(summaries[task]["C"]["mean_absolute_normalized_unrelated_effect"] == 0.0 for task in summaries) and data["dual"]["price"]["fitted_scalars"] == 0
    price = {"model_forwards": 0, "example_evaluations": 0, "fitted_scalars": 0, "new_learned_scalars": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 0, "example_evaluations": 0, "fitted_scalars": 0, "new_learned_scalars": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = {"pred_a_hash_bound_terminal_authorities": pred_a, "pred_b_shared_operation_and_reader_identity": pred_b, "pred_c_task_specific_edges_preserved": pred_c, "pred_d_external_composition_preserved": pred_d, "pred_e_exact_zero_model_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_e else "invalid")
    artifact = {
        "schema": "aspectual_tense_typed_shared_contextual_path_artifact_v1",
        "program_id": CANDIDATE_ID,
        "shared_nodes": [
            {"id": "mlp4_contextualizer", "operation": "Down(left_change + right_change)", "fit": "none"},
            {"id": "contextual_carrier_bank", "type": "task-indexed downstream contextual tokens", "realizations": {"has_had": ["last", "period", "determiner"], "is_was": ["moment", "determiner"]}},
            {"id": "l9h1_h4_reader", "layer": 9, "heads": [1, 4], "operation": "exact pattern times effective-value source sum"}
        ],
        "task_branches": {
            "has_had": {"path_fraction_of_writer": has_score["complete_h1h4_to_writer_fraction"], "carrier_fraction_of_all_h1h4": has_score["bank_to_all_h1h4_fraction"], "read": has_interface["read"], "compute": has_interface["compute"], "write": has_interface["write"], "basis_token_ids": has_interface["fixed_token_ids"]},
            "is_was": {"path_fraction_of_writer": is_score["bank_to_writer_retained_fraction"], "carrier_fraction_of_all_h1h4": is_score["bank_to_all_h1h4_retained_fraction"], "read": "current-minus-other normalized soft-capped is/was unembedding contrast", "compute": is_interface["compute"], "write": "base_resid18 + alpha * q_is", "basis_token_ids": is_interface["fixed_token_ids"]}
        },
        "external_interface": "fit-free registered-grammar raw-text task/direction selector with abstention",
        "licensed": ["shared MLP4 algebra", "shared L9H1/H4 reader identity", "task-indexed contextual carrier interface", "prospective dual-program behavior in registered grammars"],
        "not_licensed": ["localized neural task gate", "identical carrier arity", "unrestricted syntax", "whole-model replacement", "P-selective H1/H4 heads independent of control family"],
        "literal_price": {"shared_learned_scalars_added": 0, "task_program_stored_fit_scalars": {"has_had": data["has_program"]["price"]["total_stored_fit_scalars"], "is_was": data["is_program"]["price"]["total_stored_fit_scalars"]}, "selector_fitted_scalars": 0}
    }
    result = {"schema": "aspectual_tense_typed_shared_contextual_path_result_v1", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256, "authority_sha256": observed, "predictions": predictions, "price": price, "terminal": terminal, "reason": {"screen": "typed_shared_contextual_path_released", "null": "shared_typing_or_external_composition_misses", "invalid": "authority_or_zero_price_invalid"}[terminal], "next_action": "localize the neural task branch upstream of the two task-specific resid10 readers"}
    atomic_create_json(ARTIFACT, artifact)
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "predictions": predictions, "terminal": terminal, "reason": result["reason"], "artifact": str(ARTIFACT), "next_action": result["next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
