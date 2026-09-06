#!/usr/bin/env python3
"""Zero-forward audit of the typed aspectual-anchor causal path program."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_bound_authority pred_b_typed_graph_exact pred_c_discovery_metrics_exact pred_d_prospective_metrics_exact pred_e_price_dependency_scope_exact
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_managed_runner as managed

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_transparent_path_program_release_v1.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v1_result.json"
PATHS = {
    "aspectual_anchor_transparent_path_program_v1_artifact.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v1_artifact.json",
    "aspectual_anchor_mlp4_bilinear_response_factorial_v2_result.json": ROOT / "circuits/followups/aspectual_anchor_mlp4_bilinear_response_factorial_v2_result.json",
    "aspectual_anchor_mlp4_to_attention5_four_head_source_identity_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_mlp4_to_attention5_four_head_source_identity_v1_result.json",
    "aspectual_anchor_mlp4_induced_final_query_onset_v2_result.json": ROOT / "circuits/followups/aspectual_anchor_mlp4_induced_final_query_onset_v2_result.json",
    "aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2_result.json": ROOT / "circuits/followups/aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2_result.json",
    "aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1_result.json",
    "aspectual_anchor_explicit_path_lexical_holdout_v2_result.json": ROOT / "circuits/followups/aspectual_anchor_explicit_path_lexical_holdout_v2_result.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict[str, object]:
    return json.loads(PATHS[name].read_text())


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "price": {"model_forwards": 0, "fits": 0}}))
        return
    prior = json.loads(PRIOR.read_text())
    observed = {name: sha(path) for name, path in PATHS.items()}
    artifact = load("aspectual_anchor_transparent_path_program_v1_artifact.json")
    writer = load("aspectual_anchor_mlp4_bilinear_response_factorial_v2_result.json")
    source5 = load("aspectual_anchor_mlp4_to_attention5_four_head_source_identity_v1_result.json")
    onset = load("aspectual_anchor_mlp4_induced_final_query_onset_v2_result.json")
    block9 = load("aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2_result.json")
    path9 = load("aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1_result.json")
    holdout = load("aspectual_anchor_explicit_path_lexical_holdout_v2_result.json")
    program = artifact["program"]
    evidence = artifact["causal_evidence"]

    authority_ok = (
        observed == prior["authority"]
        and [writer["terminal"], source5["terminal"], onset["terminal"], block9["terminal"], path9["terminal"], holdout["terminal"]]
        == ["screen", "screen", "null", "screen", "null", "screen"]
    )
    graph_ok = (
        artifact["schema"] == "aspectual_anchor_transparent_path_program_artifact_v1"
        and program["mlp4_hidden_response"]["left_change"] == "(Left_d - Left_b) * Right_b"
        and program["mlp4_hidden_response"]["right_change"] == "Left_b * (Right_d - Right_b)"
        and program["mlp4_hidden_response"]["write_sites"] == ["last@resid5", "period@resid5", "determiner@resid5"]
        and program["attention5_transport"]["heads"] == [7, 1, 6, 8]
        and program["attention5_transport"]["sources"] == ["last", "period", "determiner"]
        and program["attention5_transport"]["destination"] == "final_query@resid6"
        and program["downstream_composition"]["attention9_reader"]["heads"] == [1, 4]
    )
    curve = onset["score"]["boundary_curve"]
    shapley = block9["score"]["factorial_shapley_target_recovery"]
    discovery_ok = (
        evidence["discovery_writer_mean_recovery"] == writer["score"]["factorial_arms"]["left_change+right_change"]["mean_target_recovery"]
        and evidence["discovery_two_term_retained_fraction"] == writer["score"]["two_factor_retained_fraction"]
        and evidence["discovery_attention5_four_head_mean_recovery"] == source5["score"]["arms"]["complete_four_heads"]["mean_target_recovery"]
        and evidence["discovery_source_identity_fraction"] == source5["score"]["bank_retained_fraction"]
        and program["downstream_composition"]["carried_curve"] == {"resid5": curve["5"]["mean_target_recovery"], "resid6": curve["6"]["mean_target_recovery"], "resid7": curve["7"]["mean_target_recovery"], "resid8": curve["8"]["mean_target_recovery"], "resid9": curve["9"]["mean_target_recovery"]}
        and program["downstream_composition"]["block9_crossing_shapley"] == {"carried9": shapley["carried9"], "attention9": shapley["attention9"], "mlp9": shapley["mlp9"]}
        and program["downstream_composition"]["attention9_reader"]["discovery_mean_recovery"] == path9["score"]["arms"]["h1h4_last_period_determiner"]["mean_target_recovery"]
    )
    holdout_arms = holdout["score"]["arms"]
    prospective_ok = (
        evidence["prospective_writer_mean_recovery"] == holdout_arms["writer_two_term"]["mean_target_recovery"]
        and evidence["prospective_attention5_four_head_mean_recovery"] == holdout_arms["attention5_complete_four"]["mean_target_recovery"]
        and evidence["prospective_attention5_four_head_all_head_fraction"] == holdout["score"]["four_head_to_all_nine_fraction"]
        and evidence["prospective_source_identity_fraction"] == holdout["score"]["bank_to_complete_fraction"]
        and evidence["prospective_target_direction_fractions"] == {"attention5_A1": holdout_arms["attention5_complete_four"]["families"]["A1"]["direction_fraction"], "attention5_A2": holdout_arms["attention5_complete_four"]["families"]["A2"]["direction_fraction"]}
    )
    required_exclusions = {"standalone native-margin prediction", "full-logit prediction", "free-form text", "new-construction transfer", "whole-model replacement"}
    price = artifact["price"]
    scope_ok = (
        price["stored_fit_scalars"] == 0
        and price["stored_fit_vectors"] == 0
        and price["selected_mlp_terms"] == 2
        and price["selected_attention5_heads"] == 4
        and price["selected_attention9_heads"] == 2
        and price["selected_source_positions"] == 3
        and price["requires_checkpoint_weights"] is True
        and price["requires_paired_base_and_donor_states"] is True
        and required_exclusions == set(artifact["scope"]["not_licensed"])
        and "failed native capability" in artifact["scope"]["failed_boundary"]
    )
    predictions = {
        "pred_a_hash_bound_authority": authority_ok,
        "pred_b_typed_graph_exact": graph_ok,
        "pred_c_discovery_metrics_exact": discovery_ok,
        "pred_d_prospective_metrics_exact": prospective_ok,
        "pred_e_price_dependency_scope_exact": scope_ok,
    }
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_transparent_path_program_release_result_v1",
        "candidate_id": prior["candidate_id"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": sha(PRIOR),
        "authority_sha256": observed,
        "predictions": predictions,
        "program_id": artifact["program_id"],
        "classification": "paired_causal_typed_tensor_program_with_prospective_lexical_recombination_evidence_not_standalone_not_whole_model",
        "price": prior["price"],
        "terminal": terminal,
    }
    payload = managed.atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
