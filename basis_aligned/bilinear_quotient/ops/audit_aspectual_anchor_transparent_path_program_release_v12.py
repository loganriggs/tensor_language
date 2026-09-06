#!/usr/bin/env python3
"""Zero-forward conformance audit for upstream predictive program v12."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_bound_valid_authority pred_b_parent_operations_preserved pred_c_local_read_and_affine_gain_exact pred_d_rank_one_write_and_rejections_exact pred_e_manifest_price_scope_and_evidence_exact
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

import aspectual_anchor_transparent_path_program_v11 as v11
import aspectual_anchor_transparent_path_program_v12 as program
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_transparent_path_program_release_v12.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v12_result.json"
PATHS = {
    "program_v12": ROOT / "ops/aspectual_anchor_transparent_path_program_v12.py",
    "artifact_v12": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v12_artifact.json",
    "program_v11_release": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v11_result.json",
    "upstream_prospective_screen": ROOT / "circuits/followups/aspectual_anchor_resid10_frozen_gain_fresh_lexicon_v2_result.json",
}
EXPECTED_PRIOR_SHA256 = "78a4f30792d2525d11e307027e3ccbd5fdc03645404b8952fa1397fe5770e1d3"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejected(fn) -> bool:
    try:
        fn()
    except program.ProgramInputError:
        return True
    return False


def manual_contrast(state, lm_head, direction):
    current = program.TOKEN_IDS["has"] if direction == "present_to_past" else program.TOKEN_IDS["had"]
    other = program.TOKEN_IDS["had"] if direction == "present_to_past" else program.TOKEN_IDS["has"]
    normalized = torch.nn.functional.rms_norm(state, (state.shape[-1],))
    current_raw = normalized @ lm_head.weight[current]
    other_raw = normalized @ lm_head.weight[other]
    return 30.0 * torch.tanh(current_raw / 30.0) - 30.0 * torch.tanh(other_raw / 30.0)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "manifest_assertions_min": 28}))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    prior = json.loads(PRIOR.read_text())
    observed = {name: sha(path) for name, path in PATHS.items()}
    loaded = {name: json.loads(path.read_text()) for name, path in PATHS.items() if name != "program_v12"}
    upstream = loaded["upstream_prospective_screen"]
    pred_a = (
        sha(PRIOR) == EXPECTED_PRIOR_SHA256
        and observed == prior["authority"]
        and loaded["program_v11_release"]["terminal"] == "release"
        and upstream["terminal"] == "screen"
        and all(upstream["predictions"].values())
    )

    tree = ast.parse(PATHS["program_v12"].read_text())
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    aliases = (
        "carrier_amplitude", "rank1_carrier_projection", "compiled_sparse_suffix_delta",
        "exact_final_logits", "exact_scored_pair", "exact_selected_margin",
        "donor_free_margin_reflection", "operational_quotient_manifest",
    )
    pred_b = functions == {"intermediate_unembedding_contrast", "predict_carrier_gain", "upstream_carrier_actuation", "program_manifest"} and all(getattr(program, name) is getattr(v11, name) for name in aliases)

    torch.manual_seed(1729)
    head = torch.nn.Linear(7, 600, bias=False)
    state = torch.randn(7)
    batch = torch.randn(3, 7)
    local_checks = []
    affine_checks = []
    for direction in sorted(program.GAIN_COEFFICIENTS):
        observed_single = program.intermediate_unembedding_contrast(state, head, direction=direction)
        observed_batch = program.intermediate_unembedding_contrast(batch, head, direction=direction)
        local_checks.extend([
            torch.allclose(observed_single, manual_contrast(state, head, direction), atol=1e-6, rtol=0.0),
            torch.allclose(observed_batch, manual_contrast(batch, head, direction), atol=1e-6, rtol=0.0),
        ])
        coefficients = program.GAIN_COEFFICIENTS[direction]
        expected_alpha = coefficients["intercept"] + coefficients["slope"] * observed_single
        affine_checks.append(torch.allclose(program.predict_carrier_gain(state, head, direction=direction), expected_alpha, atol=1e-6, rtol=0.0))
    pred_c = all(bool(x) for x in local_checks + affine_checks)

    base18 = torch.randn(7)
    q = torch.randn(7)
    q = q / q.norm()
    writes = []
    for direction in sorted(program.GAIN_COEFFICIENTS):
        result = program.upstream_carrier_actuation(state, base18, q, head, direction=direction)
        writes.extend([
            torch.allclose(result["patched_resid18"], base18 + result["alpha"] * q, atol=1e-6, rtol=0.0),
            torch.allclose(result["alpha"], program.predict_carrier_gain(state, head, direction=direction), atol=1e-6, rtol=0.0),
            torch.allclose(result["resid10_unembedding_contrast"], program.intermediate_unembedding_contrast(state, head, direction=direction), atol=1e-6, rtol=0.0),
        ])
    rejection_checks = [
        rejected(lambda: program.intermediate_unembedding_contrast(state, head, direction="sideways")),
        rejected(lambda: program.upstream_carrier_actuation(batch, base18, q, head, direction="past_to_present")),
        rejected(lambda: program.upstream_carrier_actuation(state, batch, q, head, direction="past_to_present")),
        rejected(lambda: program.upstream_carrier_actuation(state, torch.randn(8), torch.randn(8), head, direction="past_to_present")),
        rejected(lambda: program.upstream_carrier_actuation(state, base18, 2.0 * q, head, direction="past_to_present")),
        rejected(lambda: program.upstream_carrier_actuation(state, base18, q, torch.nn.Linear(8, 600, bias=False), direction="past_to_present")),
    ]
    pred_d = all(bool(x) for x in writes) and all(rejection_checks)

    manifest = program.program_manifest()
    actuator = manifest["upstream_predictive_actuator"]
    confirmation = manifest["upstream_predictive_confirmation"]
    artifact = loaded["artifact_v12"]
    families = upstream["score"]["families"]
    exclusions = {"raw-text-to-resid10 computation", "unrestricted syntax", "different surface readouts", "weight-free upstream identification", "whole-model replacement"}
    assertions = [
        manifest["program_id"] == program.PROGRAM_ID,
        manifest["operational_quotient"] == v11.operational_quotient_manifest(),
        manifest["donor_free_actuator"] == v11.program_manifest()["donor_free_actuator"],
        manifest["rank1_basis_sha256"] == v11.program_manifest()["rank1_basis_sha256"],
        actuator["read_site"] == "resid:10",
        actuator["write_site"] == "resid:18",
        actuator["fixed_token_ids"] == upstream["fixed_token_ids"] == program.TOKEN_IDS,
        actuator["coefficients"] == upstream["coefficients"] == program.GAIN_COEFFICIENTS,
        actuator["basis_sha256"] == upstream["basis_sha256"],
        actuator["evidence_result_sha256"] == program.UPSTREAM_RESULT_SHA256 == observed["upstream_prospective_screen"],
        tuple(actuator["required_runtime_inputs"]) == ("resid10", "base_resid18", "rank1_basis", "lm_head", "direction"),
        actuator["confirmation_resid18_margin_required"] is False,
        actuator["donor_activation_required"] is False,
        actuator["row_outcome_ids_required"] is False,
        confirmation["A1_recovery"] == families["A1"]["mean_recovery"],
        confirmation["A2_recovery"] == families["A2"]["mean_recovery"],
        confirmation["P_margin_reflection"] == families["P"]["mean_margin_reflection_fraction"],
        confirmation["C_normalized_unrelated_effect"] == families["C"]["mean_normalized_unrelated_effect"],
        confirmation["direction_fraction"] == 1.0,
        all(families[name]["direction_fraction"] == 1.0 for name in families),
        all(cell["passed"] for cell in upstream["capability_cells"]),
        manifest["preferred_actuator_within_tested_scope"] == "upstream_carrier_actuation",
        manifest["stored_fit_scalars"] == 1157,
        artifact["price"]["parent_stored_fit_scalars"] == 1153,
        artifact["price"]["additional_stored_fit_scalars"] == 4,
        artifact["price"]["total_stored_fit_scalars"] == 1157,
        artifact["price"]["runtime_grid_evaluations"] == 0,
        artifact["interface"]["confirmation_resid18_margin_required"] is False,
        artifact["interface"]["donor_activation_required"] is False,
        artifact["interface"]["row_outcome_ids_required"] is False,
        set(artifact["scope"]["not_licensed"]) == exclusions,
        prior["price"]["additional_stored_fit_scalars"] == 4,
        prior["price"]["total_stored_fit_scalars"] == 1157,
        prior["price"]["runtime_grid_evaluations"] == 0,
    ]
    pred_e = len(assertions) >= 28 and all(assertions)
    predictions = {
        "pred_a_hash_bound_valid_authority": pred_a,
        "pred_b_parent_operations_preserved": pred_b,
        "pred_c_local_read_and_affine_gain_exact": pred_c,
        "pred_d_rank_one_write_and_rejections_exact": pred_d,
        "pred_e_manifest_price_scope_and_evidence_exact": pred_e,
    }
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_transparent_path_program_release_result_v12",
        "candidate_id": prior["candidate_id"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": sha(PRIOR),
        "authority_sha256": observed,
        "program_sha256": observed["program_v12"],
        "artifact_sha256": observed["artifact_v12"],
        "predictions": predictions,
        "manifest_assertions": len(assertions),
        "classification": "valid_executable_upstream_predictive_operational_quotient_aspectual_program",
        "price": prior["price"],
        "terminal": terminal,
    }
    payload = atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "manifest_assertions": len(assertions), "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
