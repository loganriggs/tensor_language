#!/usr/bin/env python3
"""Zero-forward conformance audit for the q_is transparent program."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_bound_valid_authority pred_b_local_read_and_affine_gain_exact pred_c_rank_one_write_and_rejections_exact pred_d_manifest_price_scope_and_evidence_exact
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

import tense_auxiliary_is_was_transparent_path_program_v1 as program
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_transparent_path_program_release_v1.json"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_transparent_path_program_release_v1_result.json"
PATHS = {
    "program_v1": ROOT / "ops/tense_auxiliary_is_was_transparent_path_program_v1.py",
    "artifact_v1": ROOT / "circuits/followups/tense_auxiliary_is_was_transparent_path_program_v1_artifact.json",
    "prospective_upstream_screen": ROOT / "circuits/followups/tense_auxiliary_is_was_resid10_frozen_gain_fresh_lexicon_v2_result.json",
}
EXPECTED_PRIOR_SHA256 = "2ea2cd6e3ce7dffb476ea2d30e553380a9add8219024b80002b52a8a0a142dcb"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejected(fn) -> bool:
    try:
        fn()
    except program.ProgramInputError:
        return True
    return False


def manual_contrast(state, lm_head, direction):
    current = program.TOKEN_IDS["is"] if direction == "present_to_past" else program.TOKEN_IDS["was"]
    other = program.TOKEN_IDS["was"] if direction == "present_to_past" else program.TOKEN_IDS["is"]
    normalized = torch.nn.functional.rms_norm(state, (state.shape[-1],))
    current_raw = normalized @ lm_head.weight[current]
    other_raw = normalized @ lm_head.weight[other]
    return 30.0 * torch.tanh(current_raw / 30.0) - 30.0 * torch.tanh(other_raw / 30.0)


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "gpu_accessed": False, "model_forwards": 0, "manifest_assertions_min": 24}, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    prior = json.loads(PRIOR.read_text())
    observed = {name: sha(path) for name, path in PATHS.items()}
    artifact = json.loads(PATHS["artifact_v1"].read_text())
    upstream = json.loads(PATHS["prospective_upstream_screen"].read_text())
    pred_a = sha(PRIOR) == EXPECTED_PRIOR_SHA256 and observed == prior["authority"] and upstream["terminal"] == "screen" and all(upstream["predictions"].values())

    tree = ast.parse(PATHS["program_v1"].read_text())
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    torch.manual_seed(2026)
    lm_head = torch.nn.Linear(7, 600, bias=False)
    state = torch.randn(7)
    batch = torch.randn(3, 7)
    exact_checks = []
    for direction in sorted(program.GAIN_COEFFICIENTS):
        observed_single = program.intermediate_unembedding_contrast(state, lm_head, direction=direction)
        observed_batch = program.intermediate_unembedding_contrast(batch, lm_head, direction=direction)
        coefficients = program.GAIN_COEFFICIENTS[direction]
        expected_alpha = coefficients["intercept"] + coefficients["slope"] * observed_single
        exact_checks.extend([
            torch.allclose(observed_single, manual_contrast(state, lm_head, direction), atol=1e-6, rtol=0.0),
            torch.allclose(observed_batch, manual_contrast(batch, lm_head, direction), atol=1e-6, rtol=0.0),
            torch.allclose(program.predict_writer_gain(state, lm_head, direction=direction), expected_alpha, atol=1e-6, rtol=0.0),
        ])
    pred_b = functions == {"intermediate_unembedding_contrast", "predict_writer_gain", "upstream_writer_actuation", "program_manifest"} and all(bool(value) for value in exact_checks)

    base18 = torch.randn(7)
    q = torch.randn(7)
    q = q / q.norm()
    write_checks = []
    for direction in sorted(program.GAIN_COEFFICIENTS):
        result = program.upstream_writer_actuation(state, base18, q, lm_head, direction=direction)
        write_checks.extend([
            torch.allclose(result["patched_resid18"], base18 + result["alpha"] * q, atol=1e-6, rtol=0.0),
            torch.allclose(result["alpha"], program.predict_writer_gain(state, lm_head, direction=direction), atol=1e-6, rtol=0.0),
            torch.allclose(result["resid10_unembedding_contrast"], program.intermediate_unembedding_contrast(state, lm_head, direction=direction), atol=1e-6, rtol=0.0),
        ])
    rejection_checks = [
        rejected(lambda: program.intermediate_unembedding_contrast(state, lm_head, direction="sideways")),
        rejected(lambda: program.upstream_writer_actuation(batch, base18, q, lm_head, direction="past_to_present")),
        rejected(lambda: program.upstream_writer_actuation(state, batch, q, lm_head, direction="past_to_present")),
        rejected(lambda: program.upstream_writer_actuation(state, torch.randn(8), torch.randn(8), lm_head, direction="past_to_present")),
        rejected(lambda: program.upstream_writer_actuation(state, base18, 2.0 * q, lm_head, direction="past_to_present")),
        rejected(lambda: program.upstream_writer_actuation(state, base18, q, torch.nn.Linear(8, 600, bias=False), direction="past_to_present")),
    ]
    pred_c = all(bool(value) for value in write_checks + rejection_checks)

    manifest = program.program_manifest()
    families = upstream["score"]["families"]
    assertions = [
        manifest["program_id"] == program.PROGRAM_ID,
        manifest["read_site"] == "resid:10",
        manifest["write_site"] == "resid:18",
        manifest["fixed_token_ids"] == upstream["fixed_token_ids"] == program.TOKEN_IDS,
        manifest["coefficients"] == upstream["coefficients"] == program.GAIN_COEFFICIENTS,
        manifest["basis_sha256"] == upstream["basis_sha256"] == program.BASIS_SHA256,
        manifest["evidence_result_sha256"] == program.UPSTREAM_RESULT_SHA256 == observed["prospective_upstream_screen"],
        tuple(manifest["required_runtime_inputs"]) == ("resid10", "base_resid18", "rank1_basis", "lm_head", "direction"),
        manifest["confirmation_resid18_margin_required"] is False,
        manifest["donor_activation_required"] is False,
        manifest["row_id_or_outcome_required"] is False,
        manifest["stored_fit_scalars"] == 1156,
        artifact["interface"]["fixed_token_ids"] == program.TOKEN_IDS,
        artifact["interface"]["confirmation_resid18_margin_required"] is False,
        artifact["interface"]["donor_activation_required"] is False,
        artifact["interface"]["row_id_or_outcome_required"] is False,
        artifact["evidence"]["sha256"] == observed["prospective_upstream_screen"],
        artifact["evidence"]["A1_recovery"] == families["A1"]["mean_recovery"],
        artifact["evidence"]["A2_recovery"] == families["A2"]["mean_recovery"],
        artifact["evidence"]["P_margin_reflection"] == families["P"]["mean_margin_reflection_fraction"],
        artifact["evidence"]["C_normalized_unrelated_effect"] == families["C"]["mean_normalized_unrelated_effect"],
        artifact["evidence"]["direction_fraction"] == 1.0,
        all(families[name]["direction_fraction"] == 1.0 for name in families),
        all(cell["passed"] for cell in upstream["capability_cells"]),
        artifact["price"]["rank1_basis_scalars"] == prior["price"]["rank1_basis_scalars"] == 1152,
        artifact["price"]["gain_scalars"] == prior["price"]["gain_scalars"] == 4,
        artifact["price"]["total_stored_fit_scalars"] == prior["price"]["total_stored_fit_scalars"] == 1156,
        artifact["price"]["runtime_grid_or_root_evaluations"] == prior["price"]["runtime_grid_or_root_evaluations"] == 0,
        "joint q_has/q_is composition" in artifact["scope"]["not_licensed"],
    ]
    pred_d = len(assertions) >= 24 and all(assertions)
    predictions = {
        "pred_a_hash_bound_valid_authority": pred_a,
        "pred_b_local_read_and_affine_gain_exact": pred_b,
        "pred_c_rank_one_write_and_rejections_exact": pred_c,
        "pred_d_manifest_price_scope_and_evidence_exact": pred_d,
    }
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {
        "schema": "tense_auxiliary_is_was_transparent_path_program_release_result_v1",
        "candidate_id": prior["candidate_id"], "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": sha(PRIOR), "authority_sha256": observed,
        "program_sha256": observed["program_v1"], "artifact_sha256": observed["artifact_v1"],
        "predictions": predictions, "manifest_assertions": len(assertions),
        "classification": "valid_executable_upstream_predictive_q_is_program",
        "price": prior["price"], "terminal": terminal,
        "next_action": "preregister joint q_has/q_is composition and interference test" if terminal == "release" else "repair conformance before composition",
    }
    payload = atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "manifest_assertions": len(assertions), "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
