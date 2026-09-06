#!/usr/bin/env python3
"""Zero-forward scope and identity audit for transferred program v8."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_bound_valid_authority pred_b_executable_identity pred_c_transfer_evidence_exact pred_d_manifest_scope_exact pred_e_price_dependency_scope_exact
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import aspectual_anchor_transparent_path_program_v7 as v7
import aspectual_anchor_transparent_path_program_v8 as program
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_transparent_path_program_release_v8.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v8_result.json"
PATHS = {
    "aspectual_anchor_transparent_path_program_v8.py": ROOT / "ops/aspectual_anchor_transparent_path_program_v8.py",
    "aspectual_anchor_transparent_path_program_v8_artifact.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v8_artifact.json",
    "aspectual_anchor_transparent_path_program_release_v7_result.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v7_result.json",
    "aspectual_anchor_program_v7_fresh_construction_transfer_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_program_v7_fresh_construction_transfer_v1_result.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict[str, object]:
    return json.loads(PATHS[name].read_text())


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "model_forwards": 0, "example_evaluations": 0}))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    prior = json.loads(PRIOR.read_text())
    observed = {name: sha(path) for name, path in PATHS.items()}
    artifact = load("aspectual_anchor_transparent_path_program_v8_artifact.json")
    parent = load("aspectual_anchor_transparent_path_program_release_v7_result.json")
    transfer = load("aspectual_anchor_program_v7_fresh_construction_transfer_v1_result.json")
    pred_a = observed == prior["authority"] and parent["terminal"] == "release" and transfer["terminal"] == "screen" and all(parent["predictions"].values()) and all(transfer["predictions"].values())
    tree = ast.parse(PATHS["aspectual_anchor_transparent_path_program_v8.py"].read_text())
    imports = sorted(alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    inherited = ("compiled_sparse_suffix_delta", "selected_suffix_mlp_write_delta", "bilinear_hidden_delta", "exact_final_logits", "exact_scored_pair", "compiled_sparse_suffix_scored_pair")
    pred_b = imports == ["annotations", "aspectual_anchor_transparent_path_program_v7"] and functions == {"program_manifest"} and all(getattr(program, name) is getattr(v7, name) for name in inherited)
    evidence = artifact["prospective_transfer"]
    pred_c = (
        evidence["constructions"] == list(program.TRANSFER_CONSTRUCTIONS)
        and evidence["target_rows"] == transfer["score"]["program_v7"]["A1"]["count"] + transfer["score"]["program_v7"]["A2"]["count"]
        and evidence["direction_fraction"] == {family: transfer["score"]["program_v7"][family]["direction_fraction"] for family in ("A1", "A2")}
        and evidence["writer_retention"] == transfer["score"]["program_to_writer_retention"]
    )
    manifest = program.program_manifest()
    exclusions = {"standalone native-margin prediction", "free-form text", "whole-model replacement", "prospective attribution of block12 or block14 selection", "empirical fidelity of non-answer/foil vocabulary logits", "unrestricted syntax transfer"}
    pred_d = manifest["prospective_transfer_constructions"] == program.TRANSFER_CONSTRUCTIONS and manifest["prospective_transfer_target_rows"] == 32 and exclusions == set(artifact["scope"]["not_licensed"])
    price = artifact["price"]
    pred_e = price["stored_fit_scalars"] == price["stored_fit_vectors"] == 0 and price["requires_checkpoint_weights"] is True and price["requires_paired_base_and_donor_states"] is True and price["requires_native_intervening_blocks_or_final_readout"] is False and price["requires_base_resid18"] is True and prior["price"]["model_forwards"] == prior["price"]["fit_parameters"] == 0
    predictions = {"pred_a_hash_bound_valid_authority": pred_a, "pred_b_executable_identity": pred_b, "pred_c_transfer_evidence_exact": pred_c, "pred_d_manifest_scope_exact": pred_d, "pred_e_price_dependency_scope_exact": pred_e}
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_transparent_path_program_release_result_v8", "candidate_id": prior["candidate_id"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": sha(PRIOR),
        "authority_sha256": observed, "program_sha256": observed["aspectual_anchor_transparent_path_program_v8.py"],
        "artifact_sha256": observed["aspectual_anchor_transparent_path_program_v8_artifact.json"], "predictions": predictions,
        "classification": "valid_executable_prospectively_transferred_paired_causal_scored_logit_program", "price": prior["price"], "terminal": terminal,
    }
    payload = atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
