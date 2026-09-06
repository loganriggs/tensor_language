#!/usr/bin/env python3
"""Zero-forward conformance audit for source- and MLP-resolved program v4."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_bound_authority pred_b_executable_inventory pred_c_synthetic_equation_conformance pred_d_suffix_mlp_evidence_exact pred_e_price_dependency_scope_exact
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

import aspectual_anchor_transparent_path_program_v3 as v3
import aspectual_anchor_transparent_path_program_v4 as program
import circuit_fast_screen_managed_runner as managed


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_transparent_path_program_release_v4.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v4_result.json"
PATHS = {
    "aspectual_anchor_transparent_path_program_v4.py": ROOT / "ops/aspectual_anchor_transparent_path_program_v4.py",
    "aspectual_anchor_transparent_path_program_v4_artifact.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v4_artifact.json",
    "aspectual_anchor_transparent_path_program_release_v3_result.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v3_result.json",
    "aspectual_anchor_mlp11_15_bilinear_compression_split_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_mlp11_15_bilinear_compression_split_v1_result.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict[str, object]:
    return json.loads(PATHS[name].read_text())


def synthetic_conformance() -> tuple[bool, int]:
    rng = np.random.default_rng(20260906)
    left_base, right_base, left_hybrid, right_hybrid = (
        rng.normal(size=6) for _ in range(4)
    )
    terms = {
        "left_change": (left_hybrid - left_base) * right_base,
        "right_change": left_base * (right_hybrid - right_base),
        "bilinear_interaction": (left_hybrid - left_base) * (right_hybrid - right_base),
    }
    okay, cases = True, 0
    for width in range(4):
        for factors in itertools.combinations(program.MLP_FACTORS, width):
            observed = program.bilinear_hidden_delta(
                left_base, right_base, left_hybrid, right_hybrid, factors=factors
            )
            expected = sum((terms[factor] for factor in factors), np.zeros_like(left_base))
            okay = okay and np.allclose(observed, expected)
            cases += 1
    down_weight = rng.normal(size=(5, 6))
    projected_attention = rng.normal(size=5)
    base_resid, hybrid_resid = rng.normal(size=5), rng.normal(size=5)
    for boundary in (11, 15):
        hidden = sum(
            (terms[factor] for factor in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY[boundary]),
            np.zeros_like(left_base),
        )
        write = program.suffix_mlp_write_delta(
            left_base, right_base, left_hybrid, right_hybrid,
            down_weight, boundary=boundary,
        )
        okay = okay and np.allclose(write, hidden @ down_weight.T)
        cases += 1
        crossing = program.compiled_suffix_crossing_delta(
            0.71, base_resid, hybrid_resid, projected_attention,
            left_base, right_base, left_hybrid, right_hybrid,
            down_weight, boundary=boundary,
        )
        expected = 0.71 * (hybrid_resid - base_resid) + projected_attention + write
        okay = okay and np.allclose(crossing, expected)
        cases += 1

    base_pattern = rng.normal(size=(9, 7, 7))
    hybrid_pattern = rng.normal(size=(9, 7, 7))
    base_value = rng.normal(size=(7, 9, 4))
    hybrid_value = rng.normal(size=(7, 9, 4))
    roles = {"cue": (0,), "last": (1,), "period": (2,), "determiner": (3,), "self": (6,), "other": (4, 5)}
    for boundary in (11, 15):
        head, delta = program.suffix_attention_source_delta(
            base_pattern, base_value, hybrid_pattern, hybrid_value,
            boundary=boundary, query=6, role_positions=roles,
        )
        okay = okay and head == program.SUFFIX_HEAD_BY_BOUNDARY[boundary] and len(delta) == 4
        cases += 1
    state = rng.normal(size=(7, 5))
    delta = rng.normal(size=5)
    changed = program.write_query_delta(state, 6, delta)
    okay = okay and np.allclose(changed[6], state[6] + delta) and not np.shares_memory(changed, state)
    cases += 1

    rejection_calls = (
        lambda: program.bilinear_hidden_delta(left_base, right_base, left_hybrid, right_hybrid, factors=("left_change", "left_change")),
        lambda: program.bilinear_hidden_delta(left_base, right_base, left_hybrid, right_hybrid, factors=("unknown",)),
        lambda: program.suffix_mlp_write_delta(left_base, right_base, left_hybrid, right_hybrid, down_weight, boundary=12),
        lambda: program.attention_source_term(base_pattern, base_value, -1, 6, 1),
        lambda: program.attention_source_delta(base_pattern, base_value, hybrid_pattern, hybrid_value, query=6, source_positions=(1, 2), heads=program.ATTENTION5_HEADS),
        lambda: program.suffix_attention_source_delta(base_pattern, base_value, hybrid_pattern, hybrid_value, boundary=12, query=6, role_positions=roles),
        lambda: program.suffix_attention_source_delta(base_pattern, base_value, hybrid_pattern, hybrid_value, boundary=11, query=6, role_positions={key: value for key, value in roles.items() if key != "cue"}),
        lambda: program.project_selected_head_deltas((), np.zeros(36), rng.normal(size=(5, 36))),
        lambda: program.crossing_delta(0.71, base_resid, hybrid_resid, base_resid, hybrid_resid, base_resid, hybrid_resid, factors=("mlp", "mlp")),
        lambda: program.write_query_delta(state, -1, delta),
    )
    for call in rejection_calls:
        rejected = False
        try:
            call()
        except program.ProgramInputError:
            rejected = True
        okay = okay and rejected
        cases += 1
    return bool(okay and cases >= 25), cases


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "synthetic_equation_cases_min": 25}))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    prior = json.loads(PRIOR.read_text())
    observed_hashes = {name: sha(path) for name, path in PATHS.items()}
    artifact = load("aspectual_anchor_transparent_path_program_v4_artifact.json")
    v3_release = load("aspectual_anchor_transparent_path_program_release_v3_result.json")
    mlp = load("aspectual_anchor_mlp11_15_bilinear_compression_split_v1_result.json")
    authority_ok = (
        observed_hashes == prior["authority"]
        and v3_release["terminal"] == "release" and mlp["terminal"] == "screen"
        and all(v3_release["predictions"].values()) and all(mlp["predictions"].values())
    )
    tree = ast.parse(PATHS["aspectual_anchor_transparent_path_program_v4.py"].read_text())
    imports = sorted(
        alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    )
    public_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    inherited = (
        "mlp4_hidden_response", "linear_without_bias", "mlp4_write", "attention_source_term",
        "attention_source_delta", "suffix_attention_source_delta", "project_selected_head_deltas",
        "crossing_delta", "suffix_crossing_delta", "write_query_delta",
    )
    inventory_ok = (
        imports == ["annotations", "aspectual_anchor_transparent_path_program_v3"]
        and public_functions == {
            "bilinear_hidden_delta", "suffix_mlp_write_delta",
            "compiled_suffix_crossing_delta", "program_manifest",
        }
        and all(getattr(program, name) is getattr(v3, name) for name in inherited)
        and program.program_manifest()["program_id"] == artifact["program_id"]
    )
    equations_ok, equation_cases = synthetic_conformance()
    evidence = artifact["frozen_suffix_program"]
    evidence_ok = True
    for boundary in ("11", "15"):
        compiled = evidence[f"block{boundary}"]
        confirmation = mlp["score"]["confirmation"][boundary]
        evidence_ok = evidence_ok and (
            compiled["mlp_factors"] == confirmation["selected_factors"]
            and compiled["mlp_selected_to_all_fraction"] == confirmation["selected_to_all_mlp_fraction"]
            and compiled["mlp_family_increments"] == confirmation["selected_family_increments"]
            and compiled["mlp_factors"] == list(program.SUFFIX_MLP_FACTORS_BY_BOUNDARY[int(boundary)])
        )
    price = artifact["price"]
    exclusions = {
        "standalone native-margin prediction", "full-logit prediction", "free-form text",
        "new-construction transfer", "whole-model replacement",
        "transparent replacement of native blocks10,12-14,16-17 or the final readout",
    }
    manifest = program.program_manifest()
    scope_ok = (
        price["stored_fit_scalars"] == price["stored_fit_vectors"] == 0
        and price["requires_checkpoint_weights"] is True
        and price["requires_paired_base_and_donor_states"] is True
        and price["requires_native_intervening_blocks_and_final_readout"] is True
        and exclusions == set(artifact["scope"]["not_licensed"])
        and list(manifest["runtime_dependencies"]) == [
            "checkpoint weights", "paired base/donor MLP4 states",
            "paired base/hybrid attention captures",
            "paired base/hybrid suffix residual and bilinear MLP states",
            "native checkpoint suffix",
        ]
    )
    predictions = {
        "pred_a_hash_bound_authority": authority_ok,
        "pred_b_executable_inventory": inventory_ok,
        "pred_c_synthetic_equation_conformance": equations_ok,
        "pred_d_suffix_mlp_evidence_exact": evidence_ok,
        "pred_e_price_dependency_scope_exact": scope_ok,
    }
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_transparent_path_program_release_result_v4",
        "candidate_id": prior["candidate_id"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": sha(PRIOR), "authority_sha256": observed_hashes,
        "program_sha256": observed_hashes["aspectual_anchor_transparent_path_program_v4.py"],
        "artifact_sha256": observed_hashes["aspectual_anchor_transparent_path_program_v4_artifact.json"],
        "predictions": predictions, "synthetic_equation_cases": equation_cases,
        "classification": "executable_source_and_mlp_resolved_paired_causal_tensor_program_through_block15_with_native_dependencies",
        "price": prior["price"], "terminal": terminal,
    }
    payload = managed.atomic_create_json(OUT, value)
    print(json.dumps({
        "terminal": terminal, "predictions": predictions,
        "synthetic_equation_cases": equation_cases,
        "result_sha256": hashlib.sha256(payload).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
