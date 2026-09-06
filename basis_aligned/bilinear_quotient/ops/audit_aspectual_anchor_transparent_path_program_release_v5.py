#!/usr/bin/env python3
"""Zero-forward conformance audit for corrected executable aspectual program v5."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_bound_valid_authority pred_b_executable_inventory pred_c_synthetic_equation_conformance pred_d_corrected_suffix_mlp_evidence_exact pred_e_price_dependency_scope_exact
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
import aspectual_anchor_transparent_path_program_v5 as program
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_transparent_path_program_release_v5.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v5_result.json"
PATHS = {
    "aspectual_anchor_transparent_path_program_v5.py": ROOT / "ops/aspectual_anchor_transparent_path_program_v5.py",
    "aspectual_anchor_transparent_path_program_v5_artifact.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v5_artifact.json",
    "aspectual_anchor_transparent_path_program_release_v3_result.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v3_result.json",
    "aspectual_anchor_mlp11_15_bilinear_compression_split_v2_result.json": ROOT / "circuits/followups/aspectual_anchor_mlp11_15_bilinear_compression_split_v2_result.json",
    "aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit_v2_result.json": ROOT / "circuits/followups/aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit_v2_result.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict[str, object]:
    return json.loads(PATHS[name].read_text())


def synthetic_conformance() -> tuple[bool, int]:
    rng = np.random.default_rng(20260906)
    lb, rb, lh, rh = (rng.normal(size=6) for _ in range(4))
    terms = {
        "left_change": (lh - lb) * rb,
        "right_change": lb * (rh - rb),
        "bilinear_interaction": (lh - lb) * (rh - rb),
    }
    okay, cases = True, 0
    for width in range(4):
        for factors in itertools.combinations(program.MLP_FACTORS, width):
            expected = sum((terms[factor] for factor in factors), np.zeros_like(lb))
            observed = program.bilinear_hidden_delta(lb, rb, lh, rh, factors=factors)
            okay = okay and np.allclose(observed, expected)
            cases += 1
    weight = rng.normal(size=(5, 6))
    attention = rng.normal(size=5)
    base_resid, hybrid_resid = rng.normal(size=5), rng.normal(size=5)
    for boundary in (11, 15):
        expected_hidden = sum(
            (terms[factor] for factor in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY[boundary]),
            np.zeros_like(lb),
        )
        write = program.suffix_mlp_write_delta(lb, rb, lh, rh, weight, boundary=boundary)
        okay = okay and np.allclose(write, expected_hidden @ weight.T)
        cases += 1
        crossing = program.compiled_suffix_crossing_delta(
            0.71, base_resid, hybrid_resid, attention, lb, rb, lh, rh, weight,
            boundary=boundary,
        )
        okay = okay and np.allclose(crossing, 0.71 * (hybrid_resid - base_resid) + attention + write)
        cases += 1
    hidden4 = program.mlp4_hidden_response(lb, rb, lh, rh)
    okay = okay and np.allclose(hidden4, terms["left_change"] + terms["right_change"])
    cases += 1
    state = rng.normal(size=(7, 5))
    delta = rng.normal(size=5)
    changed = program.write_query_delta(state, 6, delta)
    okay = okay and np.allclose(changed[6], state[6] + delta) and not np.shares_memory(changed, state)
    cases += 1
    zero_crossing = program.crossing_delta(
        0.71, base_resid, hybrid_resid, base_resid, hybrid_resid,
        base_resid, hybrid_resid, factors=(),
    )
    okay = okay and np.array_equal(zero_crossing, np.zeros_like(base_resid))
    cases += 1
    rejection_calls = (
        lambda: program.bilinear_hidden_delta(lb, rb, lh, rh, factors=("left_change", "left_change")),
        lambda: program.bilinear_hidden_delta(lb, rb, lh, rh, factors=("unknown",)),
        lambda: program.suffix_mlp_write_delta(lb, rb, lh, rh, weight, boundary=12),
        lambda: program.attention_source_term(np.zeros((9, 7, 7)), np.zeros((7, 9, 4)), -1, 6, 1),
        lambda: program.attention_source_delta(np.zeros((9, 7, 7)), np.zeros((7, 9, 4)), np.zeros((9, 7, 7)), np.zeros((7, 9, 4)), query=6, source_positions=(1, 2), heads=program.ATTENTION5_HEADS),
        lambda: program.project_selected_head_deltas((), np.zeros(36), np.zeros((5, 36))),
        lambda: program.crossing_delta(0.71, base_resid, hybrid_resid, base_resid, hybrid_resid, base_resid, hybrid_resid, factors=("mlp", "mlp")),
        lambda: program.write_query_delta(state, -1, delta),
        lambda: program.suffix_attention_source_delta(np.zeros((9, 7, 7)), np.zeros((7, 9, 4)), np.zeros((9, 7, 7)), np.zeros((7, 9, 4)), boundary=12, query=6, role_positions={}),
        lambda: program.project_selected_head_deltas(((9, np.ones(4)),), np.zeros(36), np.zeros((5, 36))),
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
    observed = {name: sha(path) for name, path in PATHS.items()}
    artifact = load("aspectual_anchor_transparent_path_program_v5_artifact.json")
    v3_release = load("aspectual_anchor_transparent_path_program_release_v3_result.json")
    mlp = load("aspectual_anchor_mlp11_15_bilinear_compression_split_v2_result.json")
    design_audit = load("aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit_v2_result.json")
    pred_a = (
        observed == prior["authority"] and v3_release["terminal"] == "release"
        and mlp["terminal"] == design_audit["terminal"] == "screen"
        and design_audit["scientific_disposition"] == "v1_superseded_as_invalid"
        and all(v3_release["predictions"].values()) and all(mlp["predictions"].values())
        and all(design_audit["predictions"].values())
    )
    tree = ast.parse(PATHS["aspectual_anchor_transparent_path_program_v5.py"].read_text())
    imports = sorted(
        alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    )
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    inherited = (
        "mlp4_hidden_response", "linear_without_bias", "mlp4_write", "attention_source_term",
        "attention_source_delta", "suffix_attention_source_delta", "project_selected_head_deltas",
        "crossing_delta", "suffix_crossing_delta", "write_query_delta",
    )
    pred_b = (
        imports == ["annotations", "aspectual_anchor_transparent_path_program_v3"]
        and functions == {"bilinear_hidden_delta", "suffix_mlp_write_delta", "compiled_suffix_crossing_delta", "program_manifest"}
        and all(getattr(program, name) is getattr(v3, name) for name in inherited)
        and program.program_manifest()["suffix_mlp_evidence"] == "corrected three-role-context split v2"
    )
    pred_c, cases = synthetic_conformance()
    evidence = artifact["frozen_suffix_program"]
    pred_d = artifact["excluded_authority"] == {
        "aspectual_anchor_mlp11_15_bilinear_compression_split_v1_result.json": "superseded_invalid_design_mismatch",
        "aspectual_anchor_transparent_path_program_release_v4_result.json": "superseded_invalid_parent",
    }
    for boundary in ("11", "15"):
        compiled = evidence[f"block{boundary}"]
        actual = mlp["score"]["confirmation"][boundary]
        pred_d = pred_d and (
            compiled["mlp_factors"] == actual["selected_factors"]
            and compiled["mlp_selected_to_all_fraction"] == actual["selected_to_all_mlp_fraction"]
            and compiled["mlp_family_increments"] == actual["selected_family_increments"]
            and compiled["mlp_factors"] == list(program.SUFFIX_MLP_FACTORS_BY_BOUNDARY[int(boundary)])
        )
    price = artifact["price"]
    exclusions = {
        "standalone native-margin prediction", "full-logit prediction", "free-form text",
        "new-construction transfer", "whole-model replacement",
        "transparent replacement of native blocks10,12-14,16-17 or the final readout",
    }
    pred_e = (
        price["stored_fit_scalars"] == price["stored_fit_vectors"] == 0
        and price["requires_checkpoint_weights"] is True
        and price["requires_paired_base_and_donor_states"] is True
        and price["requires_native_intervening_blocks_and_final_readout"] is True
        and exclusions == set(artifact["scope"]["not_licensed"])
        and prior["price"]["model_forwards"] == prior["price"]["fit_parameters"] == 0
    )
    predictions = {
        "pred_a_hash_bound_valid_authority": pred_a,
        "pred_b_executable_inventory": pred_b,
        "pred_c_synthetic_equation_conformance": pred_c,
        "pred_d_corrected_suffix_mlp_evidence_exact": pred_d,
        "pred_e_price_dependency_scope_exact": pred_e,
    }
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_transparent_path_program_release_result_v5",
        "candidate_id": prior["candidate_id"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": sha(PRIOR), "authority_sha256": observed,
        "program_sha256": observed["aspectual_anchor_transparent_path_program_v5.py"],
        "artifact_sha256": observed["aspectual_anchor_transparent_path_program_v5_artifact.json"],
        "predictions": predictions, "synthetic_equation_cases": cases,
        "classification": "valid_executable_source_and_mlp_resolved_paired_causal_program_through_block15",
        "price": prior["price"], "terminal": terminal,
    }
    payload = atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "synthetic_equation_cases": cases, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
