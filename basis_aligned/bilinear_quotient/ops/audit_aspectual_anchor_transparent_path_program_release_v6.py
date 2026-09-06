#!/usr/bin/env python3
"""Zero-forward conformance audit for sparse suffix program v6."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_bound_valid_authority pred_b_executable_inventory pred_c_synthetic_equation_conformance pred_d_suffix_evidence_exact pred_e_price_dependency_scope_exact
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

import aspectual_anchor_transparent_path_program_v5 as v5
import aspectual_anchor_transparent_path_program_v6 as program
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_transparent_path_program_release_v6.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v6_result.json"
PATHS = {
    "aspectual_anchor_transparent_path_program_v6.py": ROOT / "ops/aspectual_anchor_transparent_path_program_v6.py",
    "aspectual_anchor_transparent_path_program_v6_artifact.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v6_artifact.json",
    "aspectual_anchor_transparent_path_program_release_v5_result.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v5_result.json",
    "aspectual_anchor_sparse_suffix_missing_block_compression_split_v2_result.json": ROOT / "circuits/followups/aspectual_anchor_sparse_suffix_missing_block_compression_split_v2_result.json",
    "aspectual_anchor_suffix_block12_14_component_factorial_split_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_suffix_block12_14_component_factorial_split_v1_result.json",
    "aspectual_anchor_mlp12_14_bilinear_compression_split_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_mlp12_14_bilinear_compression_split_v1_result.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict[str, object]:
    return json.loads(PATHS[name].read_text())


def synthetic_conformance() -> tuple[bool, int]:
    rng = np.random.default_rng(2026090606)
    lb, rb, lh, rh = (rng.normal(size=7) for _ in range(4))
    terms = {
        "left_change": (lh - lb) * rb,
        "right_change": lb * (rh - rb),
        "bilinear_interaction": (lh - lb) * (rh - rb),
    }
    okay, cases = True, 0
    for width in range(4):
        for factors in itertools.combinations(program.MLP_FACTORS, width):
            expected = sum((terms[factor] for factor in factors), np.zeros_like(lb))
            okay = okay and np.allclose(program.bilinear_hidden_delta(lb, rb, lh, rh, factors=factors), expected)
            cases += 1
    weight = rng.normal(size=(5, 7))
    for boundary, factors in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY.items():
        expected = sum((terms[factor] for factor in factors), np.zeros_like(lb)) @ weight.T
        observed = program.selected_suffix_mlp_write_delta(lb, rb, lh, rh, weight, boundary=boundary)
        okay = okay and np.allclose(observed, expected)
        cases += 1
    for _ in range(10):
        initial = rng.normal(size=5)
        lambdas = {boundary: float(rng.normal()) for boundary in program.SUFFIX_BOUNDARIES}
        attention = {boundary: rng.normal(size=5) for boundary in program.SUFFIX_SOURCE_BOUNDARIES}
        states = {boundary: tuple(rng.normal(size=7) for _ in range(4)) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
        weights = {boundary: rng.normal(size=(5, 7)) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
        expected = initial
        for boundary in program.SUFFIX_BOUNDARIES:
            expected = lambdas[boundary] * expected
            if boundary in attention:
                expected = expected + attention[boundary]
            if boundary in states:
                expected = expected + program.selected_suffix_mlp_write_delta(*states[boundary], weights[boundary], boundary=boundary)
        observed = program.compiled_sparse_suffix_delta(
            initial, lambda0_by_boundary=lambdas,
            source_attention_delta_by_boundary=attention,
            mlp_states_by_boundary=states, down_weight_by_boundary=weights,
        )
        okay = okay and np.allclose(observed, expected)
        cases += 1
    rejection_calls = (
        lambda: program.selected_suffix_mlp_write_delta(lb, rb, lh, rh, weight, boundary=13),
        lambda: program.compiled_sparse_suffix_delta(np.zeros(5), lambda0_by_boundary={}, source_attention_delta_by_boundary={}, mlp_states_by_boundary={}, down_weight_by_boundary={}),
        lambda: program.compiled_sparse_suffix_delta(np.zeros(5), lambda0_by_boundary={b: 1.0 for b in program.SUFFIX_BOUNDARIES}, source_attention_delta_by_boundary={11: np.zeros(5)}, mlp_states_by_boundary={b: (lb, rb, lh, rh) for b in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}, down_weight_by_boundary={b: weight for b in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}),
        lambda: program.compiled_sparse_suffix_delta(np.zeros(5), lambda0_by_boundary={b: 1.0 for b in program.SUFFIX_BOUNDARIES}, source_attention_delta_by_boundary={b: np.zeros(5) for b in program.SUFFIX_SOURCE_BOUNDARIES}, mlp_states_by_boundary={}, down_weight_by_boundary={b: weight for b in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}),
        lambda: program.compiled_sparse_suffix_delta(np.zeros(5), lambda0_by_boundary={b: 1.0 for b in program.SUFFIX_BOUNDARIES}, source_attention_delta_by_boundary={b: np.zeros(5) for b in program.SUFFIX_SOURCE_BOUNDARIES}, mlp_states_by_boundary={b: (lb, rb, lh, rh) for b in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}, down_weight_by_boundary={}),
        lambda: program.compiled_sparse_suffix_delta(np.zeros(5), lambda0_by_boundary={b: 1.0 for b in program.SUFFIX_BOUNDARIES}, source_attention_delta_by_boundary={b: np.zeros(5) for b in program.SUFFIX_SOURCE_BOUNDARIES}, mlp_states_by_boundary={11: (lb, rb, lh), 12: (lb, rb, lh, rh), 14: (lb, rb, lh, rh), 15: (lb, rb, lh, rh)}, down_weight_by_boundary={b: weight for b in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}),
        lambda: program.bilinear_hidden_delta(lb, rb, lh, rh, factors=("left_change", "left_change")),
        lambda: program.bilinear_hidden_delta(lb, rb, lh, rh, factors=("unknown",)),
    )
    for call in rejection_calls:
        rejected = False
        try:
            call()
        except program.ProgramInputError:
            rejected = True
        okay = okay and rejected
        cases += 1
    return bool(okay and cases >= 30), cases


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "synthetic_equation_cases_min": 30}))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    prior = json.loads(PRIOR.read_text())
    observed = {name: sha(path) for name, path in PATHS.items()}
    artifact = load("aspectual_anchor_transparent_path_program_v6_artifact.json")
    parent = load("aspectual_anchor_transparent_path_program_release_v5_result.json")
    blocks = load("aspectual_anchor_sparse_suffix_missing_block_compression_split_v2_result.json")
    components = load("aspectual_anchor_suffix_block12_14_component_factorial_split_v1_result.json")
    bilinear = load("aspectual_anchor_mlp12_14_bilinear_compression_split_v1_result.json")
    pred_a = (
        observed == prior["authority"] and parent["terminal"] == "release"
        and blocks["terminal"] == components["terminal"] == bilinear["terminal"] == "screen"
        and blocks["evidence_class"] == "post_outcome_repair_replication"
        and components["evidence_class"] == "conditional_post_selection_component_resolution"
        and bilinear["evidence_class"] == "conditional_post_selection_bilinear_resolution"
        and all(all(item["predictions"].values()) for item in (parent, blocks, components, bilinear))
    )
    tree = ast.parse(PATHS["aspectual_anchor_transparent_path_program_v6.py"].read_text())
    imports = sorted(alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    inherited = ("mlp4_hidden_response", "linear_without_bias", "attention_source_term", "attention_source_delta", "suffix_attention_source_delta", "project_selected_head_deltas", "crossing_delta", "suffix_crossing_delta", "write_query_delta", "bilinear_hidden_delta")
    pred_b = (
        imports == ["annotations", "aspectual_anchor_transparent_path_program_v5"]
        and functions == {"selected_suffix_mlp_write_delta", "compiled_sparse_suffix_delta", "program_manifest"}
        and all(getattr(program, name) is getattr(v5, name) for name in inherited)
        and program.program_manifest()["compiled_suffix_boundaries"] == program.SUFFIX_BOUNDARIES
    )
    pred_c, cases = synthetic_conformance()
    recurrence = artifact["compiled_suffix_recurrence"]
    evidence = artifact["evidence"]
    pred_d = (
        recurrence["boundaries"] == list(program.SUFFIX_BOUNDARIES)
        and {int(k): tuple(v) for k, v in recurrence["mlp_factors"].items()} == program.SUFFIX_MLP_FACTORS_BY_BOUNDARY
        and evidence["block12_14_selection"]["selected_increment_fraction"] == blocks["score"]["confirmation"]["selected_increment_fraction"]
        and evidence["component_resolution"]["selected"] == components["score"]["confirmation"]["selected_components"]
        and evidence["bilinear_resolution"]["family_increments"] == bilinear["score"]["confirmation"]["selected_family_increments"]
        and artifact["excluded_authority"]["aspectual_anchor_sparse_suffix_missing_block_compression_split_v1_result.json"] == "invalid_incommensurate_control"
    )
    price = artifact["price"]
    exclusions = {"standalone native-margin prediction", "full-logit prediction", "free-form text", "new-construction transfer", "whole-model replacement", "replacement of the exact final normalization or unembedding", "prospective attribution of block12 or block14 selection"}
    pred_e = (
        price["stored_fit_scalars"] == price["stored_fit_vectors"] == 0
        and price["requires_checkpoint_weights"] is True
        and price["requires_paired_base_and_donor_states"] is True
        and price["requires_native_intervening_blocks_and_final_readout"] is False
        and price["requires_base_resid18_and_exact_final_readout"] is True
        and exclusions == set(artifact["scope"]["not_licensed"])
        and prior["price"]["model_forwards"] == prior["price"]["fit_parameters"] == 0
    )
    predictions = {"pred_a_hash_bound_valid_authority": pred_a, "pred_b_executable_inventory": pred_b, "pred_c_synthetic_equation_conformance": pred_c, "pred_d_suffix_evidence_exact": pred_d, "pred_e_price_dependency_scope_exact": pred_e}
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_transparent_path_program_release_result_v6", "candidate_id": prior["candidate_id"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": sha(PRIOR),
        "authority_sha256": observed, "program_sha256": observed["aspectual_anchor_transparent_path_program_v6.py"],
        "artifact_sha256": observed["aspectual_anchor_transparent_path_program_v6_artifact.json"], "predictions": predictions,
        "synthetic_equation_cases": cases, "classification": "valid_executable_sparse_query_recurrence_paired_causal_program_through_resid18",
        "price": prior["price"], "terminal": terminal,
    }
    payload = atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "synthetic_equation_cases": cases, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
