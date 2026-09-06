#!/usr/bin/env python3
"""Zero-forward conformance audit for the executable aspectual path program v2."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_bound_authority pred_b_executable_inventory pred_c_synthetic_equation_conformance pred_d_prospective_evidence_exact pred_e_price_dependency_scope_exact
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import aspectual_anchor_transparent_path_program_v2 as program
import circuit_fast_screen_managed_runner as managed


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_transparent_path_program_release_v2.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v2_result.json"
PATHS = {
    "aspectual_anchor_transparent_path_program_v2.py": ROOT / "ops/aspectual_anchor_transparent_path_program_v2.py",
    "aspectual_anchor_transparent_path_program_v2_artifact.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v2_artifact.json",
    "aspectual_anchor_transparent_path_program_release_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v1_result.json",
    "aspectual_anchor_explicit_path_lexical_holdout_v2_result.json": ROOT / "circuits/followups/aspectual_anchor_explicit_path_lexical_holdout_v2_result.json",
    "aspectual_anchor_attention9_h1h4_lexical_holdout_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_attention9_h1h4_lexical_holdout_v1_result.json",
    "aspectual_anchor_blocks6_8_crossing_factorials_lexical_holdout_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_blocks6_8_crossing_factorials_lexical_holdout_v1_result.json",
    "aspectual_anchor_block9_crossing_factorial_lexical_holdout_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_block9_crossing_factorial_lexical_holdout_v1_result.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict[str, object]:
    return json.loads(PATHS[name].read_text())


def synthetic_conformance() -> tuple[bool, int]:
    rng = np.random.default_rng(20260906)
    cases = 0
    left_base, right_base, left_donor, right_donor = (
        rng.normal(size=(3, 5)) for _ in range(4)
    )
    down_weight = rng.normal(size=(4, 5))
    expected_hidden = (
        (left_donor - left_base) * right_base
        + left_base * (right_donor - right_base)
    )
    okay = np.array_equal(
        program.mlp4_hidden_response(
            left_base, right_base, left_donor, right_donor
        ),
        expected_hidden,
    )
    cases += 1
    okay = okay and np.allclose(
        program.mlp4_write(
            left_base, right_base, left_donor, right_donor, down_weight
        ),
        expected_hidden @ down_weight.T,
    )
    cases += 1

    base_pattern = rng.normal(size=(9, 7, 7))
    hybrid_pattern = rng.normal(size=(9, 7, 7))
    base_value = rng.normal(size=(7, 9, 6))
    hybrid_value = rng.normal(size=(7, 9, 6))
    sources = (2, 3, 4)
    for heads in (program.ATTENTION5_HEADS, program.ATTENTION9_HEADS):
        observed = program.attention_source_delta(
            base_pattern, base_value, hybrid_pattern, hybrid_value,
            query=6, source_positions=sources, heads=heads,
        )
        for index, head in enumerate(heads):
            expected = sum(
                hybrid_pattern[head, 6, source] * hybrid_value[source, head]
                - base_pattern[head, 6, source] * base_value[source, head]
                for source in sources
            )
            okay = okay and np.allclose(observed[index], expected)
            cases += 1

    arrays = [rng.normal(size=11) for _ in range(6)]
    allowed = ("carried", "attention", "mlp")
    for width in range(4):
        for factors in itertools.combinations(allowed, width):
            observed = program.crossing_delta(0.73, *arrays, factors=factors)
            terms = {
                "carried": 0.73 * (arrays[1] - arrays[0]),
                "attention": arrays[3] - arrays[2],
                "mlp": arrays[5] - arrays[4],
            }
            expected = sum(
                (terms[factor] for factor in factors), np.zeros_like(arrays[0])
            )
            okay = okay and np.allclose(observed, expected)
            cases += 1

    state = rng.normal(size=(7, 11))
    delta = rng.normal(size=11)
    changed = program.write_query_delta(state, 5, delta)
    okay = (
        okay and np.allclose(changed[5], state[5] + delta)
        and np.array_equal(changed[:5], state[:5])
        and np.array_equal(changed[6:], state[6:])
        and not np.shares_memory(changed, state)
    )
    cases += 1

    rejection_calls = (
        lambda: program.attention_source_term(base_pattern, base_value, -1, 6, 2),
        lambda: program.attention_source_delta(base_pattern, base_value, hybrid_pattern, hybrid_value, query=6, source_positions=(2, 3), heads=program.ATTENTION5_HEADS),
        lambda: program.attention_source_delta(base_pattern, base_value, hybrid_pattern, hybrid_value, query=6, source_positions=(2, 2, 4), heads=program.ATTENTION5_HEADS),
        lambda: program.attention_source_delta(base_pattern, base_value, hybrid_pattern, hybrid_value, query=6, source_positions=sources, heads=(0,)),
        lambda: program.crossing_delta(0.73, *arrays, factors=("carried", "carried")),
    )
    for call in rejection_calls:
        rejected = False
        try:
            call()
        except program.ProgramInputError:
            rejected = True
        okay = okay and rejected
        cases += 1
    return bool(okay and cases == 22), cases


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "synthetic_equation_cases": 22}))
        return
    prior = json.loads(PRIOR.read_text())
    observed_hashes = {name: sha(path) for name, path in PATHS.items()}
    artifact = load("aspectual_anchor_transparent_path_program_v2_artifact.json")
    v1 = load("aspectual_anchor_transparent_path_program_release_v1_result.json")
    upstream = load("aspectual_anchor_explicit_path_lexical_holdout_v2_result.json")
    attention9 = load("aspectual_anchor_attention9_h1h4_lexical_holdout_v1_result.json")
    intermediate = load("aspectual_anchor_blocks6_8_crossing_factorials_lexical_holdout_v1_result.json")
    block9 = load("aspectual_anchor_block9_crossing_factorial_lexical_holdout_v1_result.json")

    authority_ok = (
        observed_hashes == prior["authority"]
        and [v1["terminal"], upstream["terminal"], attention9["terminal"], intermediate["terminal"], block9["terminal"]]
        == ["release", "screen", "screen", "screen", "screen"]
        and all(v1["predictions"].values())
        and all(upstream["predictions"].values())
        and all(attention9["predictions"].values())
        and all(intermediate["predictions"].values())
        and all(block9["predictions"].values())
    )
    tree = ast.parse(PATHS["aspectual_anchor_transparent_path_program_v2.py"].read_text())
    imports = sorted(
        alias.name for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    )
    public_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    manifest = program.program_manifest()
    inventory_ok = (
        imports == ["annotations"]
        and public_functions == {
            "mlp4_hidden_response", "linear_without_bias", "mlp4_write",
            "attention_source_term", "attention_source_delta", "crossing_delta",
            "write_query_delta", "program_manifest",
        }
        and manifest["program_id"] == artifact["program_id"]
        and list(manifest["mlp4_terms"]) == artifact["frozen_inventory"]["mlp4_terms"]
        and list(manifest["attention5_heads"]) == artifact["frozen_inventory"]["attention5_heads"]
        and list(manifest["attention9_heads"]) == artifact["frozen_inventory"]["attention9_heads"]
        and list(manifest["source_names"]) == artifact["frozen_inventory"]["source_positions"]
        and list(manifest["crossing_boundaries"]) == artifact["frozen_inventory"]["crossing_boundaries"]
    )
    equations_ok, equation_cases = synthetic_conformance()

    evidence = artifact["prospective_evidence"]
    upstream_arms = upstream["score"]["arms"]
    intermediate_boundaries = intermediate["score"]["boundaries"]
    evidence_ok = (
        evidence["writer_mean_recovery"] == upstream_arms["writer_two_term"]["mean_target_recovery"]
        and evidence["attention5_all_head_mean_recovery"] == upstream_arms["attention5_all_nine"]["mean_target_recovery"]
        and evidence["attention5_four_head_mean_recovery"] == upstream_arms["attention5_complete_four"]["mean_target_recovery"]
        and evidence["attention5_four_head_to_all_fraction"] == upstream["score"]["four_head_to_all_nine_fraction"]
        and evidence["attention5_bank_to_four_fraction"] == upstream["score"]["bank_to_complete_fraction"]
        and {key: evidence["full_crossing_curve"][key] for key in ("resid7", "resid8", "resid9")} == intermediate["score"]["full_crossing_curve"]
        and evidence["full_crossing_curve"]["resid10"] == block9["score"]["factorial_arms"]["carried9+attention9+mlp9"]["mean_target_recovery"]
        and evidence["attention9_h1h4_mean_recovery"] == attention9["score"]["arms"]["h1h4_complete"]["mean_target_recovery"]
        and evidence["attention9_h1h4_to_writer_fraction"] == attention9["score"]["complete_h1h4_to_writer_fraction"]
        and evidence["attention9_bank_to_all_h1h4_fraction"] == attention9["score"]["bank_to_all_h1h4_fraction"]
        and evidence["resid10_to_writer_fraction"] == block9["score"]["full_crossing_to_writer_retained_fraction"]
    )
    for boundary in (6, 7, 8):
        actual = intermediate_boundaries[str(boundary)]["factorial_shapley_target_recovery"]
        compiled = evidence["crossing_shapley"][f"block{boundary}"]
        evidence_ok = evidence_ok and compiled == {
            "carried": actual[f"carried{boundary}"],
            "attention": actual[f"attention{boundary}"],
            "mlp": actual[f"mlp{boundary}"],
        }
    actual9 = block9["score"]["factorial_shapley_target_recovery"]
    evidence_ok = evidence_ok and evidence["crossing_shapley"]["block9"] == {
        "carried": actual9["carried9"],
        "attention": actual9["attention9"],
        "mlp": actual9["mlp9"],
    }

    price = artifact["price"]
    exclusions = {
        "standalone native-margin prediction", "full-logit prediction",
        "free-form text", "new-construction transfer", "whole-model replacement",
        "transparent replacement of native suffix blocks10-17",
    }
    scope_ok = (
        price["stored_fit_scalars"] == 0
        and price["stored_fit_vectors"] == 0
        and price["requires_checkpoint_weights"] is True
        and price["requires_paired_base_and_donor_states"] is True
        and price["requires_native_checkpoint_suffix_blocks10_17"] is True
        and exclusions == set(artifact["scope"]["not_licensed"])
        and list(manifest["runtime_dependencies"]) == [
            "checkpoint weights", "paired base/donor MLP4 states",
            "paired base/hybrid attention captures", "native checkpoint suffix",
        ]
        and len(artifact["interfaces"]["manipulations"]) == 4
    )
    predictions = {
        "pred_a_hash_bound_authority": authority_ok,
        "pred_b_executable_inventory": inventory_ok,
        "pred_c_synthetic_equation_conformance": equations_ok,
        "pred_d_prospective_evidence_exact": evidence_ok,
        "pred_e_price_dependency_scope_exact": scope_ok,
    }
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_transparent_path_program_release_result_v2",
        "candidate_id": prior["candidate_id"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": sha(PRIOR),
        "authority_sha256": observed_hashes,
        "program_sha256": observed_hashes["aspectual_anchor_transparent_path_program_v2.py"],
        "artifact_sha256": observed_hashes["aspectual_anchor_transparent_path_program_v2_artifact.json"],
        "predictions": predictions,
        "synthetic_equation_cases": equation_cases,
        "classification": "executable_prospectively_supported_paired_causal_tensor_program_to_resid10_not_standalone_not_whole_model",
        "price": prior["price"],
        "terminal": terminal,
    }
    payload = managed.atomic_create_json(OUT, value)
    print(json.dumps({
        "terminal": terminal, "predictions": predictions,
        "synthetic_equation_cases": equation_cases,
        "result_sha256": hashlib.sha256(payload).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
