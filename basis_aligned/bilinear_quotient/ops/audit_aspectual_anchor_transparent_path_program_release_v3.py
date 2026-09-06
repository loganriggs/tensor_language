#!/usr/bin/env python3
"""Zero-forward conformance audit for source-resolved aspectual program v3."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_bound_authority pred_b_executable_inventory pred_c_synthetic_equation_conformance pred_d_suffix_evidence_exact pred_e_price_dependency_scope_exact
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

import aspectual_anchor_transparent_path_program_v3 as program
import circuit_fast_screen_managed_runner as managed


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_transparent_path_program_release_v3.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v3_result.json"
PATHS = {
    "aspectual_anchor_transparent_path_program_v3.py": ROOT / "ops/aspectual_anchor_transparent_path_program_v3.py",
    "aspectual_anchor_transparent_path_program_v3_artifact.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v3_artifact.json",
    "aspectual_anchor_transparent_path_program_release_v2_result.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v2_result.json",
    "aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1_result.json",
    "aspectual_anchor_block15_crossing_confirmation_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_block15_crossing_confirmation_v1_result.json",
    "aspectual_anchor_attention11_15_single_head_confirmation_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_attention11_15_single_head_confirmation_v1_result.json",
    "aspectual_anchor_attention11h3_15h5_source_projection_query_diagnostic_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_attention11h3_15h5_source_projection_query_diagnostic_v1_result.json",
    "aspectual_anchor_attention11h3_15h5_source_compression_release_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_attention11h3_15h5_source_compression_release_v1_result.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict[str, object]:
    return json.loads(PATHS[name].read_text())


def synthetic_conformance() -> tuple[bool, int]:
    rng = np.random.default_rng(20260906)
    cases = 0
    okay = True
    left_base, right_base, left_donor, right_donor = (
        rng.normal(size=(3, 5)) for _ in range(4)
    )
    down_weight = rng.normal(size=(4, 5))
    expected_hidden = (
        (left_donor - left_base) * right_base
        + left_base * (right_donor - right_base)
    )
    okay = okay and np.array_equal(
        program.mlp4_hidden_response(left_base, right_base, left_donor, right_donor),
        expected_hidden,
    )
    cases += 1
    okay = okay and np.allclose(
        program.mlp4_write(left_base, right_base, left_donor, right_donor, down_weight),
        expected_hidden @ down_weight.T,
    )
    cases += 1

    base_pattern = rng.normal(size=(9, 7, 7))
    hybrid_pattern = rng.normal(size=(9, 7, 7))
    base_value = rng.normal(size=(7, 9, 4))
    hybrid_value = rng.normal(size=(7, 9, 4))
    early_sources = (1, 2, 3)
    for heads in (program.ATTENTION5_HEADS, program.ATTENTION9_HEADS):
        observed = program.attention_source_delta(
            base_pattern, base_value, hybrid_pattern, hybrid_value,
            query=6, source_positions=early_sources, heads=heads,
        )
        for index, head in enumerate(heads):
            expected = sum(
                hybrid_pattern[head, 6, source] * hybrid_value[source, head]
                - base_pattern[head, 6, source] * base_value[source, head]
                for source in early_sources
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
            expected = sum((terms[factor] for factor in factors), np.zeros_like(arrays[0]))
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

    role_positions = {
        "cue": (0,), "last": (1,), "period": (2,), "determiner": (3,),
        "self": (6,), "other": (4, 5),
    }
    suffix_deltas = {}
    projection_weight = rng.normal(size=(5, 36))
    for boundary in (11, 15):
        head, observed = program.suffix_attention_source_delta(
            base_pattern, base_value, hybrid_pattern, hybrid_value,
            boundary=boundary, query=6, role_positions=role_positions,
        )
        expected = sum(
            hybrid_pattern[head, 6, source] * hybrid_value[source, head]
            - base_pattern[head, 6, source] * base_value[source, head]
            for role in program.SUFFIX_SOURCE_BANK_BY_BOUNDARY[boundary]
            for source in role_positions[role]
        )
        okay = okay and head == program.SUFFIX_HEAD_BY_BOUNDARY[boundary] and np.allclose(observed, expected)
        suffix_deltas[boundary] = (head, observed)
        cases += 1
        zero = np.zeros(36)
        projected = program.project_selected_head_deltas(((head, observed),), zero, projection_weight)
        expected_flat = zero.copy()
        expected_flat[head * 4:(head + 1) * 4] = observed
        okay = okay and np.allclose(projected, expected_flat @ projection_weight.T) and not np.any(zero)
        cases += 1

    projected = program.project_selected_head_deltas((suffix_deltas[11],), np.zeros(36), projection_weight)
    observed_suffix = program.suffix_crossing_delta(
        0.73, arrays[0][:5], arrays[1][:5], projected, arrays[4][:5], arrays[5][:5]
    )
    expected_suffix = 0.73 * (arrays[1][:5] - arrays[0][:5]) + projected + arrays[5][:5] - arrays[4][:5]
    okay = okay and np.allclose(observed_suffix, expected_suffix)
    cases += 1

    rejection_calls = (
        lambda: program.attention_source_term(base_pattern, base_value, -1, 6, 1),
        lambda: program.attention_source_delta(base_pattern, base_value, hybrid_pattern, hybrid_value, query=6, source_positions=(1, 2), heads=program.ATTENTION5_HEADS),
        lambda: program.attention_source_delta(base_pattern, base_value, hybrid_pattern, hybrid_value, query=6, source_positions=(1, 1, 3), heads=program.ATTENTION5_HEADS),
        lambda: program.attention_source_delta(base_pattern, base_value, hybrid_pattern, hybrid_value, query=6, source_positions=early_sources, heads=(0,)),
        lambda: program.suffix_attention_source_delta(base_pattern, base_value, hybrid_pattern, hybrid_value, boundary=12, query=6, role_positions=role_positions),
        lambda: program.suffix_attention_source_delta(base_pattern, base_value, hybrid_pattern, hybrid_value, boundary=11, query=6, role_positions={key: value for key, value in role_positions.items() if key != "cue"}),
        lambda: program.suffix_attention_source_delta(base_pattern, base_value, hybrid_pattern, hybrid_value, boundary=11, query=6, role_positions={**role_positions, "other": (3, 4, 5)}),
        lambda: program.project_selected_head_deltas((), np.zeros(36), projection_weight),
        lambda: program.project_selected_head_deltas(((3, np.ones(4)), (3, np.ones(4))), np.zeros(36), projection_weight),
        lambda: program.crossing_delta(0.73, *arrays, factors=("carried", "carried")),
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
    return bool(okay and cases >= 30), cases


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "synthetic_equation_cases_min": 30}))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    prior = json.loads(PRIOR.read_text())
    observed_hashes = {name: sha(path) for name, path in PATHS.items()}
    artifact = load("aspectual_anchor_transparent_path_program_v3_artifact.json")
    v2 = load("aspectual_anchor_transparent_path_program_release_v2_result.json")
    block11 = load("aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1_result.json")
    block15 = load("aspectual_anchor_block15_crossing_confirmation_v1_result.json")
    singleton = load("aspectual_anchor_attention11_15_single_head_confirmation_v1_result.json")
    diagnostic = load("aspectual_anchor_attention11h3_15h5_source_projection_query_diagnostic_v1_result.json")
    source_release = load("aspectual_anchor_attention11h3_15h5_source_compression_release_v1_result.json")

    authority_ok = (
        observed_hashes == prior["authority"]
        and [v2["terminal"], block11["terminal"], block15["terminal"], singleton["terminal"], diagnostic["terminal"], source_release["terminal"]]
        == ["release", "screen", "screen", "screen", "screen", "release"]
        and all(all(result["predictions"].values()) for result in (v2, block11, block15, singleton, diagnostic, source_release))
    )
    tree = ast.parse(PATHS["aspectual_anchor_transparent_path_program_v3.py"].read_text())
    imports = sorted(
        alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    )
    public_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    manifest = program.program_manifest()
    inventory_ok = (
        imports == ["annotations"]
        and public_functions == {
            "mlp4_hidden_response", "linear_without_bias", "mlp4_write",
            "attention_source_term", "attention_source_delta", "suffix_attention_source_delta",
            "project_selected_head_deltas", "crossing_delta", "suffix_crossing_delta",
            "write_query_delta", "program_manifest",
        }
        and manifest["program_id"] == artifact["program_id"]
        and list(manifest["attention5_heads"]) == artifact["frozen_inventory"]["attention5_heads"]
        and list(manifest["attention9_heads"]) == artifact["frozen_inventory"]["attention9_heads"]
        and {str(key): value for key, value in manifest["suffix_head_by_boundary"].items()} == artifact["frozen_inventory"]["suffix_head_by_boundary"]
        and {str(key): list(value) for key, value in manifest["suffix_source_bank_by_boundary"].items()} == artifact["frozen_inventory"]["suffix_source_bank_by_boundary"]
    )
    equations_ok, equation_cases = synthetic_conformance()

    evidence = artifact["suffix_evidence"]
    evidence_ok = (
        evidence["block11"]["crossing_shapley"] == block11["score"]["confirmation_shapley"]
        and evidence["block15"]["crossing_shapley"] == block15["score"]["factorial_shapley_target_recovery"]
        and evidence["block11"]["dominant_head"] == singleton["score"]["boundaries"]["11"]["head"]
        and evidence["block15"]["dominant_head"] == singleton["score"]["boundaries"]["15"]["head"]
        and evidence["block11"]["dominant_head_to_four_fraction"] == singleton["score"]["boundaries"]["11"]["singleton_to_four_fraction"]
        and evidence["block15"]["dominant_head_to_four_fraction"] == singleton["score"]["boundaries"]["15"]["singleton_to_four_fraction"]
        and evidence["block11"]["source_bank"] == source_release["released_banks"]["11"]["source_roles"]
        and evidence["block15"]["source_bank"] == source_release["released_banks"]["15"]["source_roles"]
        and evidence["block11"]["source_bank_to_all_dominant_head_fraction"] == source_release["released_banks"]["11"]["confirmation_retained_fraction"]
        and evidence["block15"]["source_bank_to_all_dominant_head_fraction"] == source_release["released_banks"]["15"]["confirmation_retained_fraction"]
        and evidence["block11"]["query_projection_max_abs"] == diagnostic["score"]["query_projection_max_abs"]["11"]
        and evidence["block15"]["query_projection_max_abs"] == diagnostic["score"]["query_projection_max_abs"]["15"]
        and evidence["evidence_class"] == source_release["evidence_class"]
    )
    price = artifact["price"]
    exclusions = {
        "standalone native-margin prediction", "full-logit prediction", "free-form text",
        "new-construction transfer", "whole-model replacement",
        "source-resolved replacement of carried or MLP block11/block15 terms",
        "transparent replacement of native blocks10,12-14,16-17 or the final readout",
    }
    scope_ok = (
        price["stored_fit_scalars"] == price["stored_fit_vectors"] == 0
        and price["requires_checkpoint_weights"] is True
        and price["requires_paired_base_and_donor_states"] is True
        and price["requires_full_carried_and_mlp_suffix_deltas"] is True
        and price["requires_native_checkpoint_suffix_after_block15"] is True
        and exclusions == set(artifact["scope"]["not_licensed"])
        and list(manifest["runtime_dependencies"]) == [
            "checkpoint weights", "paired base/donor MLP4 states",
            "paired base/hybrid attention captures",
            "paired base/hybrid suffix residual and MLP deltas", "native checkpoint suffix",
        ]
    )
    predictions = {
        "pred_a_hash_bound_authority": authority_ok,
        "pred_b_executable_inventory": inventory_ok,
        "pred_c_synthetic_equation_conformance": equations_ok,
        "pred_d_suffix_evidence_exact": evidence_ok,
        "pred_e_price_dependency_scope_exact": scope_ok,
    }
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_transparent_path_program_release_result_v3",
        "candidate_id": prior["candidate_id"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": sha(PRIOR), "authority_sha256": observed_hashes,
        "program_sha256": observed_hashes["aspectual_anchor_transparent_path_program_v3.py"],
        "artifact_sha256": observed_hashes["aspectual_anchor_transparent_path_program_v3_artifact.json"],
        "predictions": predictions, "synthetic_equation_cases": equation_cases,
        "classification": "executable_source_resolved_paired_causal_tensor_program_through_block15_with_native_dependencies",
        "evidence_class": evidence["evidence_class"], "price": prior["price"],
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
