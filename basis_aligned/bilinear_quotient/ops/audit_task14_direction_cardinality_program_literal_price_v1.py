#!/usr/bin/env python3
"""Literal storage/compute/dependency audit of the selected Task14 program."""

# BQGATE: EXPERIMENT pred_a_exact_selected_storage pred_b_interface_compute_is_small pred_c_table_compression_is_real pred_d_native_dependency_prevents_standalone_claim pred_e_classification
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/task14_direction_cardinality_program_literal_price_v1.json"
PROTOTYPES = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_artifact_v1.json"
READERS = ROOT / "circuits/fast_screens/task14_mlp6_7_fixed_direction_reader_artifact_v2_result.json"
PROGRAM_RESULT = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json"
PROGRAM_RUNNER = ROOT / "ops/run_task14_mlp6_7_direction_cardinality_prototype_causal_validation.py"
FACADE = ROOT.parent / "polynomial_causal/bilin18_observed_model_facade.py"
MODEL_SOURCE = ROOT.parent.parent / "jacclust/tt_model.py"
OUT = ROOT / "circuits/followups/task14_direction_cardinality_program_literal_price_v1_result.json"

CANDIDATE_ID = "subject_verb.number_agreement.direction_cardinality_program_literal_price_v1"
EXPECTED_SHA256 = {
    PRIOR: "fee5f8ab95312977018c529dc34a02131db16e52ca940c1f8911d8a86bca36d7",
    PROTOTYPES: "cce00d8f2309b0f9e0329c094238505819f2cf8a21c04e276af2085e068c1d07",
    READERS: "9db4eefe16498cb65fb9c21ea3f2475c790c89ebb2e65a70e8ad6b7886f2ae57",
    PROGRAM_RESULT: "9a488259efdb85477f35c612696368e8a3e372338a11253615a7b53650c88fe0",
    PROGRAM_RUNNER: "8b4c4c645cf333f26cf3a81669d36ca5d952c21704aa637089bae98adfa849a4",
    FACADE: "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    MODEL_SOURCE: "49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2",
}
D_MODEL = 1152
D_HIDDEN = 4 * D_MODEL
DIRECTIONS = ("plural_to_singular", "singular_to_plural")
CARDINALITIES = tuple(range(5))


class PriceAuditError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_and_validate() -> tuple[dict, dict, dict]:
    for path, expected in EXPECTED_SHA256.items():
        if _sha256(path) != expected:
            raise PriceAuditError(f"immutable source changed: {path}")
    prototypes = json.loads(PROTOTYPES.read_text())
    readers = json.loads(READERS.read_text())
    program = json.loads(PROGRAM_RESULT.read_text())
    if prototypes.get("terminal") != "prototype_artifact":
        raise PriceAuditError("prototype artifact is not licensed")
    if readers.get("terminal") != "reader_artifact":
        raise PriceAuditError("reader artifact is not licensed")
    if program.get("terminal") != "valid_causal_screen" or not all(program["score"]["predictions"].values()):
        raise PriceAuditError("program validation is not a passing parent")
    return prototypes, readers, program


def compile_plan() -> dict:
    _load_and_validate()
    return {
        "schema": "task14_direction_cardinality_program_literal_price_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "prior_art_sha256": EXPECTED_SHA256[PRIOR],
        "inputs": {os.path.relpath(path, ROOT): digest for path, digest in EXPECTED_SHA256.items()},
        "counting_convention": "one scalar multiplication or addition is one arithmetic operation; selection/indexing is reported separately",
        "price": {"gpu_model_forwards": 0, "cpu_model_forwards": 0, "backwards": 0, "parameter_updates": 0},
    }


def evaluate() -> dict:
    prototypes, readers, _ = _load_and_validate()
    selected_keys = [f"{direction}.cardinality_{cardinality}" for direction in DIRECTIONS for cardinality in CARDINALITIES]
    control_keys = [f"{direction}.direction_only" for direction in DIRECTIONS]
    observed_keys = set(prototypes["prototypes"])
    selected_lengths = {key: len(prototypes["prototypes"][key]["coordinates"]) for key in selected_keys}
    control_lengths = {key: len(prototypes["prototypes"][key]["coordinates"]) for key in control_keys}
    reader_lengths = {key: len(readers["readers"][key]["coordinates"]) for key in DIRECTIONS}

    selected_vector_scalars = sum(selected_lengths.values())
    control_vector_scalars = sum(control_lengths.values())
    reader_scalars = sum(reader_lengths.values())
    executable_scalars = selected_vector_scalars + reader_scalars
    exhaustive_vector_scalars = len(DIRECTIONS) * (2 ** 4) * D_MODEL
    exhaustive_total_scalars = exhaustive_vector_scalars + reader_scalars

    selected_additions = D_MODEL
    reader_multiplications = D_MODEL
    reader_additions = D_MODEL - 1
    interface_arithmetic = selected_additions + reader_multiplications + reader_additions

    one_mlp_parameters = 3 * D_MODEL * D_HIDDEN + D_MODEL
    two_mlp_parameters = 2 * one_mlp_parameters
    one_mlp_scalar_arithmetic = (
        2 * D_HIDDEN * (2 * D_MODEL - 1)
        + D_HIDDEN
        + D_MODEL * (2 * D_HIDDEN - 1)
        + D_MODEL
    )
    attention_parameters = 6 * D_MODEL * D_MODEL + 1
    block_parameters = attention_parameters + one_mlp_parameters + 2
    full_model_parameters = 2 * 50_304 * D_MODEL + 18 * block_parameters

    runner_source = PROGRAM_RUNNER.read_text()
    facade_source = FACADE.read_text()
    model_source = MODEL_SOURCE.read_text()
    dependency_witnesses = {
        "direction_supplied_by_authority_row": 'direction = row["direction_id"]' in runner_source,
        "cardinality_supplied_by_intervention_subset": "len(subset)" in runner_source,
        "counterfactual_role_batch_required": "_role_batch(rows" in runner_source,
        "native_context_decomposition_required": "_decomposed_forward(model" in runner_source,
        "native_head_function_required": "_head_function(model" in runner_source,
        "native_suffix_logits_required": "replacement_heads=" in runner_source and "logits" in runner_source,
        "full_pinned_model_loaded": "load_bilin18" in facade_source and '"n_layer": 18' in facade_source,
        "model_has_native_18_block_loop": "for block in self.transformer.h" in model_source,
    }
    exact_storage = (
        observed_keys == set(selected_keys + control_keys)
        and set(selected_lengths.values()) == {D_MODEL}
        and set(control_lengths.values()) == {D_MODEL}
        and set(reader_lengths.values()) == {D_MODEL}
        and selected_vector_scalars == 11_520
        and reader_scalars == 2_304
        and executable_scalars == 13_824
        and control_vector_scalars == 2_304
    )
    interface_small = selected_additions == 1_152 and interface_arithmetic == 3_455
    compression_real = (
        exhaustive_total_scalars == 39_168
        and abs(1 - executable_scalars / exhaustive_total_scalars - 0.6470588235294118) < 1e-15
        and abs(1 - selected_vector_scalars / exhaustive_vector_scalars - 0.6875) < 1e-15
    )
    native_dependent = all(dependency_witnesses.values())
    predictions = {
        "pred_a_exact_selected_storage": exact_storage,
        "pred_b_interface_compute_is_small": interface_small,
        "pred_c_table_compression_is_real": compression_real,
        "pred_d_native_dependency_prevents_standalone_claim": native_dependent,
        "pred_e_classification": exact_storage and interface_small and compression_real and native_dependent,
    }
    return {
        "artifact_accounting": {
            "selected_vector_count": len(selected_keys),
            "selected_vector_scalars": selected_vector_scalars,
            "reader_count": len(reader_lengths),
            "reader_scalars": reader_scalars,
            "selected_executable_scalars": executable_scalars,
            "selected_executable_fp32_bytes": executable_scalars * 4,
            "excluded_direction_only_control_count": len(control_keys),
            "excluded_direction_only_control_scalars": control_vector_scalars,
            "full_artifact_including_controls_scalars": executable_scalars + control_vector_scalars,
        },
        "interface_runtime": {
            "causal_execution_vector_additions": selected_additions,
            "optional_reader_multiplications": reader_multiplications,
            "optional_reader_additions": reader_additions,
            "causal_plus_reader_scalar_arithmetic": interface_arithmetic,
            "selector_table_lookups": 1,
        },
        "exhaustive_table_comparison": {
            "direction_by_subset_vector_count": 32,
            "vector_scalars": exhaustive_vector_scalars,
            "total_with_same_readers_scalars": exhaustive_total_scalars,
            "selected_total_storage_reduction_fraction": 1 - executable_scalars / exhaustive_total_scalars,
            "selected_vector_storage_reduction_fraction": 1 - selected_vector_scalars / exhaustive_vector_scalars,
        },
        "native_reference": {
            "one_bilinear_mlp_parameters": one_mlp_parameters,
            "two_mlp6_7_parameters": two_mlp_parameters,
            "one_bilinear_mlp_scalar_arithmetic_per_token": one_mlp_scalar_arithmetic,
            "two_mlp6_7_scalar_arithmetic_per_token": 2 * one_mlp_scalar_arithmetic,
            "full_loaded_model_parameters": full_model_parameters,
            "native_blocks_eliminated_by_current_harness": 0,
            "native_parameters_eliminated_by_current_harness": 0,
            "dependency_witnesses": dependency_witnesses,
        },
        "classification": "interface_simple_not_end_to_end" if all(predictions.values()) else "unlicensed",
        "correction": "The prior 23:35 reviews counted all twelve prototype vectors as upstream program state. Two are direction-only controls: selected vectors are 11,520 scalars; adding two readers yields 13,824 selected scalars.",
        "predictions": predictions,
        "terminal": "screen" if all(predictions.values()) else "invalid",
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise PriceAuditError(f"refusing overwrite {OUT}")
    score = evaluate()
    payload = managed.atomic_create_json(OUT, {
        "schema": "task14_direction_cardinality_program_literal_price_result_v1",
        "candidate_id": CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan,
        "score": score,
        "terminal": score["terminal"],
    })
    print(json.dumps({"terminal": score["terminal"], "predictions": score["predictions"], "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
