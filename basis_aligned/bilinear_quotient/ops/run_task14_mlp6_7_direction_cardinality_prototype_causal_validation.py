#!/usr/bin/env python3
"""Prospectively validate the fixed direction-cardinality upstream program."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_artifacts pred_b_reader_predicts_installed_prototype_effect pred_c_prototype_substitutes_native_displacement pred_d_intermediate_and_each_template_transfer pred_e_cardinality_beats_direction_only pred_f_target_free_program_and_price
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_fast_screen_candidate_task14_cardinality_prototype_transfer as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_cardinality_prototype_transfer_native_capability as capability
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_fixed_direction_cardinality_upstream_program_v1.json"
PROTOTYPES = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_artifact_v1.json"
PREDICTION = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_predictions_v1.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json"
PRIOR_ART_SHA256 = "075c1f83f5801e2eb874d6df55b6070d56a6a0271716dd15e99d044e4f2c2f2d"
PROTOTYPE_SHA256 = "cce00d8f2309b0f9e0329c094238505819f2cf8a21c04e276af2085e068c1d07"
PREDICTION_SHA256 = "dfc4949687f8cf53f4399691442153492832adbb05eb24cbbfacc9e19698aff1"
CAPABILITY_RESULT_SHA256 = "b5db9ccd55b8e244458cdbeada7246fb45eaa95c6abc19cdeeba1f7bd41e6a1c"
CAPABILITY_LICENSE_SHA256 = "c595bd0edf7e92b659f3d209836bec0c6d68524b0255c120d7651f71923f5af1"
SUBSETS = factor_gate.BACKGROUND_SUBSETS
METHODS = ("base", "exact", "cardinality", "direction_only")
PATCH_CHUNK_ROWS = 256
MAXIMUM_ERROR = 5e-5
BARS = {
    "minimum_reader_cosine": 0.90,
    "maximum_reader_relative_l2_error": 0.45,
    "minimum_reader_sign_agreement": 0.85,
    "minimum_native_cosine": 0.75,
    "maximum_native_relative_l2_error": 0.75,
    "minimum_native_sign_agreement": 0.75,
    "minimum_intermediate_cosine": 0.70,
    "maximum_intermediate_relative_l2_error": 0.85,
    "minimum_intermediate_sign_agreement": 0.70,
    "minimum_template_cosine": 0.65,
    "maximum_template_relative_l2_error": 0.90,
    "minimum_template_sign_agreement": 0.65,
    "minimum_sse_reduction_over_direction_only": 0.10,
}
PRED_KEYS = (
    "pred_a_authority_capability_and_artifacts",
    "pred_b_reader_predicts_installed_prototype_effect",
    "pred_c_prototype_substitutes_native_displacement",
    "pred_d_intermediate_and_each_template_transfer",
    "pred_e_cardinality_beats_direction_only",
    "pred_f_target_free_program_and_price",
)


class CausalValidationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_price(row_count: int = 32) -> dict[str, int]:
    installs = row_count * len(SUBSETS) * len(METHODS)
    chunks = math.ceil(installs / PATCH_CHUNK_ROWS)
    return {
        "physical_model_forwards": 1 + chunks,
        "example_evaluations": row_count * len(authority.ROLES) + installs,
        "causal_installations": installs,
        "backwards": 0, "parameter_updates": 0,
        "maximum_patch_chunk_rows": PATCH_CHUNK_ROWS, "patch_chunks": chunks,
    }


def _load_artifacts() -> tuple[dict[str, object], dict[str, object]]:
    if _sha256(PROTOTYPES) != PROTOTYPE_SHA256:
        raise CausalValidationError("prototype artifact changed")
    if _sha256(PREDICTION) != PREDICTION_SHA256:
        raise CausalValidationError("prediction seal changed")
    prototypes = json.loads(PROTOTYPES.read_text())
    prediction = json.loads(PREDICTION.read_text())
    if prototypes.get("terminal") != "prototype_artifact" or not all(prototypes.get("predictions", {}).values()):
        raise CausalValidationError("prototype artifact invalid")
    if prediction.get("terminal") != "sealed_prediction" or prediction.get("causal_outcomes_opened") is not False or prediction.get("target_exact_displacements_consumed") != 0 or not all(prediction.get("predictions", {}).values()):
        raise CausalValidationError("prediction seal invalid")
    return prototypes, prediction


def validate_preflight() -> None:
    for path, expected, label in (
        (PRIOR_ART, PRIOR_ART_SHA256, "prior art"),
        (capability.RESULT, CAPABILITY_RESULT_SHA256, "capability result"),
        (capability.LICENSE, CAPABILITY_LICENSE_SHA256, "capability license"),
    ):
        if _sha256(path) != expected:
            raise CausalValidationError(f"{label} changed")
    licensing.validate_causal_preflight(
        capability.build_gate(), capability.RESULT, capability.LICENSE,
        expected_license_sha256=CAPABILITY_LICENSE_SHA256,
        causal_candidate_id=authority.CAUSAL_CANDIDATE_ID,
    )
    _load_artifacts()
    if derive_price() != {
        "physical_model_forwards": 9, "example_evaluations": 2144,
        "causal_installations": 2048, "backwards": 0, "parameter_updates": 0,
        "maximum_patch_chunk_rows": 256, "patch_chunks": 8,
    }:
        raise CausalValidationError("price changed")


def compile_plan() -> dict[str, object]:
    _, prediction = _load_artifacts()
    validate_preflight()
    return {
        "schema": "task14_mlp6_7_direction_cardinality_prototype_causal_validation_plan_v1",
        "candidate_id": authority.CAUSAL_CANDIDATE_ID,
        "split": "PROSPECTIVE_THIRD_CORPUS_COMPLETE_CAUSAL_LATTICE",
        "row_count": 32, "background_subsets": list(SUBSETS), "methods": list(METHODS),
        "prior_art_sha256": PRIOR_ART_SHA256,
        "prototype_artifact_sha256": PROTOTYPE_SHA256,
        "sealed_prediction_sha256": PREDICTION_SHA256,
        "sealed_prediction_created_utc": prediction["created_utc"],
        "capability_license_sha256": CAPABILITY_LICENSE_SHA256,
        "prototype_construction_reads_target_exact_displacement": False,
        "prototype_construction_reads_target_causal_outcome": False,
        "literal_scorer": "no fitted scale, offset, target grouping, or post-gate repair",
        "bars": dict(BARS), "price": derive_price(),
    }


def _compile_patch(tokens, heads, rows, torch):
    indices, replacements, specs = [], [], []
    for row_index, row in enumerate(rows):
        for subset in SUBSETS:
            for method in METHODS:
                indices.append(row_index)
                replacements.append(heads[(row_index, subset, method)])
                specs.append((row_index, subset, method, row["direction_id"], row["template_id"]))
    index = torch.tensor(indices, dtype=torch.long, device=tokens.device)
    return {
        "tokens": tokens[:len(rows)][index],
        "finals": torch.full_like(index, tangent.parent.SUBJECT_POSITION),
        "replacement_heads": torch.stack(replacements),
        "native_reinstall_mask": torch.zeros(len(specs), dtype=torch.bool, device=tokens.device),
        "specs": specs,
    }


def evaluate(model, torch, F, facade):
    prototype_artifact, _ = _load_artifacts()
    rows = authority.build_rows()
    count = len(rows)
    parent = tangent.parent
    device = next(model.parameters()).device
    tokens, finals = parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    _, captured, projection, role_closure, inputs = parent._decomposed_forward(model, tokens, finals, torch, F, facade)
    roles = {
        "recipient": tangent._role_slice(captured, 0, count),
        "opposite": tangent._role_slice(captured, count, 2 * count),
    }
    input_roles = {
        "recipient": tangent._role_slice(inputs, 0, count),
        "opposite": tangent._role_slice(inputs, count, 2 * count),
    }
    function = tangent._head_function(model, roles["recipient"], roles["opposite"], model.transformer.h[parent.LAYER].attn, projection, torch, F)
    prototype_vectors = {
        key: torch.tensor(value["coordinates"], dtype=torch.float32, device=device)
        for key, value in prototype_artifact["prototypes"].items()
    }
    heads = {}
    with torch.no_grad():
        for subset in SUBSETS:
            base = function(factor_gate._raw_for(input_roles["recipient"], input_roles["opposite"], subset, F)).detach()
            exact = function(factor_gate._raw_for(input_roles["recipient"], input_roles["opposite"], subset + "YZ", F)).detach()
            for index, row in enumerate(rows):
                direction = row["direction_id"]
                heads[(index, subset, "base")] = base[index]
                heads[(index, subset, "exact")] = exact[index]
                heads[(index, subset, "cardinality")] = base[index] + prototype_vectors[f"{direction}.cardinality_{len(subset)}"]
                heads[(index, subset, "direction_only")] = base[index] + prototype_vectors[f"{direction}.direction_only"]
        patch = _compile_patch(tokens, heads, rows, torch)
        margins, closures = {}, []
        for start in range(0, len(patch["specs"]), PATCH_CHUNK_ROWS):
            stop = min(start + PATCH_CHUNK_ROWS, len(patch["specs"]))
            logits, _, _, closure = parent.downstream._decomposed_forward(
                model, patch["tokens"][start:stop], patch["finals"][start:stop], torch, F, facade,
                replacement_heads=patch["replacement_heads"][start:stop],
                native_reinstall_mask=patch["native_reinstall_mask"][start:stop],
            )
            closures.append(closure)
            for local, spec in enumerate(patch["specs"][start:stop]):
                row_index, subset, method, _, _ = spec
                endpoint = rows[row_index]["endpoints"]["opposite_same_lemma"]
                margins[(row_index, subset, method)] = float(
                    logits[local, parent.SUBJECT_POSITION, endpoint["answer_id"]]
                    - logits[local, parent.SUBJECT_POSITION, endpoint["foil_id"]]
                )
    evidence = []
    for index, row in enumerate(rows):
        for subset in SUBSETS:
            base_margin = margins[(index, subset, "base")]
            evidence.append({
                "row_id": row["row_id"], "direction": row["direction_id"],
                "template": row["template_id"], "background": subset,
                "cardinality": len(subset),
                "native_exact_q": margins[(index, subset, "exact")] - base_margin,
                "cardinality_prototype_q": margins[(index, subset, "cardinality")] - base_margin,
                "direction_only_prototype_q": margins[(index, subset, "direction_only")] - base_margin,
            })
    exactness = {
        "role_state_closure_max_absolute_error": role_closure["input_state_closure_max_absolute_error"],
        "role_normalized_closure_max_absolute_error": role_closure["input_normalized_closure_max_absolute_error"],
        "downstream_state_closure_max_absolute_error": max(item["state_sum_max_absolute_error"] for item in closures),
        "downstream_normalized_closure_max_absolute_error": max(item["normalized_state_max_absolute_error"] for item in closures),
    }
    return evidence, exactness


def _stats(items, actual_field, prediction_field):
    actual = [float(item[actual_field]) for item in items]
    predicted = [float(item[prediction_field]) for item in items]
    dot = sum(x * y for x, y in zip(actual, predicted))
    actual_norm = math.sqrt(sum(x * x for x in actual))
    predicted_norm = math.sqrt(sum(x * x for x in predicted))
    return {
        "count": len(items), "cosine": dot / max(actual_norm * predicted_norm, 1e-30),
        "relative_l2_error": math.sqrt(sum((x - y) ** 2 for x, y in zip(actual, predicted))) / max(actual_norm, 1e-30),
        "sign_agreement": sum((x > 0) == (y > 0) for x, y in zip(actual, predicted)) / len(items),
        "sse": sum((x - y) ** 2 for x, y in zip(actual, predicted)),
    }


def _passes(stats, minimum_cosine, maximum_error, minimum_sign):
    return stats["cosine"] >= minimum_cosine and stats["relative_l2_error"] <= maximum_error and stats["sign_agreement"] >= minimum_sign


def score(causal, exactness, bars=BARS):
    _, sealed = _load_artifacts()
    predictions = {(item["row_id"], item["background"]): item for item in sealed["evidence"]}
    joined = []
    for item in causal:
        prediction = predictions.get((item["row_id"], item["background"]))
        if prediction is None:
            raise CausalValidationError("missing sealed prediction")
        joined.append({
            **item,
            "sealed_cardinality_reader_q": prediction["cardinality_reader_q"],
            "sealed_direction_only_reader_q": prediction["direction_only_reader_q"],
        })
    reader = _stats(joined, "cardinality_prototype_q", "sealed_cardinality_reader_q")
    native_cardinality = _stats(joined, "native_exact_q", "cardinality_prototype_q")
    native_direction_only = _stats(joined, "native_exact_q", "direction_only_prototype_q")
    intermediate = _stats(
        [item for item in joined if item["background"] not in {"", "EAUW"}],
        "native_exact_q", "cardinality_prototype_q",
    )
    templates = {
        template: _stats([item for item in joined if item["template"] == template], "native_exact_q", "cardinality_prototype_q")
        for template in ("near_beyond", "beyond_near")
    }
    reduction = 1.0 - native_cardinality["sse"] / max(native_direction_only["sse"], 1e-30)
    instrument = len(joined) == 512 and len({(x["row_id"], x["background"]) for x in joined}) == 512 and all(value <= MAXIMUM_ERROR for value in exactness.values())
    pred_b = _passes(reader, bars["minimum_reader_cosine"], bars["maximum_reader_relative_l2_error"], bars["minimum_reader_sign_agreement"])
    pred_c = _passes(native_cardinality, bars["minimum_native_cosine"], bars["maximum_native_relative_l2_error"], bars["minimum_native_sign_agreement"])
    pred_d = _passes(intermediate, bars["minimum_intermediate_cosine"], bars["maximum_intermediate_relative_l2_error"], bars["minimum_intermediate_sign_agreement"]) and all(
        _passes(item, bars["minimum_template_cosine"], bars["maximum_template_relative_l2_error"], bars["minimum_template_sign_agreement"])
        for item in templates.values()
    )
    pred_f = derive_price()["causal_installations"] == 2048 and sealed["target_exact_displacements_consumed"] == 0 and sealed["causal_outcomes_opened"] is False
    verdicts = (instrument, instrument and pred_b, instrument and pred_c, instrument and pred_d, instrument and reduction >= bars["minimum_sse_reduction_over_direction_only"], instrument and pred_f)
    return {
        **exactness,
        "reader_prediction_of_installed_cardinality_effect": reader,
        "native_substitution": {"cardinality": native_cardinality, "direction_only": native_direction_only},
        "intermediate_only_native_substitution": intermediate,
        "by_template_native_substitution": templates,
        "sse_reduction_over_direction_only": reduction,
        "provenance": {"prototype_construction_reads_target_exact_displacement": False, "prototype_construction_reads_target_causal_outcome": False, "fit_operations": 0},
        "predictions": dict(zip(PRED_KEYS, map(bool, verdicts))),
        "joined_evidence": joined,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise CausalValidationError(f"refusing overwrite {OUT}")
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    causal, exactness = evaluate(model, torch, F, facade)
    scored = score(causal, exactness)
    terminal = "valid_causal_screen" if scored["predictions"][PRED_KEYS[0]] else "invalid"
    payload = managed.atomic_create_json(OUT, {
        "schema": "task14_mlp6_7_direction_cardinality_prototype_causal_validation_result_v1",
        "candidate_id": authority.CAUSAL_CANDIDATE_ID, "terminal": terminal,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "score": scored, "causal_evidence": causal, "sealed_prediction_sha256": PREDICTION_SHA256,
    })
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"], "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
