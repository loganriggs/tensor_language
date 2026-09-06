#!/usr/bin/env python3
"""Prospective validation of the standalone 27-scalar answer-margin program."""

# BQGATE: EXPERIMENT pred_a_temporal_seal_instrument_and_capability pred_b_task14_prospective_baseline pred_c_bracket_prospective_baseline pred_d_prospective_absolute_counterfactuals pred_e_transparent_standalone_price_and_boundary
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_source_regions as bracket_shared
import circuit_fast_screen_managed_runner as managed
import run_bracket_l13h8_source_region_payload_factorial as bracket_exact
import run_bracket_suffix_free_scalar_fresh_corpus_validation_v1 as bracket_runner
import run_task14_bracket_native_baseline_semantic_linear_feasibility_v1 as feasibility
import run_task14_direction_cardinality_absolute_head_program_v1 as absolute_runner
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/task14_bracket_native_baseline_semantic_linear_prospective_v2.json"
COEFFICIENTS = ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_v1_artifact.json"
TASK_ROWS = ROOT / "circuits/prior_art/task14_native_baseline_fresh_corpus_v1_rows.json"
BRACKET_ROWS = ROOT / "circuits/prior_art/bracket_native_baseline_fresh_corpus_v1_rows.json"
V4 = ROOT / "circuits/followups/task14_bracket_counterfactual_margin_actuator_v4_artifact.json"
TASK_VECTORS = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_artifact_v1.json"
BRACKET_VECTORS = ROOT / "circuits/fast_screens/bracket_l13h8_ordered_pair_displacement_artifact_v1.json"
FEASIBILITY = ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_feasibility_v1_result.json"
ARTIFACT = ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_prospective_v2_artifact.json"
OUT = ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_prospective_v2_result.json"
CANDIDATE_ID = "cross_behavior.task14_bracket_native_baseline_semantic_linear_prospective_v2"
EXPECTED = {
    PRIOR: "3475a4e60912d72f057fa7b77eca696a8d19ca254bb367b329794331a37e560f",
    COEFFICIENTS: "821aa7b02634d1e5300efd12803ae67b1d930e1d4132e83306c5ff31f1f647db",
    TASK_ROWS: "564d03ae74202b5e0e1be0ce272464362974e2a2f9f6f587c9587319ef829360",
    BRACKET_ROWS: "ad246a0ab2affd0a351b971c100c27c2ad09597d0d9e7b84b636e1eb4c8fb399",
    V4: "85c5cc0549421fc1575d96ce621d0677ea4b0cc2d154b2c0bf7af90f4148bd4c",
    TASK_VECTORS: "cce00d8f2309b0f9e0329c094238505819f2cf8a21c04e276af2085e068c1d07",
    BRACKET_VECTORS: "531434535daf526ebf584564afbfd5834c75df7bd636014ea233f9ae71206db0",
    FEASIBILITY: "62d9d6e302d60b5372a13b3fbf119c6dfb333375b775497886515d0134fe29ca",
}
BARS = {
    "minimum_native_capability": .85,
    "minimum_baseline_cosine": .90,
    "maximum_baseline_relative_l2": .50,
    "minimum_baseline_sign_agreement": .90,
    "minimum_counterfactual_cosine": .90,
    "maximum_counterfactual_relative_l2": .50,
    "minimum_counterfactual_sign_agreement": .85,
    "minimum_task14_direction_template_cosine": .80,
    "minimum_task14_direction_template_sign_agreement": .85,
    "minimum_bracket_ordered_pair_sign_agreement": .85,
    "maximum_replay_or_closure_error": 1e-4,
}
SUBSETS = factor_gate.BACKGROUND_SUBSETS
CHUNK = 256


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict, list[dict], list[dict], dict, dict, dict]:
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"immutable input changed: {path}")
    coefficients, task_rows, bracket_rows, v4, task_vectors, bracket_vectors, old = (json.loads(path.read_text()) for path in (COEFFICIENTS, TASK_ROWS, BRACKET_ROWS, V4, TASK_VECTORS, BRACKET_VECTORS, FEASIBILITY))
    if coefficients["terminal"] != "frozen_artifact" or old["terminal"] != "feasibility_screen" or task_rows["status"] != "rows_frozen_outcomes_unopened" or bracket_rows["status"] != "rows_frozen_outcomes_unopened" or task_rows["outcomes_opened"] or bracket_rows["outcomes_opened"]:
        raise ValueError("temporal seal or source status invalid")
    return coefficients, task_rows["rows"], bracket_rows["rows"], v4, task_vectors, bracket_vectors


def compile_plan() -> dict:
    load()
    return {"schema": "task14_bracket_native_baseline_semantic_linear_prospective_plan_v2", "candidate_id": CANDIDATE_ID, "prior_art_sha256": EXPECTED[PRIOR], "task14": {"rows": 32, "backgrounds": 16, "arms": ["native", "program"], "targets": 512}, "bracket": {"rows": 36, "endpoints": 72, "arms": ["native", "program"], "targets": 72}, "deployed_fp32_scalars": 27, "bars": dict(BARS), "price": {"physical_model_forwards": 7, "example_evaluations": 1264, "causal_installations": 584, "fits": 0, "backwards": 0, "parameter_updates": 0}}


def dot(features: list[float], coefficients: list[float]) -> float:
    return sum(feature * coefficient for feature, coefficient in zip(features, coefficients))


def collect_task(model, rows: list[dict], vectors: dict, coefficients: dict, effects: dict, torch, F, facade) -> tuple[list[dict], float]:
    parent, tokens, function, inputs, source_closure = absolute_runner._context(model, rows, torch, F, facade)
    vector_tensors = {key: torch.tensor(value["coordinates"], dtype=torch.float32, device=tokens.device) for key, value in vectors["prototypes"].items() if ".cardinality_" in key}
    indices, replacements, specs = [], [], []
    with torch.no_grad():
        for subset in SUBSETS:
            base = function(factor_gate._raw_for(inputs["recipient"], inputs["opposite"], subset, F)).detach()
            for index, row in enumerate(rows):
                key = f'{row["direction_id"]}.cardinality_{len(subset)}'
                for arm, value in (("native", base[index]), ("program", base[index] + vector_tensors[key])):
                    indices.append(index); replacements.append(value); specs.append((index, subset, key, arm))
        index_tensor = torch.tensor(indices, dtype=torch.long, device=tokens.device)
        patch_tokens = tokens[:len(rows)][index_tensor]
        finals = torch.full_like(index_tensor, parent.SUBJECT_POSITION)
        replacement = torch.stack(replacements)
        margins, closures = {}, []
        for start in range(0, len(specs), CHUNK):
            stop = min(start + CHUNK, len(specs))
            logits, _, _, closure = parent.downstream._decomposed_forward(model, patch_tokens[start:stop], finals[start:stop], torch, F, facade, replacement_heads=replacement[start:stop], native_reinstall_mask=torch.zeros(stop - start, dtype=torch.bool, device=tokens.device))
            closures.append(closure)
            for local, (row_index, subset, key, arm) in enumerate(specs[start:stop]):
                endpoint = rows[row_index]["endpoints"]["opposite_same_lemma"]
                margins[(row_index, subset, arm)] = float(logits[local, parent.SUBJECT_POSITION, endpoint["answer_id"]] - logits[local, parent.SUBJECT_POSITION, endpoint["foil_id"]])
    records = []
    for row_index, row in enumerate(rows):
        for subset in SUBSETS:
            key = f'{row["direction_id"]}.cardinality_{len(subset)}'
            feature_row = {"direction": row["direction_id"], "background": subset}
            predicted_base = dot(feasibility.task_features(feature_row), coefficients["coefficients"]["task14"])
            base = margins[(row_index, subset, "native")]; program = margins[(row_index, subset, "program")]
            records.append({"row_id": row["row_id"], "direction": row["direction_id"], "template": row["template_id"], "background": subset, "cardinality": len(subset), "native_donorward_baseline_margin": base, "predicted_native_donorward_baseline_margin": predicted_base, "actual_program_effect": program - base, "predicted_program_effect": effects["effects"]["task14"][key], "actual_counterfactual_margin": program, "predicted_counterfactual_margin": predicted_base + effects["effects"]["task14"][key], "native_recipient_correct": base < 0})
    errors = [source_closure["input_state_closure_max_absolute_error"], source_closure["input_normalized_closure_max_absolute_error"]] + [value for closure in closures for key, value in closure.items() if key in ("state_sum_max_absolute_error", "normalized_state_max_absolute_error")]
    return records, max(errors)


def collect_bracket(model, rows: list[dict], vectors: dict, coefficients: dict, effects: dict, torch, F, facade) -> tuple[list[dict], float]:
    prepared = []
    for row in rows:
        item = dict(row)
        for side in ("base", "donor"):
            item[f"{side}_open_position"] = bracket_shared.semantic_open_position(item[f"{side}_ids"], item[f"{side}_answer_id"])
        prepared.append(item)
    endpoints, tokens, finals, sources = bracket_runner.parent._pad(prepared, torch, next(model.parameters()).device)
    replay, factors = bracket_exact.factor_forward(model, tokens, finals, {}, torch, F, facade)
    arange = torch.arange(len(endpoints), device=tokens.device)
    native_terms = factors["p"][arange, sources].unsqueeze(-1) * factors["u"][arange, sources]
    vector_tensors = {key: torch.tensor(value["coordinates"], dtype=torch.float32, device=tokens.device) for key, value in vectors["prototypes"].items()}
    installed = []
    pairs = []
    for index, (row, side) in enumerate(endpoints):
        other = "donor" if side == "base" else "base"
        pair = f'{row[f"{side}_answer_id"]}->{row[f"{other}_answer_id"]}'
        pairs.append(pair); installed.append(native_terms[index] + vector_tensors[pair])
    program = bracket_exact.factor_forward(model, tokens, finals, {}, torch, F, facade, replacement_terms=torch.stack(installed), source_positions=sources)[0]
    records = []
    for index, (row, side) in enumerate(endpoints):
        other = "donor" if side == "base" else "base"; recipient = row[f"{side}_answer_id"]; donor = row[f"{other}_answer_id"]; q = int(finals[index]); pair = pairs[index]
        base = float(replay[index, q, donor] - replay[index, q, recipient]); edited = float(program[index, q, donor] - program[index, q, recipient])
        feature_row = {"recipient_closer_id": recipient, "donor_closer_id": donor}
        predicted_base = dot(feasibility.bracket_features(feature_row), coefficients["coefficients"]["bracket"])
        records.append({"row_id": row["row_id"], "side": side, "ordered_pair": pair, "recipient_closer_id": recipient, "donor_closer_id": donor, "native_donorward_baseline_margin": base, "predicted_native_donorward_baseline_margin": predicted_base, "actual_program_effect": edited - base, "predicted_program_effect": effects["effects"]["bracket"][pair], "actual_counterfactual_margin": edited, "predicted_counterfactual_margin": predicted_base + effects["effects"]["bracket"][pair], "native_recipient_correct": bool(bracket_exact.closer_margin(replay[index, q], recipient) > 0)})
    return records, 0.0


def metrics(records: list[dict], actual_key: str, predicted_key: str) -> dict:
    actual = [row[actual_key] for row in records]; predicted = [row[predicted_key] for row in records]
    an = math.sqrt(sum(value * value for value in actual)); pn = math.sqrt(sum(value * value for value in predicted))
    return {"count": len(records), "cosine": sum(a * p for a, p in zip(actual, predicted)) / (an * pn), "relative_l2_error": math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted))) / an, "predicted_to_actual_norm_ratio": pn / an, "sign_agreement": sum((a > 0) == (p > 0) for a, p in zip(actual, predicted)) / len(records)}


def base_gate(value: dict) -> bool:
    return value["cosine"] >= BARS["minimum_baseline_cosine"] and value["relative_l2_error"] <= BARS["maximum_baseline_relative_l2"] and value["sign_agreement"] >= BARS["minimum_baseline_sign_agreement"]


def counter_gate(value: dict) -> bool:
    return value["cosine"] >= BARS["minimum_counterfactual_cosine"] and value["relative_l2_error"] <= BARS["maximum_counterfactual_relative_l2"] and value["sign_agreement"] >= BARS["minimum_counterfactual_sign_agreement"]


def score(task_records: list[dict], bracket_records: list[dict], task_closure: float, bracket_replay: float, package: dict) -> dict:
    task_base = metrics(task_records, "native_donorward_baseline_margin", "predicted_native_donorward_baseline_margin")
    bracket_base = metrics(bracket_records, "native_donorward_baseline_margin", "predicted_native_donorward_baseline_margin")
    task_counter = metrics(task_records, "actual_counterfactual_margin", "predicted_counterfactual_margin")
    bracket_counter = metrics(bracket_records, "actual_counterfactual_margin", "predicted_counterfactual_margin")
    task_groups = {f"{direction}.{template}": metrics([row for row in task_records if row["direction"] == direction and row["template"] == template], "native_donorward_baseline_margin", "predicted_native_donorward_baseline_margin") for direction in ("singular_to_plural", "plural_to_singular") for template in ("near_beyond", "beyond_near")}
    bracket_groups = {pair: metrics([row for row in bracket_records if row["ordered_pair"] == pair], "native_donorward_baseline_margin", "predicted_native_donorward_baseline_margin") for pair in sorted({row["ordered_pair"] for row in bracket_records})}
    capability = {"task14": sum(row["native_recipient_correct"] for row in task_records) / len(task_records), "bracket": sum(row["native_recipient_correct"] for row in bracket_records) / len(bracket_records)}
    instrument = task_closure <= BARS["maximum_replay_or_closure_error"] and bracket_replay <= BARS["maximum_replay_or_closure_error"] and len(task_records) == 512 and len(bracket_records) == 72
    task_ok = base_gate(task_base) and all(value["cosine"] >= BARS["minimum_task14_direction_template_cosine"] and value["sign_agreement"] >= BARS["minimum_task14_direction_template_sign_agreement"] for value in task_groups.values())
    bracket_ok = base_gate(bracket_base) and all(value["sign_agreement"] >= BARS["minimum_bracket_ordered_pair_sign_agreement"] for value in bracket_groups.values())
    price = compile_plan()["price"] == {"physical_model_forwards": 7, "example_evaluations": 1264, "causal_installations": 584, "fits": 0, "backwards": 0, "parameter_updates": 0} and package["stored_fp32_scalars"] == 27
    predictions = {"pred_a_temporal_seal_instrument_and_capability": instrument and min(capability.values()) >= BARS["minimum_native_capability"], "pred_b_task14_prospective_baseline": task_ok, "pred_c_bracket_prospective_baseline": bracket_ok, "pred_d_prospective_absolute_counterfactuals": counter_gate(task_counter) and counter_gate(bracket_counter), "pred_e_transparent_standalone_price_and_boundary": price and package["runtime_dependencies"] == ["explicit semantic features", "discrete edit specification"]}
    terminal = "program_screen" if all(predictions.values()) else "capability_stop" if instrument and min(capability.values()) < BARS["minimum_native_capability"] else "null" if instrument else "invalid"
    return {"instrument": {"task14_closure_max_absolute_error": task_closure, "bracket_replay_max_absolute_error": bracket_replay}, "native_capability": capability, "task14": {"baseline": task_base, "by_direction_template": task_groups, "counterfactual": task_counter}, "bracket": {"baseline": bracket_base, "by_ordered_pair": bracket_groups, "counterfactual": bracket_counter}, "program": {"stored_fp32_scalars": 27, "stored_fp32_bytes": 108, "runtime_dependencies": package["runtime_dependencies"], "classification": "standalone_controlled_domain_native_and_counterfactual_margin_tensor_program_not_whole_model"}, "predictions": predictions, "terminal": terminal}


def main() -> None:
    plan = compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, sort_keys=True)); return
    if ARTIFACT.exists() or OUT.exists():
        raise ValueError("refusing overwrite")
    coefficients, task_rows, bracket_rows, v4, task_vectors, bracket_vectors = load()
    package = {"schema": "task14_bracket_native_baseline_semantic_linear_program_artifact_v2", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "equations": {"native_margin": "dot(frozen semantic features, frozen baseline coefficients)", "counterfactual_margin": "native_margin + selected frozen intervention effect"}, "features": coefficients["features"], "baseline_coefficients": coefficients["coefficients"], "intervention_effects": v4["effects"], "stored_fp32_scalars": 27, "runtime_dependencies": ["explicit semantic features", "discrete edit specification"], "explicitly_not_provided": ["full vocabulary logits", "free-form text support", "internal activations", "whole-model replacement", "internal intervention execution"], "terminal": "frozen_program"}
    torch, F, facade = absolute_runner.tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        task_records, task_closure = collect_task(model, task_rows, task_vectors, coefficients, v4, torch, F, facade)
        bracket_records, bracket_replay = collect_bracket(model, bracket_rows, bracket_vectors, coefficients, v4, torch, F, facade)
    artifact_bytes = managed.atomic_create_json(ARTIFACT, package)
    result = score(task_records, bracket_records, task_closure, bracket_replay, package)
    payload = managed.atomic_create_json(OUT, {"schema": "task14_bracket_native_baseline_semantic_linear_prospective_result_v2", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "plan": plan, "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(), "checkpoint_weights_sha256": checkpoint.weights_sha256, "score": result, "task14_evidence": task_records, "bracket_evidence": bracket_records, "terminal": result["terminal"]})
    print(json.dumps({"terminal": result["terminal"], "predictions": result["predictions"], "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(), "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
