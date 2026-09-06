#!/usr/bin/env python3
"""Cross-fitted feasibility test for minimal semantic native-margin generators."""

# BQGATE: EXPERIMENT pred_a_instrument_and_folds pred_b_task14_semantic_baseline pred_c_bracket_semantic_baseline pred_d_absolute_counterfactual_margins pred_e_fixed_price_and_scope
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import circuit_fast_screen_managed_runner as managed
import run_bracket_l13h8_source_region_payload_factorial as bracket_exact
import run_bracket_suffix_free_scalar_fresh_corpus_validation_v1 as bracket_runner
import run_task14_direction_cardinality_absolute_head_program_v1 as absolute_runner
import run_task14_mlp6_7_direction_cardinality_prototype_causal_validation as task_validation
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/task14_bracket_native_baseline_semantic_linear_feasibility_v1.json"
AMEND = ROOT / "circuits/prior_art/task14_bracket_native_baseline_semantic_linear_feasibility_v1_authority_amendment.json"
V4 = ROOT / "circuits/followups/task14_bracket_counterfactual_margin_actuator_v4_artifact.json"
TASK_EFFECTS = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json"
BRACKET_EFFECTS = ROOT / "circuits/followups/bracket_suffix_free_scalar_fresh_corpus_validation_v1_result.json"
ROWS = ROOT / "circuits/prior_art/bracket_suffix_free_fresh_corpus_v1_rows.json"
OUT = ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_feasibility_v1_result.json"
CANDIDATE_ID = "cross_behavior.task14_bracket_native_baseline_semantic_linear_feasibility_v1"
EXPECTED = {
    PRIOR: "c7d7b9ba662473c8f18f3d8d18854113ab4ee0cf6cb2008f3197198fd097badb",
    AMEND: "034c4bf80eed6d46a7c27cac153d34dd541c4bb51d1b7a03fc3c274f63e62b6d",
    V4: "85c5cc0549421fc1575d96ce621d0677ea4b0cc2d154b2c0bf7af90f4148bd4c",
    TASK_EFFECTS: "9a488259efdb85477f35c612696368e8a3e372338a11253615a7b53650c88fe0",
    BRACKET_EFFECTS: "6b8db79cc8c72500586a01966eb11c9d9cde89b35221f35b8ada0928d5c78bdf",
    ROWS: "d808806fd1b05f834cf6ef4fa71465464c0403f66dc13ece8a24cffcc40142f9",
    Path(task_validation.authority.__file__): "b8ac252137b9844ee5c417c073497e18df337c1e4200d17513ef6e31a0915ab1",
    Path(task_validation.__file__): "8b4c4c645cf333f26cf3a81669d36ca5d952c21704aa637089bae98adfa849a4",
}
BARS = {
    "minimum_baseline_cosine": .90,
    "maximum_baseline_relative_l2": .50,
    "minimum_baseline_sign_agreement": .90,
    "minimum_each_fold_cosine": .85,
    "maximum_each_fold_relative_l2": .60,
    "minimum_counterfactual_cosine": .90,
    "maximum_counterfactual_relative_l2": .50,
    "minimum_counterfactual_sign_agreement": .85,
    "maximum_replay_or_closure_error": 1e-4,
}
SUBSETS = factor_gate.BACKGROUND_SUBSETS
CHUNK = 256


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict, dict, dict]:
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"immutable input changed: {path}")
    v4, task, bracket = (json.loads(path.read_text()) for path in (V4, TASK_EFFECTS, BRACKET_EFFECTS))
    if task["terminal"] != "valid_causal_screen" or bracket["terminal"] != "predictive_screen":
        raise ValueError("effect authority status invalid")
    return v4, task, bracket


def compile_plan() -> dict:
    load()
    return {
        "schema": "task14_bracket_native_baseline_semantic_linear_feasibility_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "prior_art_sha256": EXPECTED[PRIOR],
        "authority_amendment_sha256": EXPECTED[AMEND],
        "task14": {"rows": 32, "backgrounds": 16, "targets": 512, "coefficients_per_fold": 6, "folds": 2},
        "bracket": {"rows": 36, "endpoints": 72, "coefficients_per_fold": 5, "folds": 2},
        "bars": dict(BARS),
        "price": {"physical_model_forwards": 4, "example_evaluations": 680, "closed_form_fits": 4, "backwards": 0, "parameter_updates": 0},
    }


def task_folds(rows: list[dict]) -> dict[str, int]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["direction_id"], row["template_id"])].append(row)
    result = {}
    for values in groups.values():
        for index, row in enumerate(sorted(values, key=lambda value: value["row_id"])):
            result[row["row_id"]] = index % 2
    return result


def collect_task(model, torch, F, facade) -> tuple[list[dict], float]:
    rows = task_validation.authority.build_rows()
    parent, tokens, function, inputs, source_closure = absolute_runner._context(model, rows, torch, F, facade)
    indices, replacements, specs = [], [], []
    with torch.no_grad():
        for subset in SUBSETS:
            base = function(factor_gate._raw_for(inputs["recipient"], inputs["opposite"], subset, F)).detach()
            for index in range(len(rows)):
                indices.append(index); replacements.append(base[index]); specs.append((index, subset))
        index_tensor = torch.tensor(indices, dtype=torch.long, device=tokens.device)
        patch_tokens = tokens[:len(rows)][index_tensor]
        finals = torch.full_like(index_tensor, parent.SUBJECT_POSITION)
        replacement = torch.stack(replacements)
        records, closures = [], []
        fold_map = task_folds(rows)
        for start in range(0, len(specs), CHUNK):
            stop = min(start + CHUNK, len(specs))
            logits, _, _, closure = parent.downstream._decomposed_forward(model, patch_tokens[start:stop], finals[start:stop], torch, F, facade, replacement_heads=replacement[start:stop], native_reinstall_mask=torch.zeros(stop - start, dtype=torch.bool, device=tokens.device))
            closures.append(closure)
            for local, (row_index, subset) in enumerate(specs[start:stop]):
                row = rows[row_index]; endpoint = row["endpoints"]["opposite_same_lemma"]
                baseline = float(logits[local, parent.SUBJECT_POSITION, endpoint["answer_id"]] - logits[local, parent.SUBJECT_POSITION, endpoint["foil_id"]])
                records.append({"row_id": row["row_id"], "direction": row["direction_id"], "template": row["template_id"], "background": subset, "cardinality": len(subset), "fold": fold_map[row["row_id"]], "native_donorward_baseline_margin": baseline})
    errors = [source_closure["input_state_closure_max_absolute_error"], source_closure["input_normalized_closure_max_absolute_error"]]
    errors += [value for closure in closures for key, value in closure.items() if key in ("state_sum_max_absolute_error", "normalized_state_max_absolute_error")]
    return records, max(errors)


def collect_bracket(model, torch, F, facade) -> tuple[list[dict], float]:
    rows = bracket_runner._rows()
    endpoints, tokens, finals, _ = bracket_runner.parent._pad(rows, torch, next(model.parameters()).device)
    replay, _ = bracket_exact.factor_forward(model, tokens, finals, {}, torch, F, facade)
    fold_map = {row_id: index % 2 for index, row_id in enumerate(sorted(row["row_id"] for row in rows))}
    records = []
    for index, (row, side) in enumerate(endpoints):
        other = "donor" if side == "base" else "base"
        recipient = row[f"{side}_answer_id"]; donor = row[f"{other}_answer_id"]
        q = int(finals[index])
        baseline = float(replay[index, q, donor] - replay[index, q, recipient])
        records.append({"row_id": row["row_id"], "side": side, "recipient_closer_id": recipient, "donor_closer_id": donor, "ordered_pair": f"{recipient}->{donor}", "fold": fold_map[row["row_id"]], "native_donorward_baseline_margin": baseline})
    return records, 0.0


def task_features(row: dict) -> list[float]:
    return [1.0, 1.0 if row["direction"] == "singular_to_plural" else -1.0] + [1.0 if letter in row["background"] else 0.0 for letter in "EAUW"]


def bracket_features(row: dict) -> list[float]:
    return [1.0, float(row["recipient_closer_id"] == 1), float(row["recipient_closer_id"] == 8), float(row["donor_closer_id"] == 1), float(row["donor_closer_id"] == 8)]


def cross_fit(records: list[dict], feature_fn) -> tuple[list[dict], dict[str, list[float]]]:
    output = [dict(row) for row in records]; coefficients = {}
    for heldout in (0, 1):
        train = [row for row in output if row["fold"] != heldout]
        test_indices = [index for index, row in enumerate(output) if row["fold"] == heldout]
        x = np.asarray([feature_fn(row) for row in train], dtype=np.float64)
        y = np.asarray([row["native_donorward_baseline_margin"] for row in train], dtype=np.float64)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        coefficients[str(heldout)] = beta.tolist()
        for index in test_indices:
            output[index]["predicted_native_donorward_baseline_margin"] = float(np.dot(np.asarray(feature_fn(output[index])), beta))
    return output, coefficients


def metrics(records: list[dict], actual_key: str, predicted_key: str) -> dict:
    actual = [row[actual_key] for row in records]; predicted = [row[predicted_key] for row in records]
    an = math.sqrt(sum(value * value for value in actual)); pn = math.sqrt(sum(value * value for value in predicted))
    return {"count": len(records), "cosine": sum(a * p for a, p in zip(actual, predicted)) / (an * pn), "relative_l2_error": math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted))) / an, "predicted_to_actual_norm_ratio": pn / an, "sign_agreement": sum((a > 0) == (p > 0) for a, p in zip(actual, predicted)) / len(records)}


def baseline_gate(overall: dict, folds: dict) -> bool:
    return overall["cosine"] >= BARS["minimum_baseline_cosine"] and overall["relative_l2_error"] <= BARS["maximum_baseline_relative_l2"] and overall["sign_agreement"] >= BARS["minimum_baseline_sign_agreement"] and all(value["cosine"] >= BARS["minimum_each_fold_cosine"] and value["relative_l2_error"] <= BARS["maximum_each_fold_relative_l2"] for value in folds.values())


def counterfactual_gate(value: dict) -> bool:
    return value["cosine"] >= BARS["minimum_counterfactual_cosine"] and value["relative_l2_error"] <= BARS["maximum_counterfactual_relative_l2"] and value["sign_agreement"] >= BARS["minimum_counterfactual_sign_agreement"]


def score(task_records: list[dict], bracket_records: list[dict], task_closure: float, bracket_replay: float, task_coefficients: dict, bracket_coefficients: dict) -> dict:
    v4, task_effects, bracket_effects = load()
    task_effect_map = {(row["row_id"], row["background"]): row for row in task_effects["score"]["joined_evidence"]}
    bracket_effect_map = {(row["row_id"], row["side"]): row for row in bracket_effects["evidence"]}
    for row in task_records:
        effect = task_effect_map[(row["row_id"], row["background"])]
        key = f'{row["direction"]}.cardinality_{row["cardinality"]}'
        row["actual_program_effect"] = effect["cardinality_prototype_q"]
        row["predicted_program_effect"] = v4["effects"]["task14"][key]
        row["actual_counterfactual_margin"] = row["native_donorward_baseline_margin"] + row["actual_program_effect"]
        row["predicted_counterfactual_margin"] = row["predicted_native_donorward_baseline_margin"] + row["predicted_program_effect"]
    for row in bracket_records:
        effect = bracket_effect_map[(row["row_id"], row["side"])]
        row["actual_program_effect"] = effect["actual_program_donorward_effect"]
        row["predicted_program_effect"] = v4["effects"]["bracket"][row["ordered_pair"]]
        row["actual_counterfactual_margin"] = row["native_donorward_baseline_margin"] + row["actual_program_effect"]
        row["predicted_counterfactual_margin"] = row["predicted_native_donorward_baseline_margin"] + row["predicted_program_effect"]
    task_baseline = metrics(task_records, "native_donorward_baseline_margin", "predicted_native_donorward_baseline_margin")
    bracket_baseline = metrics(bracket_records, "native_donorward_baseline_margin", "predicted_native_donorward_baseline_margin")
    task_folds = {str(fold): metrics([row for row in task_records if row["fold"] == fold], "native_donorward_baseline_margin", "predicted_native_donorward_baseline_margin") for fold in (0, 1)}
    bracket_folds = {str(fold): metrics([row for row in bracket_records if row["fold"] == fold], "native_donorward_baseline_margin", "predicted_native_donorward_baseline_margin") for fold in (0, 1)}
    task_counterfactual = metrics(task_records, "actual_counterfactual_margin", "predicted_counterfactual_margin")
    bracket_counterfactual = metrics(bracket_records, "actual_counterfactual_margin", "predicted_counterfactual_margin")
    instrument = task_closure <= BARS["maximum_replay_or_closure_error"] and bracket_replay <= BARS["maximum_replay_or_closure_error"] and len(task_records) == 512 and len(bracket_records) == 72 and {row["fold"] for row in task_records} == {0, 1} and {row["fold"] for row in bracket_records} == {0, 1}
    price = compile_plan()["price"] == {"physical_model_forwards": 4, "example_evaluations": 680, "closed_form_fits": 4, "backwards": 0, "parameter_updates": 0}
    predictions = {"pred_a_instrument_and_folds": instrument, "pred_b_task14_semantic_baseline": baseline_gate(task_baseline, task_folds), "pred_c_bracket_semantic_baseline": baseline_gate(bracket_baseline, bracket_folds), "pred_d_absolute_counterfactual_margins": counterfactual_gate(task_counterfactual) and counterfactual_gate(bracket_counterfactual), "pred_e_fixed_price_and_scope": price and all(len(value) == 6 for value in task_coefficients.values()) and all(len(value) == 5 for value in bracket_coefficients.values())}
    terminal = "feasibility_screen" if all(predictions.values()) else "null" if instrument else "invalid"
    return {"instrument": {"task14_closure_max_absolute_error": task_closure, "bracket_replay_max_absolute_error": bracket_replay}, "task14": {"baseline": task_baseline, "by_fold": task_folds, "counterfactual": task_counterfactual, "coefficients_by_heldout_fold": task_coefficients}, "bracket": {"baseline": bracket_baseline, "by_fold": bracket_folds, "counterfactual": bracket_counterfactual, "coefficients_by_heldout_fold": bracket_coefficients}, "scope": "retrospective semantic-linear feasibility only", "predictions": predictions, "terminal": terminal}


def main() -> None:
    plan = compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise ValueError("refusing overwrite")
    torch, F, facade = absolute_runner.tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        task_raw, task_closure = collect_task(model, torch, F, facade)
        bracket_raw, bracket_replay = collect_bracket(model, torch, F, facade)
    task_records, task_coefficients = cross_fit(task_raw, task_features)
    bracket_records, bracket_coefficients = cross_fit(bracket_raw, bracket_features)
    result = score(task_records, bracket_records, task_closure, bracket_replay, task_coefficients, bracket_coefficients)
    payload = managed.atomic_create_json(OUT, {"schema": "task14_bracket_native_baseline_semantic_linear_feasibility_result_v1", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256, "score": result, "task14_evidence": task_records, "bracket_evidence": bracket_records, "terminal": result["terminal"]})
    print(json.dumps({"terminal": result["terminal"], "predictions": result["predictions"], "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
