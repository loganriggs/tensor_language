#!/usr/bin/env python3
"""Compile and audit the two-behavior counterfactual answer-margin actuator."""

# BQGATE: EXPERIMENT pred_a_immutable_sources_and_exact_compilation pred_b_task14_effect_recurrence pred_c_bracket_effect_recurrence pred_d_exact_margin_actuation_identity pred_e_suffix_and_price_reduction
# BQLANE: cpu
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import task14_bracket_margin_actuator as actuator

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/task14_bracket_counterfactual_margin_actuator_v4.json"
V3 = ROOT / "circuits/followups/task14_bracket_compiled_predictive_dispatcher_v3_artifact.json"
TASK = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json"
BRACKET = ROOT / "circuits/followups/bracket_suffix_free_scalar_fresh_corpus_validation_v1_result.json"
NULL = ROOT / "circuits/followups/task14_absolute_head_two_reader_context_v3_result.json"
ARTIFACT = ROOT / "circuits/followups/task14_bracket_counterfactual_margin_actuator_v4_artifact.json"
OUT = ROOT / "circuits/followups/task14_bracket_counterfactual_margin_actuator_v4_result.json"
CANDIDATE_ID = "cross_behavior.task14_bracket_counterfactual_margin_actuator_v4"
EXPECTED = {
    PRIOR: "a0ce72f3d1e7dc3ce6ae7eb887cd471f47cdd63f1899ea30ee9ab1dff535edfa",
    V3: "a8ccb6525dc8f3fd83c49a41eeaebd7cd5547606bf134ac1db9c678ff0eb7b40",
    TASK: "9a488259efdb85477f35c612696368e8a3e372338a11253615a7b53650c88fe0",
    BRACKET: "6b8db79cc8c72500586a01966eb11c9d9cde89b35221f35b8ada0928d5c78bdf",
    NULL: "a7893f3bdb9545afd234cd1d72e28262861f11f0cc85ec1007805c3c8a1acb18",
}
OLD_SCALARS = 14992
NEW_SCALARS = 16


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(actual: list[float], predicted: list[float]) -> dict:
    dot = sum(a * p for a, p in zip(actual, predicted))
    aa = sum(a * a for a in actual)
    pp = sum(p * p for p in predicted)
    return {
        "count": len(actual),
        "cosine": dot / math.sqrt(aa * pp),
        "relative_l2_error": math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / aa),
        "sign_agreement": sum((a >= 0) == (p >= 0) for a, p in zip(actual, predicted)) / len(actual),
    }


def load_sources() -> tuple[dict, dict, dict]:
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"immutable input changed: {path}")
    v3, task, bracket, null = (json.loads(path.read_text()) for path in (V3, TASK, BRACKET, NULL))
    if task["terminal"] != "valid_causal_screen" or bracket["terminal"] != "predictive_screen" or null["terminal"] != "null":
        raise ValueError("source terminal status invalid")
    return v3, task, bracket


def compile_plan() -> dict:
    load_sources()
    return {
        "schema": "task14_bracket_counterfactual_margin_actuator_plan_v4",
        "candidate_id": CANDIDATE_ID,
        "prior_art_sha256": EXPECTED[PRIOR],
        "stored_fp32_scalars": NEW_SCALARS,
        "stored_fp32_bytes": NEW_SCALARS * 4,
        "storage_reduction_fraction_from_v3": 1 - NEW_SCALARS / OLD_SCALARS,
        "price": {"model_forwards": 0, "example_evaluations": 0, "backwards": 0, "fits": 0, "parameter_updates": 0},
    }


def build(v3: dict) -> dict:
    task = {key: value["predicted_donorward_effect"] for key, value in v3["programs"]["task14"].items()}
    bracket = dict(v3["programs"]["bracket"]["predicted_effects"])
    return {
        "schema": "task14_bracket_counterfactual_margin_actuator_artifact_v4",
        "candidate_id": CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "api": {"input": ["native_unedited_answer_margin", "discrete_edit_specification"], "output": "predicted_counterfactual_answer_margin"},
        "equation": "predicted_counterfactual_margin = native_unedited_margin + selected_frozen_effect",
        "effects": {"task14": task, "bracket": bracket},
        "runtime": {"table_lookups": 1, "scalar_additions": 1, "model_forwards_after_input": 0},
        "residual_dependencies": ["native unedited answer margin", "external intervention specification"],
        "explicitly_not_provided": ["native baseline-margin generation", "full logits", "internal activations", "whole-model replacement", "internal intervention execution"],
    }


def evaluate(artifact: dict, v3: dict, task: dict, bracket: dict) -> dict:
    exact = artifact["effects"]["task14"] == {k: v["predicted_donorward_effect"] for k, v in v3["programs"]["task14"].items()} and artifact["effects"]["bracket"] == v3["programs"]["bracket"]["predicted_effects"] and len(artifact["effects"]["task14"]) == 10 and len(artifact["effects"]["bracket"]) == 6
    task_actual, task_predicted = [], []
    identity_error = 0.0
    for row in task["score"]["joined_evidence"]:
        predicted = actuator.task14_effect(artifact, row["direction"].split("_to_")[0], row["direction"].split("_to_")[1], row["cardinality"])
        actual = row["cardinality_prototype_q"]
        task_actual.append(actual); task_predicted.append(predicted)
        for baseline in (-10.0, -1.0, 0.0, 1.0, 10.0):
            identity_error = max(identity_error, abs((actuator.actuate(baseline, predicted) - (baseline + actual)) - (predicted - actual)))
    task_score = score(task_actual, task_predicted)
    bracket_actual, bracket_predicted = [], []
    pair_sign = {}
    for row in bracket["evidence"]:
        recipient, donor = map(int, row["ordered_pair"].split("->"))
        predicted = actuator.bracket_effect(artifact, recipient, donor)
        actual = row["actual_program_donorward_effect"]
        bracket_actual.append(actual); bracket_predicted.append(predicted)
        pair_sign.setdefault(row["ordered_pair"], []).append((actual >= 0) == (predicted >= 0))
        for baseline in (-10.0, -1.0, 0.0, 1.0, 10.0):
            identity_error = max(identity_error, abs((actuator.actuate(baseline, predicted) - (baseline + actual)) - (predicted - actual)))
    bracket_score = score(bracket_actual, bracket_predicted)
    rejects = False
    try:
        actuator.task14_effect(artifact, "singular", "singular", 0)
    except actuator.ActuatorError:
        try:
            actuator.bracket_effect(artifact, 2, 8)
        except actuator.ActuatorError:
            rejects = True
    price_ok = sum(len(x) for x in artifact["effects"].values()) == NEW_SCALARS and artifact["runtime"] == {"table_lookups": 1, "scalar_additions": 1, "model_forwards_after_input": 0} and 1 - NEW_SCALARS / OLD_SCALARS >= 0.9989
    predictions = {
        "pred_a_immutable_sources_and_exact_compilation": exact and rejects,
        "pred_b_task14_effect_recurrence": task_score["cosine"] >= .95 and task_score["relative_l2_error"] <= .35 and task_score["sign_agreement"] >= .95,
        "pred_c_bracket_effect_recurrence": bracket_score["cosine"] >= .99 and bracket_score["relative_l2_error"] <= .15 and bracket_score["sign_agreement"] == 1 and all(all(values) for values in pair_sign.values()),
        "pred_d_exact_margin_actuation_identity": identity_error <= 1e-12,
        "pred_e_suffix_and_price_reduction": price_ok and artifact["residual_dependencies"] == ["native unedited answer margin", "external intervention specification"],
    }
    return {
        "task14_effect_recurrence": task_score,
        "bracket_effect_recurrence": bracket_score,
        "bracket_ordered_pair_sign_agreement": {key: sum(values) / len(values) for key, values in pair_sign.items()},
        "margin_actuation_identity_max_absolute_error": identity_error,
        "storage": {"v3_scalars": OLD_SCALARS, "actuator_scalars": NEW_SCALARS, "actuator_bytes": NEW_SCALARS * 4, "reduction_fraction": 1 - NEW_SCALARS / OLD_SCALARS},
        "residual_dependencies": artifact["residual_dependencies"],
        "classification": "baseline_conditioned_counterfactual_margin_actuator_not_whole_model",
        "predictions": predictions,
        "terminal": "screen" if all(predictions.values()) else "null",
    }


def main() -> None:
    plan = compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, sort_keys=True)); return
    if ARTIFACT.exists() or OUT.exists():
        raise ValueError("refusing overwrite")
    v3, task, bracket = load_sources()
    artifact = build(v3)
    artifact_bytes = managed.atomic_create_json(ARTIFACT, artifact)
    result = evaluate(artifact, v3, task, bracket)
    payload = managed.atomic_create_json(OUT, {"schema": "task14_bracket_counterfactual_margin_actuator_result_v4", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "plan": plan, "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(), "score": result, "terminal": result["terminal"]})
    print(json.dumps({"terminal": result["terminal"], "predictions": result["predictions"], "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(), "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
