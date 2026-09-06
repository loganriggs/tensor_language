#!/usr/bin/env python3
"""Compile and audit the strongest honest hybrid margin program."""

# BQGATE: EXPERIMENT pred_a_exact_component_extraction pred_b_task14_standalone_prospective_gate pred_c_bracket_conditioned_prospective_gate pred_d_exact_composition_contract pred_e_literal_price_and_dependency_boundary
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

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/task14_standalone_bracket_conditioned_hybrid_v6.json"
V2_RESULT = ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_prospective_v2_result.json"
V2_ARTIFACT = ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_prospective_v2_artifact.json"
V5_RESULT = ROOT / "circuits/followups/task14_bracket_margin_actuator_composition_contract_v5_result.json"
V5_ARTIFACT = ROOT / "circuits/followups/task14_bracket_margin_actuator_composition_contract_v5_artifact.json"
ERROR = ROOT / "circuits/followups/task14_bracket_counterfactual_error_budget_audit_v3_result.json"
V4 = ROOT / "circuits/followups/task14_bracket_counterfactual_margin_actuator_v4_artifact.json"
ARTIFACT = ROOT / "circuits/followups/task14_standalone_bracket_conditioned_hybrid_v6_artifact.json"
OUT = ROOT / "circuits/followups/task14_standalone_bracket_conditioned_hybrid_v6_result.json"
CANDIDATE_ID = "cross_behavior.task14_standalone_bracket_conditioned_hybrid_v6"
EXPECTED = {PRIOR: "b769d8a68227c43870a1bed6d4cf866f69b5620cf08bbc95322d354502cd2a6f", V2_RESULT: "1d2f99a6c965ed0d6794cb83a6fb0c8953d11e9a599e769b02d4a0f612d89ea4", V2_ARTIFACT: "013532f8481ef6e0c959e1ffdb045fb76214c9b3f72452c92d8e12543580ae5a", V5_RESULT: "cef063b5875912af744a0205b55fc0a04296eedd67563ce36f678315dc47032a", V5_ARTIFACT: "3867afee8732515c8947560ac3c095f459959942f075d0ecd5b0f44e2c91ed2b", ERROR: "845ad74db5488e7af84dd0e9e3a3e61d47407ee6b5741208cf2fe13410639f3a", V4: "85c5cc0549421fc1575d96ce621d0677ea4b0cc2d154b2c0bf7af90f4148bd4c"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict, dict, dict, dict]:
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"immutable input changed: {path}")
    result, artifact, composition, v4 = (json.loads(path.read_text()) for path in (V2_RESULT, V2_ARTIFACT, V5_RESULT, V4))
    if result["terminal"] != "null" or composition["terminal"] != "screen":
        raise ValueError("parent status changed")
    return result, artifact, composition, v4


def compile_plan() -> dict:
    load()
    return {"schema": "task14_standalone_bracket_conditioned_hybrid_plan_v6", "candidate_id": CANDIDATE_ID, "prior_art_sha256": EXPECTED[PRIOR], "inventory": {"task14_baseline_coefficients": 6, "task14_effects": 10, "bracket_effects": 6, "total_fp32_scalars": 22, "total_fp32_bytes": 88}, "price": {"model_forwards": 0, "example_evaluations": 0, "fits": 0, "backwards": 0, "parameter_updates": 0}}


def build(v2_artifact: dict, v4: dict) -> dict:
    return {"schema": "task14_standalone_bracket_conditioned_hybrid_artifact_v6", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "programs": {"task14": {"features": v2_artifact["features"]["task14"], "native_margin_coefficients": v2_artifact["baseline_coefficients"]["task14"], "intervention_effects": v4["effects"]["task14"], "equations": {"native_margin": "dot(features, native_margin_coefficients)", "counterfactual_margin": "native_margin + selected intervention_effect"}}, "bracket": {"intervention_effects": v4["effects"]["bracket"], "equation": "counterfactual_margin = supplied_native_margin + selected intervention_effect"}}, "composition": {"same_slot": "right_biased_overwrite_from_immutable_baseline", "different_slots": "commuting_product_update", "effect_history_summed": False}, "runtime_dependencies": {"task14": ["direction", "E/A/U/W membership", "edit specification"], "bracket": ["native unedited donorward margin", "ordered closer edit"]}, "stored_fp32_scalars": 22, "explicitly_not_provided": ["standalone bracket native margin", "full vocabulary logits", "whole-model replacement", "internal intervention execution"], "terminal": "frozen_hybrid_program"}


def metrics(records: list[dict], actual_key: str, predicted_key: str) -> dict:
    actual = [row[actual_key] for row in records]; predicted = [row[predicted_key] for row in records]
    an = math.sqrt(sum(value * value for value in actual)); pn = math.sqrt(sum(value * value for value in predicted))
    return {"count": len(records), "cosine": sum(a * p for a, p in zip(actual, predicted)) / (an * pn), "relative_l2_error": math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted))) / an, "predicted_to_actual_norm_ratio": pn / an, "sign_agreement": sum((a > 0) == (p > 0) for a, p in zip(actual, predicted)) / len(records)}


def evaluate(package: dict, result: dict, v2_artifact: dict, composition: dict, v4: dict) -> dict:
    extraction = package["programs"]["task14"]["native_margin_coefficients"] == v2_artifact["baseline_coefficients"]["task14"] and package["programs"]["task14"]["intervention_effects"] == v4["effects"]["task14"] and package["programs"]["bracket"]["intervention_effects"] == v4["effects"]["bracket"] and len(package["programs"]["task14"]["native_margin_coefficients"]) == 6 and len(package["programs"]["task14"]["intervention_effects"]) == 10 and len(package["programs"]["bracket"]["intervention_effects"]) == 6
    task_score = result["score"]["task14"]
    task_gate = task_score["baseline"]["cosine"] >= .90 and task_score["baseline"]["relative_l2_error"] <= .50 and task_score["baseline"]["sign_agreement"] >= .90 and task_score["counterfactual"]["cosine"] >= .90 and task_score["counterfactual"]["relative_l2_error"] <= .50 and task_score["counterfactual"]["sign_agreement"] >= .85 and all(value["cosine"] >= .80 and value["sign_agreement"] >= .85 for value in task_score["by_direction_template"].values())
    bracket_effect = metrics(result["bracket_evidence"], "actual_program_effect", "predicted_program_effect")
    bracket_gate = bracket_effect["cosine"] >= .99 and bracket_effect["relative_l2_error"] <= .15 and bracket_effect["sign_agreement"] == 1.0
    composition_gate = composition["terminal"] == "screen" and all(composition["score"]["predictions"].values()) and composition["score"]["exhaustive_cases"] == {"identity": 35, "idempotence": 95, "same_slot_ordered_overwrite": 905, "independent_slot_commutativity": 2250}
    price = package["stored_fp32_scalars"] == 22 and sum((len(package["programs"]["task14"]["native_margin_coefficients"]), len(package["programs"]["task14"]["intervention_effects"]), len(package["programs"]["bracket"]["intervention_effects"]))) == 22 and package["runtime_dependencies"] == {"task14": ["direction", "E/A/U/W membership", "edit specification"], "bracket": ["native unedited donorward margin", "ordered closer edit"]}
    predictions = {"pred_a_exact_component_extraction": extraction, "pred_b_task14_standalone_prospective_gate": task_gate, "pred_c_bracket_conditioned_prospective_gate": bracket_gate, "pred_d_exact_composition_contract": composition_gate, "pred_e_literal_price_and_dependency_boundary": price}
    return {"task14_prospective": task_score, "bracket_newest_corpus_effect_recurrence": bracket_effect, "composition_exhaustive_cases": composition["score"]["exhaustive_cases"], "storage": {"fp32_scalars": 22, "bytes": 88}, "runtime_dependencies": package["runtime_dependencies"], "classification": "task14_standalone_bracket_baseline_conditioned_predictive_composable_manipulable_margin_program_not_whole_model", "prospective_combined_v2_status_preserved": "null", "predictions": predictions, "terminal": "screen" if all(predictions.values()) else "null"}


def main() -> None:
    plan = compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, sort_keys=True)); return
    if ARTIFACT.exists() or OUT.exists():
        raise ValueError("refusing overwrite")
    result, v2_artifact, composition, v4 = load(); package = build(v2_artifact, v4)
    artifact_bytes = managed.atomic_create_json(ARTIFACT, package); score = evaluate(package, result, v2_artifact, composition, v4)
    payload = managed.atomic_create_json(OUT, {"schema": "task14_standalone_bracket_conditioned_hybrid_result_v6", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "plan": plan, "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(), "score": score, "terminal": score["terminal"]})
    print(json.dumps({"terminal": score["terminal"], "predictions": score["predictions"], "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(), "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
