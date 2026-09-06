#!/usr/bin/env python3
"""Compile and audit one immutable dispatcher for the Task14 and bracket programs."""

# BQGATE: EXPERIMENT pred_a_immutable_exact_compilation pred_b_task14_dispatch_complete pred_c_bracket_dispatch_and_zero_complete pred_d_role_prompt_dependency_eliminated pred_e_residual_boundary_and_price
# BQLANE: cpu
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import task14_bracket_compiled_dispatcher as dispatcher


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/task14_bracket_compiled_dispatcher_v2.json"
TASK14_ARTIFACT = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_artifact_v1.json"
TASK14_VALIDATION = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json"
BRACKET_ARTIFACT = ROOT / "circuits/fast_screens/bracket_l13h8_ordered_pair_displacement_artifact_v1.json"
BRACKET_VALIDATION = ROOT / "circuits/followups/bracket_l13h8_ordered_pair_displacement_program_ood_validation_v1_result.json"
PACKAGE_OUT = ROOT / "circuits/followups/task14_bracket_compiled_dispatcher_v2_artifact.json"
OUT = ROOT / "circuits/followups/task14_bracket_compiled_dispatcher_v2_result.json"
CANDIDATE_ID = "cross_behavior.task14_bracket_compiled_dispatcher_v2"
EXPECTED = {
    PRIOR: "25009573d26f5c866750d139bb45749d40cbcdab5318c259d7c68edcc9f69017",
    TASK14_ARTIFACT: "cce00d8f2309b0f9e0329c094238505819f2cf8a21c04e276af2085e068c1d07",
    TASK14_VALIDATION: "9a488259efdb85477f35c612696368e8a3e372338a11253615a7b53650c88fe0",
    BRACKET_ARTIFACT: "531434535daf526ebf584564afbfd5834c75df7bd636014ea233f9ae71206db0",
    BRACKET_VALIDATION: "3b267f069647824fb7557e9784c63becb0366f94fe4d274fea343ae2bc802e5f",
}
FORBIDDEN_API_TERMS = (
    "prompt", "row_id", "template", "family", "activation", "logit", "model",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> tuple[dict, dict, dict, dict]:
    for path, expected in EXPECTED.items():
        if _sha(path) != expected:
            raise ValueError(f"immutable input changed: {path}")
    return tuple(json.loads(path.read_text()) for path in (
        TASK14_ARTIFACT, TASK14_VALIDATION, BRACKET_ARTIFACT, BRACKET_VALIDATION,
    ))  # type: ignore[return-value]


def compile_plan() -> dict:
    _load()
    return {
        "schema": "task14_bracket_compiled_dispatcher_plan_v2",
        "candidate_id": CANDIDATE_ID,
        "prior_art_sha256": EXPECTED[PRIOR],
        "source_sha256": {path.name: value for path, value in EXPECTED.items() if path != PRIOR},
        "selected_vectors": {"task14": 10, "bracket": 6, "total": 16},
        "vector_width": dispatcher.WIDTH,
        "stored_fp32_scalars": 16 * dispatcher.WIDTH,
        "stored_fp32_bytes": 16 * dispatcher.WIDTH * 4,
        "dispatch_price": {"model_forwards": 0, "example_evaluations": 0, "backwards": 0, "fits": 0, "parameter_updates": 0},
        "residual_dependencies": ["external intervention specification", "native base activation", "native downstream suffix/logit path"],
    }


def build_package(task14: dict, bracket: dict) -> dict:
    task_vectors = {
        key: value["coordinates"] for key, value in task14["prototypes"].items()
        if ".cardinality_" in key
    }
    bracket_vectors = {key: value["coordinates"] for key, value in bracket["prototypes"].items()}
    return {
        "schema": "task14_bracket_compiled_dispatcher_artifact_v2",
        "candidate_id": CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_sha256": {path.name: value for path, value in EXPECTED.items() if path != PRIOR},
        "api": {
            "task14": ["recipient_number", "donor_number", "cardinality"],
            "bracket": ["recipient_closer_id", "donor_closer_id"],
        },
        "sites": {"task14": "layer11.head3.final_position", "bracket": "layer13.head8.semantic_opener_position"},
        "programs": {
            "task14": {"vectors": task_vectors},
            "bracket": {"vectors": bracket_vectors, "licensed_zero_self_pairs": list(dispatcher.CLOSERS)},
        },
        "residual_dependencies": ["external intervention specification", "native base activation", "native downstream suffix/logit path"],
        "explicitly_not_provided": ["raw-prompt selector", "base-state constructor", "suffix predictor", "whole-model replacement"],
    }


def _rejects(call) -> bool:
    try:
        call()
    except dispatcher.DispatchError:
        return True
    return False


def evaluate(package: dict, task14_source: dict, task14_validation: dict, bracket_source: dict, bracket_validation: dict) -> dict:
    task_vectors = package["programs"]["task14"]["vectors"]
    bracket_vectors = package["programs"]["bracket"]["vectors"]
    exact = (
        set(task_vectors) == {key for key in task14_source["prototypes"] if ".cardinality_" in key}
        and set(bracket_vectors) == set(bracket_source["prototypes"])
        and all(task_vectors[key] == task14_source["prototypes"][key]["coordinates"] for key in task_vectors)
        and all(bracket_vectors[key] == bracket_source["prototypes"][key]["coordinates"] for key in bracket_vectors)
    )
    task_rows = task14_validation["score"]["joined_evidence"]
    task_ok = all(
        dispatcher.dispatch_task14(
            package,
            recipient_number="singular" if row["direction"] == "singular_to_plural" else "plural",
            donor_number="plural" if row["direction"] == "singular_to_plural" else "singular",
            cardinality=row["cardinality"],
        ) is task_vectors[f'{row["direction"]}.cardinality_{row["cardinality"]}']
        for row in task_rows
    )
    task_rejections = all((
        _rejects(lambda: dispatcher.dispatch_task14(package, recipient_number="dual", donor_number="plural", cardinality=0)),
        _rejects(lambda: dispatcher.dispatch_task14(package, recipient_number="singular", donor_number="singular", cardinality=0)),
        _rejects(lambda: dispatcher.dispatch_task14(package, recipient_number="singular", donor_number="plural", cardinality=5)),
        _rejects(lambda: dispatcher.dispatch_task14(package, recipient_number="singular", donor_number="plural", cardinality=True)),
    ))
    bracket_rows = bracket_validation["evidence"]
    target_rows = [row for row in bracket_rows if row["program_role"] == "target"]
    control_rows = [row for row in bracket_rows if row["program_role"] == "control"]
    bracket_ok = all(
        dispatcher.dispatch_bracket(
            package,
            recipient_closer_id=int(row["ordered_pair"].split("->")[0]),
            donor_closer_id=int(row["ordered_pair"].split("->")[1]),
        ) is bracket_vectors[row["dispatch"]]
        for row in target_rows
    )
    zeros_ok = all(
        dispatcher.dispatch_bracket(
            package,
            recipient_closer_id=int(row["ordered_pair"].split("->")[0]),
            donor_closer_id=int(row["ordered_pair"].split("->")[1]),
        ) == [0.0] * dispatcher.WIDTH
        for row in control_rows
    )
    bracket_rejections = all((
        _rejects(lambda: dispatcher.dispatch_bracket(package, recipient_closer_id=2, donor_closer_id=8)),
        _rejects(lambda: dispatcher.dispatch_bracket(package, recipient_closer_id=1, donor_closer_id=2)),
    ))
    api_text = json.dumps(package["api"], sort_keys=True).lower()
    signatures = " ".join(str(inspect.signature(function)).lower() for function in (
        dispatcher.dispatch_task14, dispatcher.dispatch_bracket,
    ))
    role_free = all(term not in api_text and term not in signatures for term in FORBIDDEN_API_TERMS)
    scalar_count = sum(len(vector) for vector in task_vectors.values()) + sum(len(vector) for vector in bracket_vectors.values())
    boundary = scalar_count == 18432 and package["residual_dependencies"] == [
        "external intervention specification", "native base activation", "native downstream suffix/logit path",
    ]
    predictions = {
        "pred_a_immutable_exact_compilation": exact and len(task_vectors) == 10 and len(bracket_vectors) == 6,
        "pred_b_task14_dispatch_complete": len(task_rows) == 512 and task_ok and task_rejections,
        "pred_c_bracket_dispatch_and_zero_complete": len(target_rows) == 144 and len(control_rows) == 216 and bracket_ok and zeros_ok and bracket_rejections,
        "pred_d_role_prompt_dependency_eliminated": role_free,
        "pred_e_residual_boundary_and_price": boundary,
    }
    return {
        "coverage": {"task14_cells": len(task_rows), "bracket_targets": len(target_rows), "bracket_zero_controls": len(control_rows)},
        "storage": {"selected_vectors": 16, "fp32_scalars": scalar_count, "fp32_bytes": scalar_count * 4},
        "eliminated_dispatch_dependencies": ["recipient role prompt", "donor role prompt", "row lookup", "template lookup", "family lookup", "model call"],
        "residual_dependencies": package["residual_dependencies"],
        "classification": "compiled_intervention_dispatcher_not_autonomous_predictor",
        "predictions": predictions,
        "terminal": "screen" if all(predictions.values()) else "null",
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if PACKAGE_OUT.exists() or OUT.exists():
        raise ValueError("refusing to overwrite compiled dispatcher artifact or result")
    task14, task14_validation, bracket, bracket_validation = _load()
    package = build_package(task14, bracket)
    package_bytes = managed.atomic_create_json(PACKAGE_OUT, package)
    score = evaluate(package, task14, task14_validation, bracket, bracket_validation)
    payload = managed.atomic_create_json(OUT, {
        "schema": "task14_bracket_compiled_dispatcher_result_v2",
        "candidate_id": CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan,
        "compiled_artifact": {"path": str(PACKAGE_OUT.relative_to(ROOT)), "sha256": hashlib.sha256(package_bytes).hexdigest()},
        "score": score,
        "terminal": score["terminal"],
    })
    print(json.dumps({"terminal": score["terminal"], "predictions": score["predictions"], "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()

