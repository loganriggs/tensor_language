#!/usr/bin/env python3
"""Exact post-result counterfactual error and cancellation audit."""

# BQGATE: EXPERIMENT pred_a_exact_error_identity pred_b_behavior_error_budgets_reported pred_c_cancellation_reported pred_d_no_rescue_or_new_fit
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
PRIOR = ROOT / "circuits/prior_art/task14_bracket_counterfactual_error_budget_audit_v3.json"
SOURCE = ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_prospective_v2_result.json"
ARTIFACT = ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_prospective_v2_artifact.json"
OUT = ROOT / "circuits/followups/task14_bracket_counterfactual_error_budget_audit_v3_result.json"
CANDIDATE_ID = "cross_behavior.task14_bracket_counterfactual_error_budget_audit_v3"
EXPECTED = {PRIOR: "bba8c0c7f00374c9aa0a0eeb599f04312a3e305e20256504157a055418327693", SOURCE: "1d2f99a6c965ed0d6794cb83a6fb0c8953d11e9a599e769b02d4a0f612d89ea4", ARTIFACT: "013532f8481ef6e0c959e1ffdb045fb76214c9b3f72452c92d8e12543580ae5a"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> dict:
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"immutable input changed: {path}")
    source = json.loads(SOURCE.read_text())
    if source["terminal"] != "null":
        raise ValueError("prospective null status changed")
    return source


def compile_plan() -> dict:
    source = load()
    return {"schema": "task14_bracket_counterfactual_error_budget_audit_plan_v3", "candidate_id": CANDIDATE_ID, "prior_art_sha256": EXPECTED[PRIOR], "source_result_sha256": EXPECTED[SOURCE], "targets": len(source["task14_evidence"]) + len(source["bracket_evidence"]), "price": {"model_forwards": 0, "example_evaluations": 0, "fits": 0, "backwards": 0, "parameter_updates": 0}}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values); position = fraction * (len(ordered) - 1); lower = int(position); upper = min(lower + 1, len(ordered) - 1); weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def budget(records: list[dict]) -> tuple[dict, float]:
    baseline_error = [row["predicted_native_donorward_baseline_margin"] - row["native_donorward_baseline_margin"] for row in records]
    effect_error = [row["predicted_program_effect"] - row["actual_program_effect"] for row in records]
    total_error = [row["predicted_counterfactual_margin"] - row["actual_counterfactual_margin"] for row in records]
    identity = max(abs(total - (base + effect)) for total, base, effect in zip(total_error, baseline_error, effect_error))
    base_norm = math.sqrt(sum(value * value for value in baseline_error)); effect_norm = math.sqrt(sum(value * value for value in effect_error)); total_norm = math.sqrt(sum(value * value for value in total_error)); target_norm = math.sqrt(sum(row["actual_counterfactual_margin"] ** 2 for row in records))
    cross = sum(base * effect for base, effect in zip(baseline_error, effect_error))
    cosine = cross / (base_norm * effect_norm) if base_norm and effect_norm else 0.0
    cancellation = [(abs(row["native_donorward_baseline_margin"]) + abs(row["actual_program_effect"])) / max(abs(row["actual_counterfactual_margin"]), 1e-12) for row in records]
    return {"count": len(records), "l2_norms": {"baseline_error": base_norm, "effect_error": effect_norm, "counterfactual_error": total_norm, "actual_counterfactual_target": target_norm}, "relative_to_counterfactual_error": {"baseline_error": base_norm / total_norm, "effect_error": effect_norm / total_norm}, "relative_to_target": {"baseline_error": base_norm / target_norm, "effect_error": effect_norm / target_norm, "counterfactual_error": total_norm / target_norm}, "baseline_effect_error_cosine": cosine, "baseline_effect_error_cross_term": 2 * cross, "squared_norm_identity_error": abs(total_norm ** 2 - (base_norm ** 2 + effect_norm ** 2 + 2 * cross)), "cancellation_amplification": {"median": percentile(cancellation, .5), "p90": percentile(cancellation, .9), "maximum": max(cancellation)}}, identity


def main() -> None:
    plan = compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise ValueError("refusing overwrite")
    source = load(); task, task_identity = budget(source["task14_evidence"]); bracket, bracket_identity = budget(source["bracket_evidence"])
    bracket["cancellation_by_ordered_pair"] = {pair: budget([row for row in source["bracket_evidence"] if row["ordered_pair"] == pair])[0]["cancellation_amplification"] for pair in sorted({row["ordered_pair"] for row in source["bracket_evidence"]})}
    identity = max(task_identity, bracket_identity)
    predictions = {"pred_a_exact_error_identity": identity <= 1e-12 and task["squared_norm_identity_error"] <= 1e-10 and bracket["squared_norm_identity_error"] <= 1e-10, "pred_b_behavior_error_budgets_reported": all(key in value for value in (task, bracket) for key in ("l2_norms", "relative_to_counterfactual_error", "baseline_effect_error_cosine", "baseline_effect_error_cross_term")), "pred_c_cancellation_reported": "cancellation_by_ordered_pair" in bracket and len(bracket["cancellation_by_ordered_pair"]) == 6, "pred_d_no_rescue_or_new_fit": compile_plan()["price"] == {"model_forwards": 0, "example_evaluations": 0, "fits": 0, "backwards": 0, "parameter_updates": 0} and source["terminal"] == "null"}
    terminal = "diagnostic_complete" if all(predictions.values()) else "invalid"
    result = {"exact_error_identity_max_absolute_error": identity, "task14": task, "bracket": bracket, "prospective_v2_status_preserved": "null", "interpretation_scope": "post-result localization only; no rescue or claim strengthening", "predictions": predictions, "terminal": terminal}
    payload = managed.atomic_create_json(OUT, {"schema": "task14_bracket_counterfactual_error_budget_audit_result_v3", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "plan": plan, "score": result, "terminal": terminal})
    print(json.dumps({"terminal": terminal, "predictions": predictions, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
