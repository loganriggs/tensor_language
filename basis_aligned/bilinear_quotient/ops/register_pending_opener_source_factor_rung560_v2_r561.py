#!/usr/bin/env python3
"""Register R560 v1 invalidity, v2 repair, and pre-outcome R561 audit."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    append_artifacts, append_claim_revision, append_evidence_event, circuit_path,
    file_sha256, rebuild_registry_v2, validate_v2,
)

TAG = "task.bracket.pending_opener"
OLD_CLAIM = "pending_opener_state.v26"
NEW_CLAIM = "pending_opener_state.v27"
V1_PLAN = "pending_opener_source_factor_interchange.r560.preregistered.v1"
V1_INVALID = "pending_opener_source_factor_interchange.r560.v1.invalid_implementation"
V2_PLAN = "pending_opener_source_factor_interchange.r560.v2.preregistered.v1"
AUDIT_PLAN = "pending_opener_source_factor_audit.r561.preregistered.v1"
SPLIT_ID = "pending_opener_three_value_fresh_split_r545_v1"
PATHS = {
    "r560_v1_failed_runlog": (
        "basis_aligned/bilinear_quotient/runlogs/pending_opener_source_factor_interchange_rung560.log", "runlog"),
    "r560_v2_correction": (
        "basis_aligned/polynomial_causal/PENDING_OPENER_SOURCE_FACTOR_RUNG560_V2_CORRECTION.md", "correction"),
    "r560_v2_implementation": (
        "basis_aligned/bilinear_quotient/ops/pending_opener_source_factor_interchange_rung560_v2.py", "implementation"),
    "r560_v2_test": (
        "basis_aligned/bilinear_quotient/ops/test_pending_opener_source_factor_interchange_rung560_v2.py", "test"),
    "r561_source_factor_audit_preregistration": (
        "basis_aligned/polynomial_causal/PENDING_OPENER_SOURCE_FACTOR_AUDIT_RUNG561_PREREGISTRATION.md", "preregistration"),
    "r561_source_factor_audit_implementation": (
        "basis_aligned/bilinear_quotient/ops/pending_opener_source_factor_audit_rung561.py", "audit_implementation"),
    "r561_source_factor_audit_test": (
        "basis_aligned/bilinear_quotient/ops/test_pending_opener_source_factor_audit_rung561.py", "test"),
}
FAMILIES = [
    "direct_three_value_type_substitution", "completed_then_reopened_three_value_order",
    "pending_type_preserved_surface_rewrite", "pending_type_preserved_distance_extension",
    "pending_type_preserved_nonopener_punctuation",
]
METRICS = [
    {"name": "target_factor_recovery", "estimate": None, "ci95": None,
     "bar": "every family/direction median>=0.50, bootstrap lower mean>0, positive fraction>=0.75"},
    {"name": "answer_preserving_closer_change", "estimate": None, "ci95": None,
     "bar": "every family/direction mean absolute<=0.10 and <=0.25 complete-head effect"},
    {"name": "answer_preserving_full_vocab_change", "estimate": None, "ci95": None,
     "bar": "every family/direction mean RMS <=0.25 complete-head RMS"},
    {"name": "adjacent_wrong_source_recovery", "estimate": None, "ci95": None,
     "bar": "every target family/direction mean absolute<=0.25"},
    {"name": "score_payload_interaction", "estimate": None, "ci95": None,
     "bar": "reported; not used to weaken selection bars"},
]


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def event(event_id: str, stage: str, verdict: str, failure_kind, prereg, result, supersedes, inputs, notes):
    return {
        "event_id": event_id, "claim_id": OLD_CLAIM, "test_type": "composition",
        "stage": stage, "verdict": verdict, "failure_kind": failure_kind,
        "family_ids": FAMILIES,
        "site_id": "attention13.head8.semantic_source_score_times_projected_value.final_query",
        "split_plan_id": SPLIT_ID, "evaluation_role": "FIT_factor_choice_then_single_SELECT_validation",
        "metrics": METRICS, "prereg_artifact_id": prereg, "result_artifact_id": result,
        "input_artifact_ids": inputs, "seed": 560,
        "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "supersedes_event_id": supersedes, "replicates_event_id": None,
        "sections": ["PENDING_OPENER_SOURCE_FACTOR_INTERCHANGE_RUNG560_PREREGISTRATION.md",
                     "PENDING_OPENER_SOURCE_FACTOR_RUNG560_V2_CORRECTION.md"],
        "notes": notes,
    }


def main() -> None:
    assert not (BQ / "pending_opener_source_factor_interchange_rung560_results.json").exists()
    runlog = (REPO / PATHS["r560_v1_failed_runlog"][0]).read_text()
    assert "KeyError: 'direct_three_value_type_substitution'" in runlog
    append_artifacts(TAG, {key: frozen(*value) for key, value in PATHS.items()})
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(item["event_id"] == V1_INVALID for item in record["evidence_events"]):
        append_evidence_event(TAG, event(
            V1_INVALID, "invalid", "invalid", "implementation_failure",
            "r560_source_factor_preregistration", "r560_v1_failed_runlog", V1_PLAN,
            ["r560_source_factor_implementation", "r560_source_position_audit"],
            "FIT model evaluation completed, but no result was written and no model value was displayed; split envelope KeyError.",
        ))
    record = json.loads(path.read_text())
    if not any(item["event_id"] == V2_PLAN for item in record["evidence_events"]):
        append_evidence_event(TAG, event(
            V2_PLAN, "preregistered", "inconclusive", None,
            "r560_source_factor_preregistration", None, V1_INVALID,
            ["r560_source_position_audit", "r560_v2_correction", "r560_v2_implementation", "r560_v2_test"],
            "Shape-only wrapper selects raw[split]; scientific protocol is unchanged from v1.",
        ))
    record = json.loads(path.read_text())
    if not any(item["event_id"] == AUDIT_PLAN for item in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": AUDIT_PLAN, "claim_id": OLD_CLAIM, "test_type": "null_control",
            "stage": "preregistered", "verdict": "inconclusive", "failure_kind": None,
            "family_ids": FAMILIES, "site_id": "r560_saved_source_factor_statistics",
            "split_plan_id": SPLIT_ID, "evaluation_role": "pre_outcome_CPU_terminal_audit",
            "metrics": [
                {"name": "all_target_control_and_wrong_source_cells_recomputed", "estimate": None,
                 "ci95": None, "bar": "exact agreement"},
                {"name": "FIT_choice_SELECT_and_interactions_recomputed", "estimate": None,
                 "ci95": None, "bar": "exact agreement"},
            ],
            "prereg_artifact_id": "r561_source_factor_audit_preregistration", "result_artifact_id": None,
            "input_artifact_ids": ["r560_v2_implementation", "r561_source_factor_audit_implementation",
                                   "r561_source_factor_audit_test"],
            "seed": None, "checkpoint_sha256": None,
            "supersedes_event_id": None, "replicates_event_id": None,
            "sections": ["PENDING_OPENER_SOURCE_FACTOR_AUDIT_RUNG561_PREREGISTRATION.md"],
            "notes": "Frozen before the v2 result; imports no R560 scorer and makes zero model calls.",
        })
    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({"claim_id": NEW_CLAIM, "revision": 27, "status": "specified", "supersedes": OLD_CLAIM,
                      "evidence_event_ids": list(previous["evidence_event_ids"]) + [V1_INVALID, V2_PLAN, AUDIT_PLAN],
                      "next_missing": (
                          "execute the shape-only R560 v2 repair and immediately apply the pre-outcome R561 CPU audit; "
                          "keep FINAL_TEST/OOD closed"
                      )})
        append_claim_revision(TAG, claim)
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({"tag": TAG, "claim_id": NEW_CLAIM, "v1": "invalid_implementation",
                      "v2": "preregistered", "audit": "preregistered", "model_outcomes_opened": False}, indent=2))


if __name__ == "__main__":
    main()
