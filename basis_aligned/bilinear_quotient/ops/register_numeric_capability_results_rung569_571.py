#!/usr/bin/env python3
"""Register the held R569/R570 gates and R571 structural audit."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import append_artifacts, append_claim_revision, append_evidence_event, circuit_path, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

RESULT_PATH = "basis_aligned/bilinear_quotient/numeric_two_hypothesis_capability_rung569_570_results.json"
AUDIT_PATH = "basis_aligned/bilinear_quotient/numeric_two_hypothesis_capability_rung571_audit.json"
AUDIT_SCRIPT = "basis_aligned/bilinear_quotient/ops/audit_numeric_two_hypothesis_capability_rung571.py"


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def register_result(tag: str, event_id: str, prereg_event: str, claim_id: str, prereg_id: str,
                    split_id: str, families: list[str], metrics: list[dict], seed: int) -> None:
    record = json.loads(circuit_path(tag).read_text())
    if not any(event["event_id"] == event_id for event in record["evidence_events"]):
        append_evidence_event(tag, {
            "event_id": event_id, "test_type": "capability", "stage": "complete", "verdict": "held",
            "failure_kind": None, "family_ids": families, "site_id": None,
            "evaluation_role": "FIT_then_hypothesis_conditional_SELECT", "metrics": metrics,
            "result_artifact_id": "r569_r570_result", "prereg_artifact_id": prereg_id,
            "input_artifact_ids": ["r567_rows", "r567_receipt", "r569_r570_script"],
            "split_plan_id": split_id, "seed": seed,
            "checkpoint_sha256": json.loads((REPO / RESULT_PATH).read_text())["checkpoint_weights_sha256"],
            "supersedes_event_id": prereg_event, "replicates_event_id": None,
            "sections": [RESULT_PATH.removeprefix("basis_aligned/")], "claim_id": claim_id,
        })


def register_audit(tag: str, event_id: str, claim_id: str, split_id: str, families: list[str]) -> None:
    record = json.loads(circuit_path(tag).read_text())
    if not any(event["event_id"] == event_id for event in record["evidence_events"]):
        append_evidence_event(tag, {
            "event_id": event_id, "test_type": "null_control", "stage": "complete", "verdict": "held",
            "failure_kind": None, "family_ids": families, "site_id": None,
            "evaluation_role": "post_result_structural_audit",
            "metrics": [
                {"name": "saved_statistics_recomputed", "estimate": 1344, "ci95": None, "bar": "exactly 1344 across both hypotheses"},
                {"name": "decision_cells_recomputed", "estimate": 50, "ci95": None, "bar": "exactly 50 across both hypotheses"},
                {"name": "terminal_decisions_reproduced", "estimate": 1.0, "ci95": None, "bar": "true"},
            ],
            "result_artifact_id": "r571_audit", "prereg_artifact_id": None,
            "input_artifact_ids": ["r569_r570_result", "r571_audit_script"], "split_plan_id": split_id,
            "seed": 571, "checkpoint_sha256": None, "supersedes_event_id": None, "replicates_event_id": None,
            "sections": [AUDIT_PATH.removeprefix("basis_aligned/")], "claim_id": claim_id,
        })


def revise(tag: str, old_id: str, new_id: str, revision: int, result_event: str, audit_event: str,
           next_missing: str, validated_except: set[str] | None = None) -> None:
    record = json.loads(circuit_path(tag).read_text())
    if any(claim["claim_id"] == new_id for claim in record["claims"]):
        return
    old = next(claim for claim in record["claims"] if claim["claim_id"] == old_id)
    claim = copy.deepcopy(old)
    claim.update({"claim_id": new_id, "revision": revision, "status": "specified", "supersedes": old_id,
                  "evidence_event_ids": old["evidence_event_ids"] + [result_event, audit_event],
                  "next_missing": next_missing})
    for family in claim["counterfactual_families"]:
        family["status"] = "frozen" if validated_except and family["family_id"] in validated_except else "validated"
    append_claim_revision(tag, claim)


def main() -> None:
    common_artifacts = {"r569_r570_result": frozen(RESULT_PATH, "result"),
                        "r571_audit": frozen(AUDIT_PATH, "audit"),
                        "r571_audit_script": frozen(AUDIT_SCRIPT, "implementation")}
    list_tag, seq_tag = "task.numbered_list.index_successor", "task.numeric_sequence.continuation"
    append_artifacts(list_tag, common_artifacts)
    append_artifacts(seq_tag, common_artifacts)
    list_families = ["list_two_line_state_shift", "list_three_line_state_shift", "list_surface_preserved",
                     "list_middle_index_break", "list_repeated_index_control", "list_step_two_conflict"]
    seq_families = ["sequence_digit_state_shift", "sequence_word_state_shift", "sequence_cross_format_shift",
                    "sequence_digit_surface_preserved", "sequence_word_surface_preserved", "sequence_middle_value_break",
                    "sequence_digit_copy_control", "sequence_word_copy_control", "sequence_step_two_conflict"]
    register_result(list_tag, "numbered_list_native_capability.r569.held.v1",
                    "numbered_list_native_capability.r569.preregistered.v1", "numbered_list_index_successor.v2",
                    "r569_prereg", "numbered_list_successor_split_r567_v1", list_families, [
                        {"name": "state_and_invariance_candidate_margin", "estimate": 0.75, "ci95": None,
                         "bar": ">=0.75 correct and bootstrap lower mean margin >0 in every endpoint cell"},
                        {"name": "step_two_structural_conflict", "estimate": 1.0, "ci95": None,
                         "bar": ">=0.75 favor final-label+1 and bootstrap lower mean margin >0"},
                        {"name": "split_opening", "estimate": 1.0, "ci95": None,
                         "bar": "list SELECT opens only after complete list FIT pass; FINAL_TEST/OOD closed"},
                    ], 569)
    register_result(seq_tag, "numeric_sequence_native_capability.r570.held.v1",
                    "numeric_sequence_native_capability.r570.preregistered.v1", "numeric_sequence_continuation.v1",
                    "r570_prereg", "numeric_sequence_continuation_split_r567_v1", seq_families, [
                        {"name": "state_and_invariance_candidate_margin", "estimate": 1.0, "ci95": None,
                         "bar": ">=0.75 correct and bootstrap lower mean margin >0 in every endpoint cell"},
                        {"name": "middle_value_necessity", "estimate": 1.0, "ci95": None,
                         "bar": ">=0.65 positive drops and bootstrap lower mean drop >0"},
                        {"name": "split_opening", "estimate": 1.0, "ci95": None,
                         "bar": "sequence SELECT opens only after complete sequence FIT pass; FINAL_TEST/OOD closed"},
                    ], 570)
    register_audit(list_tag, "numbered_list_capability_audit.r571.held.v1", "numbered_list_index_successor.v2",
                   "numbered_list_successor_split_r567_v1", list_families)
    register_audit(seq_tag, "numeric_sequence_capability_audit.r571.held.v1", "numeric_sequence_continuation.v1",
                   "numeric_sequence_continuation_split_r567_v1", seq_families)
    revise(list_tag, "numbered_list_index_successor.v2", "numbered_list_index_successor.v3", 3,
           "numbered_list_native_capability.r569.held.v1", "numbered_list_capability_audit.r571.held.v1",
           "run raw-row R572 confirmation of the list +2 conflict, then preregister exact L8H7/L8H3 value-path ceilings")
    revise(seq_tag, "numeric_sequence_continuation.v1", "numeric_sequence_continuation.v2", 2,
           "numeric_sequence_native_capability.r570.held.v1", "numeric_sequence_capability_audit.r571.held.v1",
           "preregister a complete-state site screen that tests digit, word, and cross-format interchange plus copy/+2 selectivity",
           {"sequence_step_two_conflict"})
    for tag in (list_tag, seq_tag):
        validate_v2(json.loads(circuit_path(tag).read_text()))
    rebuild_registry_v2()
    print(json.dumps({"r569": "held", "r570": "held", "r571": "held_structural_audit"}, indent=2))


if __name__ == "__main__":
    main()
