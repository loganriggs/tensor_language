#!/usr/bin/env python3
"""Register R569 and R570 before opening either FIT outcome."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import append_artifacts, append_evidence_event, circuit_path, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

SCRIPT = "basis_aligned/bilinear_quotient/ops/numeric_two_hypothesis_capability_rung569_570.py"


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def register(tag: str, event_id: str, claim_id: str, prereg_id: str, prereg_path: str,
             split_id: str, families: list[str], metrics: list[dict], input_ids: list[str]) -> None:
    append_artifacts(tag, {
        prereg_id: frozen(prereg_path, "preregistration"),
        "r569_r570_script": frozen(SCRIPT, "implementation"),
    })
    record = json.loads(circuit_path(tag).read_text())
    if not any(event["event_id"] == event_id for event in record["evidence_events"]):
        append_evidence_event(tag, {
            "event_id": event_id, "test_type": "capability", "stage": "preregistered", "verdict": "inconclusive",
            "failure_kind": None, "family_ids": families, "site_id": None,
            "evaluation_role": "FIT_then_hypothesis_conditional_SELECT", "metrics": metrics,
            "result_artifact_id": None, "prereg_artifact_id": prereg_id,
            "input_artifact_ids": input_ids + ["r569_r570_script"], "split_plan_id": split_id,
            "seed": 569 if tag == "task.numbered_list.index_successor" else 570,
            "checkpoint_sha256": None, "supersedes_event_id": None, "replicates_event_id": None,
            "sections": [prereg_path.removeprefix("basis_aligned/")], "claim_id": claim_id,
        })


def main() -> None:
    register(
        "task.numbered_list.index_successor", "numbered_list_native_capability.r569.preregistered.v1",
        "numbered_list_index_successor.v2", "r569_prereg",
        "basis_aligned/polynomial_causal/NUMBERED_LIST_NATIVE_CAPABILITY_RUNG569_PREREGISTRATION.md",
        "numbered_list_successor_split_r567_v1",
        ["list_two_line_state_shift", "list_three_line_state_shift", "list_surface_preserved",
         "list_middle_index_break", "list_repeated_index_control", "list_step_two_conflict"],
        [
            {"name": "state_and_invariance_candidate_margin", "estimate": None, "ci95": None,
             "bar": ">=0.75 correct and bootstrap lower mean margin >0 in every endpoint cell"},
            {"name": "step_two_structural_conflict", "estimate": None, "ci95": None,
             "bar": ">=0.75 favor final-label+1 and bootstrap lower mean margin >0"},
            {"name": "split_opening", "estimate": None, "ci95": None,
             "bar": "list SELECT opens only after complete list FIT pass; FINAL_TEST/OOD closed"},
        ], ["r567_rows", "r567_receipt", "r568_role_overlay"],
    )
    register(
        "task.numeric_sequence.continuation", "numeric_sequence_native_capability.r570.preregistered.v1",
        "numeric_sequence_continuation.v1", "r570_prereg",
        "basis_aligned/polynomial_causal/NUMERIC_SEQUENCE_NATIVE_CAPABILITY_RUNG570_PREREGISTRATION.md",
        "numeric_sequence_continuation_split_r567_v1",
        ["sequence_digit_state_shift", "sequence_word_state_shift", "sequence_cross_format_shift",
         "sequence_digit_surface_preserved", "sequence_word_surface_preserved", "sequence_middle_value_break",
         "sequence_digit_copy_control", "sequence_word_copy_control", "sequence_step_two_conflict"],
        [
            {"name": "state_and_invariance_candidate_margin", "estimate": None, "ci95": None,
             "bar": ">=0.75 correct and bootstrap lower mean margin >0 in every endpoint cell"},
            {"name": "middle_value_necessity", "estimate": None, "ci95": None,
             "bar": ">=0.65 positive drops and bootstrap lower mean drop >0"},
            {"name": "split_opening", "estimate": None, "ci95": None,
             "bar": "sequence SELECT opens only after complete sequence FIT pass; FINAL_TEST/OOD closed"},
        ], ["r567_rows", "r567_receipt"],
    )
    for tag in ("task.numbered_list.index_successor", "task.numeric_sequence.continuation"):
        validate_v2(json.loads(circuit_path(tag).read_text()))
    rebuild_registry_v2()
    print(json.dumps({"r569": "preregistered", "r570": "preregistered", "outcomes_opened": []}, indent=2))


if __name__ == "__main__":
    main()
