#!/usr/bin/env python3
"""Register R572 before its raw-row confirmation run."""
import json
import sys
from pathlib import Path

BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import append_artifacts, append_evidence_event, circuit_path, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

TAG = "task.numbered_list.index_successor"
EVENT = "numbered_list_conflict_confirmation.r572.preregistered.v1"

def frozen(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}

def main():
    append_artifacts(TAG, {
        "r572_prereg": frozen("basis_aligned/polynomial_causal/NUMBERED_LIST_CONFLICT_CONFIRMATION_RUNG572_PREREGISTRATION.md", "preregistration"),
        "r572_script": frozen("basis_aligned/bilinear_quotient/ops/numbered_list_conflict_confirmation_rung572.py", "implementation"),
    })
    record = json.loads(circuit_path(TAG).read_text())
    if not any(event["event_id"] == EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": EVENT, "test_type": "null_control", "stage": "preregistered", "verdict": "inconclusive",
            "failure_kind": None, "family_ids": ["list_step_two_conflict"], "site_id": None,
            "evaluation_role": "FIT_and_SELECT_raw_confirmation",
            "metrics": [
                {"name": "reference_aggregate_match", "estimate": None, "ci95": None, "bar": "all differences <=1e-6"},
                {"name": "fit_structural_preference", "estimate": None, "ci95": None, "bar": "1.0 positive fraction and bootstrap lower mean >0"},
                {"name": "select_structural_preference", "estimate": None, "ci95": None, "bar": "1.0 positive fraction and bootstrap lower mean >0"},
            ],
            "result_artifact_id": None, "prereg_artifact_id": "r572_prereg",
            "input_artifact_ids": ["r567_rows", "r569_r570_result", "r572_script"],
            "split_plan_id": "numbered_list_successor_split_r567_v1", "seed": 572,
            "checkpoint_sha256": None, "supersedes_event_id": None, "replicates_event_id": None,
            "sections": ["polynomial_causal/NUMBERED_LIST_CONFLICT_CONFIRMATION_RUNG572_PREREGISTRATION.md"],
            "claim_id": "numbered_list_index_successor.v3",
        })
    final = json.loads(circuit_path(TAG).read_text()); validate_v2(final); rebuild_registry_v2()
    print(json.dumps({"event_id": EVENT, "stage": "preregistered"}, indent=2))

if __name__ == "__main__":
    main()
