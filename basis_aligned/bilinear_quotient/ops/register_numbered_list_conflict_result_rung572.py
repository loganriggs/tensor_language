#!/usr/bin/env python3
"""Register the held R572 raw-row confirmation."""
import copy
import json
import sys
from pathlib import Path

BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import append_artifacts, append_claim_revision, append_evidence_event, circuit_path, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

TAG = "task.numbered_list.index_successor"
EVENT = "numbered_list_conflict_confirmation.r572.held.v1"
RESULT = "basis_aligned/bilinear_quotient/numbered_list_conflict_confirmation_rung572_results.json"

def main():
    append_artifacts(TAG, {"r572_result": {"path": RESULT, "sha256": file_sha256(REPO / RESULT), "kind": "result", "status": "frozen"}})
    record = json.loads(circuit_path(TAG).read_text())
    if not any(event["event_id"] == EVENT for event in record["evidence_events"]):
        result = json.loads((REPO / RESULT).read_text())
        append_evidence_event(TAG, {
            "event_id": EVENT, "test_type": "null_control", "stage": "complete", "verdict": "held",
            "failure_kind": None, "family_ids": ["list_step_two_conflict"], "site_id": None,
            "evaluation_role": "FIT_and_SELECT_raw_confirmation",
            "metrics": [
                {"name": "reference_aggregate_match", "estimate": 8.046627044677734e-7, "ci95": None, "bar": "all differences <=1e-6"},
                {"name": "fit_structural_preference", "estimate": 1.0, "ci95": [1.4634599789977074, None], "bar": "1.0 positive fraction and bootstrap lower mean >0"},
                {"name": "select_structural_preference", "estimate": 1.0, "ci95": [1.7064363367855548, None], "bar": "1.0 positive fraction and bootstrap lower mean >0"},
            ],
            "result_artifact_id": "r572_result", "prereg_artifact_id": "r572_prereg",
            "input_artifact_ids": ["r567_rows", "r569_r570_result", "r572_script"],
            "split_plan_id": "numbered_list_successor_split_r567_v1", "seed": 572,
            "checkpoint_sha256": result["checkpoint_weights_sha256"],
            "supersedes_event_id": "numbered_list_conflict_confirmation.r572.preregistered.v1", "replicates_event_id": None,
            "sections": [RESULT.removeprefix("basis_aligned/")], "claim_id": "numbered_list_index_successor.v3",
        })
    record = json.loads(circuit_path(TAG).read_text())
    if not any(claim["claim_id"] == "numbered_list_index_successor.v4" for claim in record["claims"]):
        old = next(claim for claim in record["claims"] if claim["claim_id"] == "numbered_list_index_successor.v3")
        claim = copy.deepcopy(old)
        claim.update({"claim_id": "numbered_list_index_successor.v4", "revision": 4, "status": "specified",
                      "supersedes": "numbered_list_index_successor.v3",
                      "evidence_event_ids": old["evidence_event_ids"] + [EVENT],
                      "next_missing": "preregister exact L8H7/L8H3 final-label/all-label score-versus-value factor localization with all four invariance families"})
        append_claim_revision(TAG, claim)
    final = json.loads(circuit_path(TAG).read_text()); validate_v2(final); rebuild_registry_v2()
    print(json.dumps({"event_id": EVENT, "claim_id": "numbered_list_index_successor.v4", "verdict": "held"}, indent=2))

if __name__ == "__main__":
    main()
