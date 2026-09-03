#!/usr/bin/env python3
"""Register R546 before any R545 model outcome."""

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
OLD_CLAIM, NEW_CLAIM = "pending_opener_state.v17", "pending_opener_state.v18"
EVENT = "pending_opener_three_value_confirmation.r546.preregistered.v1"
PREREG = "basis_aligned/polynomial_causal/PENDING_OPENER_THREE_VALUE_CONFIRMATION_RUNG546_PREREGISTRATION.md"
IMPLEMENTATION = "basis_aligned/bilinear_quotient/ops/pending_opener_three_value_confirmation_rung546.py"
TEST = "basis_aligned/bilinear_quotient/ops/test_pending_opener_three_value_confirmation_rung546.py"


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    append_artifacts(TAG, {
        "r546_three_value_confirmation_preregistration": frozen(PREREG, "preregistration"),
        "r546_three_value_confirmation_implementation": frozen(IMPLEMENTATION, "implementation"),
        "r546_three_value_confirmation_test": frozen(TEST, "test"),
    })
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    family_ids = [family["family_id"] for family in next(
        claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)["counterfactual_families"]]
    if not any(event["event_id"] == EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": EVENT, "claim_id": OLD_CLAIM, "test_type": "full_swap_ceiling",
            "stage": "preregistered", "verdict": "inconclusive", "failure_kind": None,
            "family_ids": family_ids, "site_id": "attention13.head8.output.final_position",
            "split_plan_id": "pending_opener_three_value_fresh_split_r545_v1",
            "evaluation_role": "fresh_FIT_SELECT_confirmation_only",
            "metrics": [
                {"name": "three_value_native_capability", "estimate": None, "ci95": None,
                 "bar": ">=75% correct per ordered pair plus pooled bootstrap-lower margin>0"},
                {"name": "l13h8_target_complete_state_interchange", "estimate": None, "ci95": None,
                 "bar": "both target families/directions pass pooled and every ordered-pair movement bars"},
                {"name": "l13h8_answer_preserving_control_liveness", "estimate": None, "ci95": None,
                 "bar": "all three controls/directions have bootstrap-lower abs endpoint>0.03 and logit RMS>0.01"},
            ],
            "prereg_artifact_id": "r546_three_value_confirmation_preregistration",
            "result_artifact_id": None,
            "input_artifact_ids": [
                "r545_three_value_rows", "r545_three_value_rows_receipt", "r545_three_value_rows_builder",
                "r545_three_value_rows_test", "r546_three_value_confirmation_implementation",
                "r546_three_value_confirmation_test",
            ],
            "seed": 546,
            "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            "supersedes_event_id": None, "replicates_event_id": None,
            "sections": ["PENDING_OPENER_THREE_VALUE_CONFIRMATION_RUNG546_PREREGISTRATION.md"],
        })
    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM, "revision": 18, "status": "specified", "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [EVENT],
            "next_missing": (
                "execute and independently audit the frozen 204-forward R546 FIT/SELECT confirmation; no projector "
                "fit and no FINAL_TEST/OOD access before its verdict"
            ),
        })
        append_claim_revision(TAG, claim)
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "status": "specified", "event": EVENT,
        "planned_forwards": 204, "model_outcomes_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
