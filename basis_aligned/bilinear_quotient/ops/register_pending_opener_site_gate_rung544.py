#!/usr/bin/env python3
"""Register the frozen R544 capability/site experiment before execution."""

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
OLD_CLAIM, NEW_CLAIM = "pending_opener_state.v13", "pending_opener_state.v14"
EVENT = "pending_opener_four_closer_site_gate.r544.preregistered.v1"
PREREG = "basis_aligned/polynomial_causal/PENDING_OPENER_FOUR_CLOSER_SITE_GATE_RUNG544_PREREGISTRATION.md"
IMPLEMENTATION = "basis_aligned/bilinear_quotient/ops/pending_opener_four_closer_site_gate_rung544.py"
TEST = "basis_aligned/bilinear_quotient/ops/test_pending_opener_four_closer_site_gate_rung544.py"


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    artifacts = {
        "r544_site_gate_preregistration": frozen(PREREG, "preregistration"),
        "r544_site_gate_implementation": frozen(IMPLEMENTATION, "implementation"),
        "r544_site_gate_test": frozen(TEST, "test"),
    }
    append_artifacts(TAG, artifacts)
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(item["event_id"] == EVENT for item in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": EVENT,
            "claim_id": OLD_CLAIM,
            "test_type": "full_swap_ceiling",
            "stage": "preregistered",
            "verdict": "inconclusive",
            "failure_kind": None,
            "family_ids": [
                "direct_four_closer_type_substitution",
                "completed_then_reopened_four_closer_order",
                "pending_type_preserved_surface_paraphrase",
                "pending_type_preserved_distance_shift",
                "pending_type_preserved_nonopener_punctuation",
            ],
            "site_id": None,
            "split_plan_id": "pending_opener_unique_joint_split_r543_v2",
            "evaluation_role": "FIT_SELECT_only",
            "metrics": [
                {"name": "four_closer_native_capability", "estimate": None, "ci95": None,
                 "bar": ">=75% correct per ordered pair and pooled bootstrap-lower closer margin>0"},
                {"name": "target_full_state_interchange", "estimate": None, "ci95": None,
                 "bar": "both families/directions pass pooled and every ordered-pair positive-movement bars"},
                {"name": "answer_preserving_full_state_liveness", "estimate": None, "ci95": None,
                 "bar": "all three families/directions have bootstrap-lower abs endpoint>0.03 and logit RMS>0.01"},
            ],
            "prereg_artifact_id": "r544_site_gate_preregistration",
            "result_artifact_id": None,
            "input_artifact_ids": [
                "r543_v2_rows", "r543_v2_rows_receipt", "r543_v2_rows_builder",
                "r543_v2_rows_test", "r543_v2_correction", "r544_site_gate_implementation",
                "r544_site_gate_test",
            ],
            "seed": 544,
            "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["PENDING_OPENER_FOUR_CLOSER_SITE_GATE_RUNG544_PREREGISTRATION.md"],
        })
    record = json.loads(path.read_text())
    if not any(item["claim_id"] == NEW_CLAIM for item in record["claims"]):
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 14,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [EVENT],
            "next_missing": (
                "execute the frozen 450-forward FIT/SELECT four-closer capability and full-state gate; "
                "no subspace fit and no FINAL_TEST/OOD access before its result is audited"
            ),
        })
        append_claim_revision(TAG, claim)
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "status": "specified",
        "event": EVENT, "planned_forwards": 450, "outcomes_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
