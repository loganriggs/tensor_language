#!/usr/bin/env python3
"""Register R540 cross-family DAS before optimization starts."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    append_artifacts, append_claim_revision, append_evidence_event,
    circuit_path, file_sha256, rebuild_registry_v2, validate_v2,
)

TAG = "task.bracket.pending_opener"
OLD_CLAIM, NEW_CLAIM = "pending_opener_state.v9", "pending_opener_state.v10"
EVENT = "pending_opener_cross_family_das.r540.preregistered.v1"
PREREG = "basis_aligned/polynomial_causal/PENDING_OPENER_CROSS_FAMILY_DAS_RUNG540_PREREGISTRATION.md"
IMPLEMENTATION = "basis_aligned/bilinear_quotient/ops/pending_opener_cross_family_das_rung540.py"


def frozen(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main():
    path = circuit_path(TAG)
    artifacts = {"r540_das_preregistration": frozen(PREREG, "preregistration"),
                 "r540_das_implementation": frozen(IMPLEMENTATION, "implementation")}
    append_artifacts(TAG, artifacts)
    record = json.loads(path.read_text())
    if not any(item["event_id"] == EVENT for item in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": EVENT, "claim_id": OLD_CLAIM,
            "test_type": "cross_family_transfer", "stage": "preregistered",
            "verdict": "inconclusive", "failure_kind": None,
            "family_ids": [
                "opener_type_substitution", "closed_then_reopened_type",
                "pending_state_preserved_surface_edit", "nonopener_punctuation_substitution"],
            "site_id": "residual.block8.entry.final_position",
            "split_plan_id": "pending_opener_joint_split_r537_v1",
            "evaluation_role": "FIT_optimize_SELECT_choose",
            "metrics": [
                {"name": "two_way_cross_family_normalized_recovery", "estimate": None, "ci95": None,
                 "bar": "median>=0.50, bootstrap lower>0, positive fraction>=0.75 in every target cell"},
                {"name": "answer_preserving_control_leakage", "estimate": None, "ci95": None,
                 "bar": "mean absolute<=0.10 logit and <=0.25 of full-state effect in every control cell"},
                {"name": "operational_response_equivalence", "estimate": None, "ci95": None,
                 "bar": "response cosine>=0.90 and RMS difference<=0.15 across training sources"},
            ],
            "prereg_artifact_id": "r540_das_preregistration", "result_artifact_id": None,
            "input_artifact_ids": [
                "r537_rows", "r537_controls", "r538_site_result_v2", "r539_control_result",
                "r540_das_implementation"],
            "seed": 540,
            "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            "supersedes_event_id": None, "replicates_event_id": None,
            "sections": ["PENDING_OPENER_CROSS_FAMILY_DAS_RUNG540_PREREGISTRATION.md"],
        })
    record = json.loads(path.read_text())
    if not any(item["claim_id"] == NEW_CLAIM for item in record["claims"]):
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({"claim_id": NEW_CLAIM, "revision": 10, "status": "site_live",
                      "supersedes": OLD_CLAIM,
                      "evidence_event_ids": list(previous["evidence_event_ids"]) + [EVENT],
                      "next_missing": (
                          "run the frozen 45-fit multi-seed cross-family DAS on FIT/SELECT; functional "
                          "response equivalence, not subspace overlap, decides shared identity")})
        append_claim_revision(TAG, claim)
    final = json.loads(path.read_text()); validate_v2(final); rebuild_registry_v2()
    print(json.dumps({"tag": TAG, "claim_id": NEW_CLAIM, "status": "site_live",
                      "event": EVENT, "fits": 45, "outcomes_opened": False}, indent=2))


if __name__ == "__main__": main()
