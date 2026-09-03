#!/usr/bin/env python3
"""Preserve R538's invalid first receipt and bind its verifier-only rerun."""

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
OLD_CLAIM = "pending_opener_state.v5"
NEW_CLAIM = "pending_opener_state.v6"
OLD_PREREG_EVENT = "pending_opener_common_site_ceiling.r537.preregistered.v1"
INVALID_EVENT = "pending_opener_common_site_ceiling.r538.invalid_unverified_checkpoint.v1"
INVALID_RESULT = "basis_aligned/bilinear_quotient/pending_opener_common_site_rung538_invalid_unverified_checkpoint_results.json"
V2_IMPLEMENTATION = "basis_aligned/bilinear_quotient/ops/pending_opener_common_site_rung538_v2.py"


def frozen(path: str, kind: str) -> dict:
    source = REPO / path
    assert source.is_file(), path
    return {"path": path, "sha256": file_sha256(source), "kind": kind, "status": "frozen"}


def main() -> None:
    invalid = json.loads((REPO / INVALID_RESULT).read_text())
    assert invalid["pred_b_common_live_site"] is True
    artifacts = {
        "r538_site_invalid_unverified_checkpoint_result": frozen(INVALID_RESULT, "invalid_result"),
        "r538_site_implementation_v2": frozen(V2_IMPLEMENTATION, "implementation"),
    }
    append_artifacts(TAG, artifacts)
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(item["event_id"] == INVALID_EVENT for item in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": INVALID_EVENT,
            "claim_id": "pending_opener_state.v3",
            "test_type": "full_swap_ceiling",
            "stage": "invalid",
            "verdict": "invalid",
            "failure_kind": "invalid_instrument",
            "family_ids": ["opener_type_substitution", "closed_then_reopened_type"],
            "site_id": None,
            "split_plan_id": "pending_opener_joint_split_r537_v1",
            "evaluation_role": "FIT_SELECT_only",
            "metrics": [
                {
                    "name": "signed_donorward_movement", "estimate": None, "ci95": None,
                    "bar": "positive both directions with group-bootstrap lower bound>0",
                },
                {
                    "name": "individual_direction_success", "estimate": None, "ci95": None,
                    "bar": ">=0.70 at one common site",
                },
            ],
            "prereg_artifact_id": "r537_preregistration",
            "result_artifact_id": "r538_site_invalid_unverified_checkpoint_result",
            "input_artifact_ids": [
                "r537_rows", "r537_rows_receipt", "r537_capability_result",
                "r538_site_preregistration", "r538_site_implementation",
            ],
            "seed": 538,
            "checkpoint_sha256": None,
            "supersedes_event_id": OLD_PREREG_EVENT,
            "replicates_event_id": None,
            "sections": ["PENDING_OPENER_COMMON_SITE_RUNG538_PREREGISTRATION.md"],
            "notes": (
                "Scientific thresholds and outcomes are not scored: the implementation compared "
                "the expected checkpoint hash with itself instead of verifying loaded bytes."
            ),
        })

    record = json.loads(path.read_text())
    if not any(item["claim_id"] == NEW_CLAIM for item in record["claims"]):
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM, "revision": 6, "status": "specified",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": [
                "pending_opener_capability.r537.complete.v1", INVALID_EVENT,
            ],
            "next_missing": (
                "rerun the unchanged 15-site contract through the source-closed loader that verifies "
                "the actual checkpoint bytes; preserve the first outcome as invalid"
            ),
        })
        append_claim_revision(TAG, claim)
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "status": "specified",
        "invalid_event": INVALID_EVENT,
        "correction": "actual checkpoint byte verification; scientific contract unchanged",
    }, indent=2))


if __name__ == "__main__":
    main()
