#!/usr/bin/env python3
"""Register the completed R537 capability gate and advance its claim."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    append_artifacts,
    append_claim_revision,
    append_evidence_event,
    circuit_path,
    file_sha256,
    rebuild_registry_v2,
    validate_v2,
)

TAG = "task.bracket.pending_opener"
OLD_CLAIM = "pending_opener_state.v3"
NEW_CLAIM = "pending_opener_state.v4"
PREREG_EVENT = "pending_opener_capability.r537.preregistered.v1"
COMPLETE_EVENT = "pending_opener_capability.r537.complete.v1"
RESULT_PATH = "basis_aligned/bilinear_quotient/pending_opener_capability_rung537_results.json"
IMPLEMENTATION_PATH = "basis_aligned/bilinear_quotient/ops/pending_opener_capability_rung537.py"


def frozen(path: str, kind: str) -> dict:
    source = REPO / path
    assert source.is_file(), path
    return {"path": path, "sha256": file_sha256(source), "kind": kind, "status": "frozen"}


def main() -> None:
    result = json.loads((REPO / RESULT_PATH).read_text())
    assert result["pred_e_dataset_authorized_for_site_screen"] is True
    assert result["forbidden_splits_opened"] == []
    assert result["model_forwards"] == 32 and result["model_backwards"] == 0
    for split in ("FIT", "SELECT"):
        for summary in result["summaries"][split].values():
            assert summary["passed"] is True

    artifacts = {
        "r537_capability_result": frozen(RESULT_PATH, "result"),
        "r537_capability_implementation": frozen(IMPLEMENTATION_PATH, "implementation"),
    }
    append_artifacts(TAG, artifacts)

    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(item["event_id"] == COMPLETE_EVENT for item in record["evidence_events"]):
        interchange = [
            result["summaries"][split][family]
            for split in ("FIT", "SELECT")
            for family in ("opener_type_substitution", "closed_then_reopened_type")
        ]
        event = {
            "event_id": COMPLETE_EVENT,
            "claim_id": OLD_CLAIM,
            "test_type": "capability",
            "stage": "complete",
            "verdict": "held",
            "failure_kind": None,
            "family_ids": [
                "opener_type_substitution", "closed_then_reopened_type",
                "pending_state_preserved_surface_edit", "nonopener_punctuation_substitution",
            ],
            "site_id": None,
            "split_plan_id": "pending_opener_joint_split_r537_v1",
            "evaluation_role": "FIT_SELECT_only",
            "metrics": [
                {
                    "name": "two_endpoint_correct_fraction",
                    "estimate": min(x["both_endpoints_correct_fraction"] for x in interchange),
                    "ci95": None,
                    "bar": ">=0.75 per answer-changing family on FIT and SELECT",
                },
                {
                    "name": "symmetric_logit_separation",
                    "estimate": min(x["mean_symmetric_logit_separation"] for x in interchange),
                    "ci95": [min(x["bootstrap95_lower_symmetric_separation"] for x in interchange), None],
                    "bar": "mean>0.5 and group-bootstrap lower bound>0",
                },
            ],
            "prereg_artifact_id": "r537_preregistration",
            "result_artifact_id": "r537_capability_result",
            "input_artifact_ids": [
                "r537_rows", "r537_rows_receipt", "r537_controls",
                "r537_controls_receipt", "r537_capability_implementation",
            ],
            "seed": 537,
            "checkpoint_sha256": None,
            "supersedes_event_id": PREREG_EVENT,
            "replicates_event_id": None,
            "sections": ["PENDING_OPENER_MULTI_COUNTERFACTUAL_RUNG537_PREREGISTRATION.md"],
        }
        append_evidence_event(TAG, event)

    record = json.loads(path.read_text())
    if not any(item["claim_id"] == NEW_CLAIM for item in record["claims"]):
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 4,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": [
                COMPLETE_EVENT,
                "pending_opener_common_site_ceiling.r537.preregistered.v1",
            ],
            "next_missing": (
                "run the frozen common-site full-state interchange screen on FIT/SELECT; "
                "do not fit a DAS projector unless one site is live for both answer-changing families"
            ),
        })
        append_claim_revision(TAG, claim)

    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "status": "specified",
        "event_id": COMPLETE_EVENT,
        "result_sha256": artifacts["r537_capability_result"]["sha256"],
        "next": "common_site_full_state_interchange",
    }, indent=2))


if __name__ == "__main__":
    main()
