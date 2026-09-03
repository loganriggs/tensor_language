#!/usr/bin/env python3
"""Promote pending-opener to specified and preregister capability/site screens."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    append_claim_revision,
    append_evidence_event,
    circuit_path,
    file_sha256,
    rebuild_registry_v2,
    validate_v2,
)

TAG = "task.bracket.pending_opener"
CLAIM_ID = "pending_opener_state.v3"


def frozen(path: str, kind: str) -> dict:
    source = REPO / path
    assert source.is_file(), path
    return {"path": path, "sha256": file_sha256(source), "kind": kind, "status": "frozen"}


ARTIFACTS = {
    "r537_controls": frozen("basis_aligned/bilinear_quotient/pending_opener_controls_rung537.json", "dataset"),
    "r537_controls_receipt": frozen("basis_aligned/bilinear_quotient/pending_opener_controls_rung537_receipt.json", "split"),
    "r537_controls_builder": frozen("basis_aligned/bilinear_quotient/ops/pending_opener_controls_rung537.py", "builder"),
}


def make_claim(record: dict) -> dict:
    previous = next(claim for claim in record["claims"] if claim["claim_id"] == "pending_opener_state.v2")
    claim = copy.deepcopy(previous)
    claim.update({
        "claim_id": CLAIM_ID,
        "revision": 3,
        "status": "specified",
        "supersedes": "pending_opener_state.v2",
        "evidence_event_ids": [],
        "next_missing": "run preregistered FIT/SELECT capability gates; only then open the common-site ceiling ladder",
    })
    claim["counterfactual_families"].append({
        "family_id": "nonopener_punctuation_substitution",
        "role": "invariance",
        "changes": ["one comma token to one colon token before the opener"],
        "holds_fixed": ["pending parenthesis", "correct closer", "token length", "all tokens after the edited punctuation"],
        "builder_artifact_id": "r537_controls_builder",
        "control_ids": ["wrong square closer", "wrong curly closer", "random subspace"],
        "split_plan_id": "pending_opener_joint_split_r537_v1",
        "status": "frozen",
    })
    return claim


def event(event_id: str, test_type: str, family_ids: list[str], metrics: list[dict]) -> dict:
    return {
        "event_id": event_id,
        "claim_id": CLAIM_ID,
        "test_type": test_type,
        "stage": "preregistered",
        "verdict": "inconclusive",
        "failure_kind": None,
        "family_ids": family_ids,
        "site_id": None,
        "split_plan_id": "pending_opener_joint_split_r537_v1",
        "evaluation_role": "FIT_SELECT_only",
        "metrics": metrics,
        "prereg_artifact_id": "r537_preregistration",
        "result_artifact_id": None,
        "input_artifact_ids": ["r537_rows", "r537_rows_receipt", "r537_controls", "r537_controls_receipt"],
        "seed": None,
        "checkpoint_sha256": None,
        "supersedes_event_id": None,
        "replicates_event_id": None,
        "sections": ["PENDING_OPENER_MULTI_COUNTERFACTUAL_RUNG537_PREREGISTRATION.md"],
    }


EVENTS = [
    event(
        "pending_opener_capability.r537.preregistered.v1",
        "capability",
        ["opener_type_substitution", "closed_then_reopened_type", "pending_state_preserved_surface_edit", "nonopener_punctuation_substitution"],
        [
            {"name": "two_endpoint_correct_fraction", "estimate": None, "ci95": None, "bar": ">=0.75 per answer-changing family on FIT and SELECT"},
            {"name": "symmetric_logit_separation", "estimate": None, "ci95": None, "bar": "mean>0.5 and group-bootstrap lower bound>0"},
        ],
    ),
    event(
        "pending_opener_common_site_ceiling.r537.preregistered.v1",
        "full_swap_ceiling",
        ["opener_type_substitution", "closed_then_reopened_type"],
        [
            {"name": "signed_donorward_movement", "estimate": None, "ci95": None, "bar": "positive both directions with group-bootstrap lower bound>0"},
            {"name": "individual_direction_success", "estimate": None, "ci95": None, "bar": ">=0.70 at one common site"},
        ],
    ),
]


def main() -> None:
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    claim = make_claim(record)
    existing = next((item for item in record["claims"] if item["claim_id"] == CLAIM_ID), None)
    if existing is None:
        append_claim_revision(TAG, claim, artifacts=ARTIFACTS)
    else:
        assert existing == claim
        assert all(record["artifacts"].get(key) == value for key, value in ARTIFACTS.items())
    for item in EVENTS:
        current = json.loads(path.read_text())
        if not any(old["event_id"] == item["event_id"] for old in current["evidence_events"]):
            append_evidence_event(TAG, item)
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({"tag": TAG, "claim_id": CLAIM_ID, "status": "specified", "preregistered_events": [item["event_id"] for item in EVENTS], "gpu_used": False}, indent=2))


if __name__ == "__main__":
    main()
