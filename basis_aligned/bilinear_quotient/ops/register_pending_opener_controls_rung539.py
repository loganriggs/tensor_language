#!/usr/bin/env python3
"""Register R539 control ceilings before their outcomes are opened."""

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
OLD_CLAIM = "pending_opener_state.v7"
NEW_CLAIM = "pending_opener_state.v8"
EVENT = "pending_opener_control_ceilings.r539.preregistered.v1"
PREREG = "basis_aligned/polynomial_causal/PENDING_OPENER_CONTROL_CEILINGS_RUNG539_PREREGISTRATION.md"
IMPLEMENTATION = "basis_aligned/bilinear_quotient/ops/pending_opener_control_ceilings_rung539.py"


def frozen(path: str, kind: str) -> dict:
    source = REPO / path
    assert source.is_file(), path
    return {"path": path, "sha256": file_sha256(source), "kind": kind, "status": "frozen"}


def main() -> None:
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    artifacts = {
        "r539_control_preregistration": frozen(PREREG, "preregistration"),
        "r539_control_implementation": frozen(IMPLEMENTATION, "implementation"),
    }
    append_artifacts(TAG, artifacts)

    record = json.loads(path.read_text())
    if not any(item["event_id"] == EVENT for item in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": EVENT, "claim_id": OLD_CLAIM,
            "test_type": "null_control", "stage": "preregistered",
            "verdict": "inconclusive", "failure_kind": None,
            "family_ids": [
                "pending_state_preserved_surface_edit",
                "nonopener_punctuation_substitution",
            ],
            "site_id": "residual.block8.entry.final_position",
            "split_plan_id": "pending_opener_joint_split_r537_v1",
            "evaluation_role": "FIT_SELECT_only",
            "metrics": [
                {"name": "mean_absolute_endpoint_ceiling", "estimate": None, "ci95": None,
                 "bar": "group-bootstrap lower bound>0.05 logit in every split and direction"},
                {"name": "full_vocabulary_logit_rms", "estimate": None, "ci95": None,
                 "bar": "mean>0.01 logit in every split and direction"},
            ],
            "prereg_artifact_id": "r539_control_preregistration",
            "result_artifact_id": None,
            "input_artifact_ids": [
                "r537_rows", "r537_controls", "r538_site_result_v2",
                "r538_site_terminal_audit", "r539_control_implementation",
            ],
            "seed": 539,
            "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            "supersedes_event_id": None, "replicates_event_id": None,
            "sections": ["PENDING_OPENER_CONTROL_CEILINGS_RUNG539_PREREGISTRATION.md"],
        })
    record = json.loads(path.read_text())
    if not any(item["claim_id"] == NEW_CLAIM for item in record["claims"]):
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM, "revision": 8, "status": "site_live",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [EVENT],
            "next_missing": (
                "execute the frozen resid8 full-state ceilings for both answer-preserving families; "
                "no DAS optimization until their causal informativeness is known"
            ),
        })
        append_claim_revision(TAG, claim)
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "status": "site_live",
        "event": EVENT, "outcomes_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
