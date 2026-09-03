#!/usr/bin/env python3
"""Register R539's live invariance/control ceilings."""

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
OLD_CLAIM = "pending_opener_state.v8"
NEW_CLAIM = "pending_opener_state.v9"
PREREG_EVENT = "pending_opener_control_ceilings.r539.preregistered.v1"
COMPLETE_EVENT = "pending_opener_control_ceilings.r539.complete.v1"
RESULT = "basis_aligned/bilinear_quotient/pending_opener_control_ceilings_rung539_results.json"
AUDIT = "basis_aligned/bilinear_quotient/pending_opener_control_ceilings_rung539_terminal_audit.json"
AUDIT_IMPL = "basis_aligned/bilinear_quotient/ops/pending_opener_control_ceilings_rung539_terminal_audit.py"


def frozen(path: str, kind: str) -> dict:
    source = REPO / path
    return {"path": path, "sha256": file_sha256(source), "kind": kind, "status": "frozen"}


def main() -> None:
    result = json.loads((REPO / RESULT).read_text())
    audit = json.loads((REPO / AUDIT).read_text())
    assert result["pred_a_exact_instrument"] is True
    assert result["pred_b_surface_invariance_causally_testable"] is True
    assert result["pred_c_nonopener_control_causally_testable"] is True
    assert audit["all_checks_pass"] is True
    artifacts = {
        "r539_control_result": frozen(RESULT, "result"),
        "r539_control_terminal_audit": frozen(AUDIT, "audit"),
        "r539_control_terminal_audit_implementation": frozen(AUDIT_IMPL, "audit_implementation"),
    }
    append_artifacts(TAG, artifacts)
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(item["event_id"] == COMPLETE_EVENT for item in record["evidence_events"]):
        cells = [
            result["reports"][split][family][direction]
            for split in ("FIT", "SELECT")
            for family in ("pending_state_preserved_surface_edit", "nonopener_punctuation_substitution")
            for direction in ("base_to_donor", "donor_to_base")
        ]
        append_evidence_event(TAG, {
            "event_id": COMPLETE_EVENT, "claim_id": "pending_opener_state.v7",
            "test_type": "null_control", "stage": "complete", "verdict": "held",
            "failure_kind": None,
            "family_ids": [
                "pending_state_preserved_surface_edit", "nonopener_punctuation_substitution",
            ],
            "site_id": "residual.block8.entry.final_position",
            "split_plan_id": "pending_opener_joint_split_r537_v1",
            "evaluation_role": "FIT_SELECT_only",
            "metrics": [
                {"name": "mean_absolute_endpoint_ceiling",
                 "estimate": min(item["mean_absolute_endpoint_change"] for item in cells),
                 "ci95": [min(item["bootstrap95_lower_mean_absolute"] for item in cells), None],
                 "bar": "group-bootstrap lower bound>0.05 logit in every split and direction"},
                {"name": "full_vocabulary_logit_rms",
                 "estimate": min(item["mean_full_vocabulary_logit_rms"] for item in cells),
                 "ci95": None, "bar": "mean>0.01 logit in every split and direction"},
            ],
            "prereg_artifact_id": "r539_control_preregistration",
            "result_artifact_id": "r539_control_result",
            "input_artifact_ids": [
                "r537_rows", "r537_controls", "r538_site_result_v2",
                "r538_site_terminal_audit", "r539_control_implementation",
            ],
            "seed": 539,
            "checkpoint_sha256": result["checkpoint_weights_sha256"],
            "supersedes_event_id": PREREG_EVENT, "replicates_event_id": None,
            "sections": ["PENDING_OPENER_CONTROL_CEILINGS_RUNG539_PREREGISTRATION.md"],
        })
    record = json.loads(path.read_text())
    if not any(item["claim_id"] == NEW_CLAIM for item in record["claims"]):
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM, "revision": 9, "status": "site_live",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": [event for event in previous["evidence_event_ids"]
                                   if event != PREREG_EVENT] + [COMPLETE_EVENT],
            "next_missing": (
                "freeze multi-seed shared-versus-family-specific DAS at resid8, using two-way "
                "cross-family transfer and the live R539 controls; do not require coordinate overlap"
            ),
        })
        append_claim_revision(TAG, claim)
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "status": "site_live",
        "event": COMPLETE_EVENT,
        "minimum_endpoint_lower": audit["minimum_endpoint_bootstrap_lower"],
        "minimum_logit_rms": audit["minimum_full_vocabulary_logit_rms"],
    }, indent=2))


if __name__ == "__main__":
    main()
