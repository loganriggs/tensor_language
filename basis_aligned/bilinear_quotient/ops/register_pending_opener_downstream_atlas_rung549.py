#!/usr/bin/env python3
"""Register the R549 downstream-response atlas before any R549 model call."""

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
OLD_CLAIM, NEW_CLAIM = "pending_opener_state.v20", "pending_opener_state.v21"
EVENT = "pending_opener_downstream_response_atlas.r549.preregistered.v1"
PATHS = {
    "r549_downstream_response_atlas_preregistration": (
        "basis_aligned/polynomial_causal/PENDING_OPENER_DOWNSTREAM_RESPONSE_ATLAS_RUNG549_PREREGISTRATION.md",
        "preregistration"),
    "r549_downstream_response_atlas_implementation": (
        "basis_aligned/bilinear_quotient/ops/pending_opener_downstream_response_atlas_rung549.py",
        "implementation"),
    "r549_downstream_response_atlas_test": (
        "basis_aligned/bilinear_quotient/ops/test_pending_opener_downstream_response_atlas_rung549.py", "test"),
}


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    append_artifacts(TAG, {key: frozen(*value) for key, value in PATHS.items()})
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    families = [family["family_id"] for family in next(
        claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)["counterfactual_families"]]
    if not any(old["event_id"] == EVENT for old in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": EVENT,
            "claim_id": OLD_CLAIM,
            "test_type": "cross_family_transfer",
            "stage": "preregistered",
            "verdict": "inconclusive",
            "failure_kind": None,
            "family_ids": families,
            "site_id": "attention13.head8.output.final_position_to_41_later_component_writes",
            "split_plan_id": "pending_opener_three_value_fresh_split_r545_v1",
            "evaluation_role": "FIT_selection_SELECT_validation_downstream_response_screen",
            "metrics": [
                {"name": "fit_leave_family_out_transition_accuracy", "estimate": None, "ci95": None,
                 "bar": ">=0.50 in both directions for FIT eligibility"},
                {"name": "fit_answer_preserving_template_cosine", "estimate": None, "ci95": None,
                 "bar": "median maximum absolute cosine <=0.40"},
                {"name": "select_leave_family_out_transition_accuracy", "estimate": None, "ci95": None,
                 "bar": ">=0.50 in both directions for the FIT-selected candidate"},
                {"name": "select_answer_preserving_template_cosine", "estimate": None, "ci95": None,
                 "bar": "median maximum absolute cosine <=0.35 for the FIT-selected candidate"},
                {"name": "select_patch_to_natural_response_norm_ratio", "estimate": None, "ci95": None,
                 "bar": "median >=0.05 for the FIT-selected candidate"},
            ],
            "prereg_artifact_id": "r549_downstream_response_atlas_preregistration",
            "result_artifact_id": None,
            "input_artifact_ids": [
                "r545_three_value_rows", "r545_three_value_rows_receipt",
                "r546_three_value_confirmation_result", "r548_three_value_confirmation_audit",
                "r549_downstream_response_atlas_implementation", "r549_downstream_response_atlas_test",
            ],
            "seed": None,
            "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["PENDING_OPENER_DOWNSTREAM_RESPONSE_ATLAS_RUNG549_PREREGISTRATION.md"],
            "notes": (
                "FIT alone selects at most one later component. SELECT validates cross-construction transition "
                "identity and answer-preserving separation. Readout alignment is diagnostic and cannot select."
            ),
        })

    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 21,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [EVENT],
            "next_missing": (
                "execute and independently audit the frozen 204-forward R549 FIT-selection/SELECT-validation "
                "downstream-response atlas; FINAL_TEST/OOD remain unopened"
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
