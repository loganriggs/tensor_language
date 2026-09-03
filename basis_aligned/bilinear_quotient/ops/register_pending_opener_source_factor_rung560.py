#!/usr/bin/env python3
"""Register the held R560 source audit and frozen factor interchange."""

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
OLD_CLAIM = "pending_opener_state.v25"
NEW_CLAIM = "pending_opener_state.v26"
AUDIT_EVENT = "pending_opener_source_position_audit.r560.complete.held.v1"
PLAN_EVENT = "pending_opener_source_factor_interchange.r560.preregistered.v1"
SPLIT_ID = "pending_opener_three_value_fresh_split_r545_v1"
PATHS = {
    "r560_source_factor_preregistration": (
        "basis_aligned/polynomial_causal/PENDING_OPENER_SOURCE_FACTOR_INTERCHANGE_RUNG560_PREREGISTRATION.md",
        "preregistration"),
    "r560_source_position_audit": (
        "basis_aligned/bilinear_quotient/pending_opener_source_positions_rung560_audit.json", "audit"),
    "r560_source_position_audit_implementation": (
        "basis_aligned/bilinear_quotient/ops/pending_opener_source_positions_rung560.py", "audit_implementation"),
    "r560_source_position_audit_test": (
        "basis_aligned/bilinear_quotient/ops/test_pending_opener_source_positions_rung560.py", "test"),
    "r560_source_factor_implementation": (
        "basis_aligned/bilinear_quotient/ops/pending_opener_source_factor_interchange_rung560.py", "implementation"),
    "r560_source_factor_test": (
        "basis_aligned/bilinear_quotient/ops/test_pending_opener_source_factor_interchange_rung560.py", "test"),
}
FAMILIES = [
    "direct_three_value_type_substitution", "completed_then_reopened_three_value_order",
    "pending_type_preserved_surface_rewrite", "pending_type_preserved_distance_extension",
    "pending_type_preserved_nonopener_punctuation",
]


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    audit = json.loads((REPO / PATHS["r560_source_position_audit"][0]).read_text())
    assert audit["all_checks_pass"] is True and audit["row_count"] == 540
    assert audit["inconsistent_proposed_variable_endpoint_labels"] == 108
    assert audit["unequal_length_distance_rows"] == 108
    assert audit["model_loaded"] is False and audit["outcomes_opened"] == []
    append_artifacts(TAG, {key: frozen(*value) for key, value in PATHS.items()})
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(event["event_id"] == AUDIT_EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": AUDIT_EVENT, "claim_id": OLD_CLAIM, "test_type": "null_control",
            "stage": "complete", "verdict": "held", "failure_kind": None,
            "family_ids": FAMILIES,
            "site_id": "r545_semantic_pending_opener_source_positions",
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "outcome_blind_source_position_and_metadata_audit",
            "metrics": [
                {"name": "FIT_SELECT_rows_with_valid_semantic_source", "estimate": 540,
                 "ci95": None, "bar": "540/540"},
                {"name": "unequal_length_distance_rows_supported", "estimate": 108,
                 "ci95": None, "bar": "108/108"},
                {"name": "inconsistent_donor_metadata_labels_found", "estimate": 108,
                 "ci95": None, "bar": "reported and excluded from source authority"},
            ],
            "prereg_artifact_id": "r560_source_factor_preregistration",
            "result_artifact_id": "r560_source_position_audit",
            "input_artifact_ids": ["r545_three_value_rows", "r560_source_position_audit_implementation",
                                   "r560_source_position_audit_test"],
            "seed": None, "checkpoint_sha256": None,
            "supersedes_event_id": None, "replicates_event_id": None,
            "sections": ["PENDING_OPENER_SOURCE_FACTOR_INTERCHANGE_RUNG560_PREREGISTRATION.md"],
            "notes": (
                "The registered correct closer, not the inconsistent proposed_variable_donor label, binds each "
                "endpoint to its final semantic opener position."
            ),
        })
    record = json.loads(path.read_text())
    if not any(event["event_id"] == PLAN_EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": PLAN_EVENT, "claim_id": OLD_CLAIM, "test_type": "composition",
            "stage": "preregistered", "verdict": "inconclusive", "failure_kind": None,
            "family_ids": FAMILIES,
            "site_id": "attention13.head8.semantic_source_score_times_projected_value.final_query",
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "FIT_factor_choice_then_single_SELECT_validation",
            "metrics": [
                {"name": "target_factor_recovery", "estimate": None, "ci95": None,
                 "bar": "every family/direction median>=0.50, bootstrap lower mean>0, positive fraction>=0.75"},
                {"name": "answer_preserving_closer_change", "estimate": None, "ci95": None,
                 "bar": "every family/direction mean absolute<=0.10 and <=0.25 complete-head effect"},
                {"name": "answer_preserving_full_vocab_change", "estimate": None, "ci95": None,
                 "bar": "every family/direction mean RMS <=0.25 complete-head RMS"},
                {"name": "adjacent_wrong_source_recovery", "estimate": None, "ci95": None,
                 "bar": "every target family/direction mean absolute<=0.25"},
                {"name": "score_payload_interaction", "estimate": None, "ci95": None,
                 "bar": "reported; not used to weaken selection bars"},
            ],
            "prereg_artifact_id": "r560_source_factor_preregistration",
            "result_artifact_id": None,
            "input_artifact_ids": [
                "r545_three_value_rows", "r545_three_value_rows_receipt",
                "r546_three_value_confirmation_result", "r548_three_value_confirmation_audit",
                "r560_source_position_audit", "r560_source_factor_implementation", "r560_source_factor_test",
            ],
            "seed": 560,
            "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            "supersedes_event_id": None, "replicates_event_id": None,
            "sections": ["PENDING_OPENER_SOURCE_FACTOR_INTERCHANGE_RUNG560_PREREGISTRATION.md"],
            "notes": "Exact score, projected-value, and joint source terms; no hidden-dimension search.",
        })
    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM, "revision": 26, "status": "specified", "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [AUDIT_EVENT, PLAN_EVENT],
            "next_missing": (
                "execute and independently audit the frozen R560 semantic-source score/payload factor interchange; "
                "keep FINAL_TEST/OOD closed and do not return to a linear-dimension sweep"
            ),
        })
        append_claim_revision(TAG, claim)
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({"tag": TAG, "claim_id": NEW_CLAIM, "event": PLAN_EVENT,
                      "source_rows_audited": 540, "metadata_defects_recorded": 108,
                      "model_outcomes_opened": False}, indent=2))


if __name__ == "__main__":
    main()
