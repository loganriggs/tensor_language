#!/usr/bin/env python3
"""Register the audited R560 source-factor null and R561 audit."""

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
OLD_CLAIM = "pending_opener_state.v27"
NEW_CLAIM = "pending_opener_state.v28"
PLAN = "pending_opener_source_factor_interchange.r560.v2.preregistered.v1"
AUDIT_PLAN = "pending_opener_source_factor_audit.r561.preregistered.v1"
RESULT_EVENT = "pending_opener_source_factor_interchange.r560.v2.complete.null.v1"
AUDIT_EVENT = "pending_opener_source_factor_audit.r561.complete.held.v1"
SPLIT_ID = "pending_opener_three_value_fresh_split_r545_v1"
PATHS = {
    "r560_source_factor_result": (
        "basis_aligned/bilinear_quotient/pending_opener_source_factor_interchange_rung560_results.json", "result"),
    "r561_source_factor_audit_result": (
        "basis_aligned/bilinear_quotient/pending_opener_source_factor_interchange_rung561_audit.json", "audit"),
}
FAMILIES = [
    "direct_three_value_type_substitution", "completed_then_reopened_three_value_order",
    "pending_type_preserved_surface_rewrite", "pending_type_preserved_distance_extension",
    "pending_type_preserved_nonopener_punctuation",
]


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    result = json.loads((REPO / PATHS["r560_source_factor_result"][0]).read_text())
    audit = json.loads((REPO / PATHS["r561_source_factor_audit_result"][0]).read_text())
    assert result["pred_a_exact_instrument"] is True
    assert result["pred_b_fit_selective_source_factor_exists"] is False
    assert result["pred_c_selected_source_factor_holds"] is False
    assert result["fit_choice"]["selected_arm"] is None
    assert result["evaluated_splits"] == ["FIT"] and result["forbidden_splits_opened"] == []
    assert audit["result_sha256"] == file_sha256(REPO / PATHS["r560_source_factor_result"][0])
    assert audit["decision"] == "source_factor_null" and audit["fit_cells_recomputed"] == 42
    payload = result["fit_reports"]["payload"]
    target_recoveries = [cell["mean"] for family in payload["targets"].values() for cell in family.values()]
    punctuation = payload["controls"]["pending_type_preserved_nonopener_punctuation"]
    punctuation_rms = [cell["fraction_of_complete_head_full_vocabulary_rms"] for cell in punctuation.values()]
    wrong = [cell["mean_absolute_recovery"] for family in payload["wrong_source_controls"].values()
             for cell in family.values()]
    assert min(target_recoveries) > .96 and max(target_recoveries) > 1.26
    assert min(punctuation_rms) > .25 and min(wrong) > .25

    append_artifacts(TAG, {key: frozen(*value) for key, value in PATHS.items()})
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(event["event_id"] == RESULT_EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": RESULT_EVENT, "claim_id": OLD_CLAIM, "test_type": "composition",
            "stage": "complete", "verdict": "null", "failure_kind": "scientific_null",
            "family_ids": FAMILIES,
            "site_id": "attention13.head8.semantic_source_score_times_projected_value.final_query",
            "split_plan_id": SPLIT_ID, "evaluation_role": "FIT_factor_choice_then_single_SELECT_validation",
            "metrics": [
                {"name": "payload_target_mean_recovery_range", "estimate": [min(target_recoveries), max(target_recoveries)],
                 "ci95": None, "bar": "all four target cells pass"},
                {"name": "score_target_mean_recovery_range", "estimate": [
                    min(cell["mean"] for family in result["fit_reports"]["score"]["targets"].values()
                        for cell in family.values()),
                    max(cell["mean"] for family in result["fit_reports"]["score"]["targets"].values()
                        for cell in family.values())], "ci95": None, "bar": "target bars; failed"},
                {"name": "payload_nonopener_punctuation_RMS_ratio_range", "estimate": punctuation_rms,
                 "ci95": None, "bar": "<=0.25 in both directions; failed"},
                {"name": "payload_adjacent_wrong_source_absolute_recovery_range", "estimate": [min(wrong), max(wrong)],
                 "ci95": None, "bar": "<=0.25 in every target direction; failed"},
            ],
            "prereg_artifact_id": "r560_source_factor_preregistration",
            "result_artifact_id": "r560_source_factor_result",
            "input_artifact_ids": ["r560_source_position_audit", "r560_v2_correction", "r560_v2_implementation",
                                   "r560_v2_test"],
            "seed": 560, "checkpoint_sha256": result["checkpoint_weights_sha256"],
            "supersedes_event_id": PLAN, "replicates_event_id": None,
            "sections": ["PENDING_OPENER_SOURCE_FACTOR_INTERCHANGE_RUNG560_PREREGISTRATION.md"],
            "notes": (
                "Payload carries nearly all target effect and score carries almost none, but the exact single-source "
                "payload term failed punctuation-RMS and adjacent-wrong-source selectivity; SELECT remained closed."
            ),
        })
    record = json.loads(path.read_text())
    if not any(event["event_id"] == AUDIT_EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": AUDIT_EVENT, "claim_id": OLD_CLAIM, "test_type": "null_control",
            "stage": "complete", "verdict": "held", "failure_kind": None,
            "family_ids": FAMILIES, "site_id": "r560_saved_source_factor_statistics",
            "split_plan_id": SPLIT_ID, "evaluation_role": "pre_outcome_CPU_terminal_audit",
            "metrics": [
                {"name": "FIT_decision_cells_recomputed", "estimate": 42, "ci95": None, "bar": "42/42"},
                {"name": "interaction_rows_recomputed", "estimate": 288, "ci95": None, "bar": "288/288"},
                {"name": "terminal_null_recomputed", "estimate": 1.0, "ci95": None, "bar": "exact"},
            ],
            "prereg_artifact_id": "r561_source_factor_audit_preregistration",
            "result_artifact_id": "r561_source_factor_audit_result",
            "input_artifact_ids": ["r560_source_factor_result", "r561_source_factor_audit_implementation",
                                   "r561_source_factor_audit_test"],
            "seed": None, "checkpoint_sha256": result["checkpoint_weights_sha256"],
            "supersedes_event_id": AUDIT_PLAN, "replicates_event_id": RESULT_EVENT,
            "sections": ["PENDING_OPENER_SOURCE_FACTOR_AUDIT_RUNG561_PREREGISTRATION.md"],
            "notes": "Independent pre-outcome CPU audit exactly reproduced the R560 decision.",
        })
    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({"claim_id": NEW_CLAIM, "revision": 28, "status": "specified", "supersedes": OLD_CLAIM,
                      "evidence_event_ids": list(previous["evidence_event_ids"]) + [RESULT_EVENT, AUDIT_EVENT],
                      "next_missing": (
                          "build fresh rows and test an exact semantic source-region payload decomposition with "
                          "same-pending-state matched donors; retain punctuation RMS as a required control and do not "
                          "return to a linear-dimension sweep"
                      )})
        append_claim_revision(TAG, claim)
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({"tag": TAG, "claim_id": NEW_CLAIM, "verdict": "single_source_factor_null",
                      "payload_target_recovery_range": [min(target_recoveries), max(target_recoveries)],
                      "punctuation_RMS_ratios": punctuation_rms, "SELECT_opened": False}, indent=2))


if __name__ == "__main__":
    main()
