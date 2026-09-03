#!/usr/bin/env python3
"""Register R544's four-value capability null and the fresh three-value next hypothesis."""

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
OLD_CLAIM = "pending_opener_state.v14"
REJECTED_CLAIM = "pending_opener_state.v15"
NEXT_CLAIM = "pending_opener_state.v16"
PREREG_EVENT = "pending_opener_four_closer_site_gate.r544.preregistered.v1"
RESULT_EVENT = "pending_opener_four_closer_site_gate.r544.native_curly_null.v1"
RESULT = "basis_aligned/bilinear_quotient/pending_opener_four_closer_site_gate_rung544_results.json"
AUDIT = "basis_aligned/bilinear_quotient/pending_opener_four_closer_site_gate_rung544_audit.json"
AUDIT_IMPL = "basis_aligned/bilinear_quotient/ops/pending_opener_four_closer_site_gate_rung544_audit.py"


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    result = json.loads((REPO / RESULT).read_text())
    audit = json.loads((REPO / AUDIT).read_text())
    assert result["pred_a_exact_instrument"] is True
    assert result["pred_b_four_closer_capability"] is False
    assert result["pred_c_common_target_and_control_live_site"] is False
    assert result["passing_sites_in_frozen_order"] == ["attn13h8"]
    assert result["selected_site"] is None and result["strong_null"] is True
    assert result["forbidden_splits_opened"] == []
    assert audit["failed_native_values"] == ["}"]
    assert audit["attn13h8_target_and_control_full_state_gate_held"] is True

    append_artifacts(TAG, {
        "r544_site_gate_result": frozen(RESULT, "result"),
        "r544_site_gate_terminal_audit": frozen(AUDIT, "audit"),
        "r544_site_gate_terminal_audit_implementation": frozen(AUDIT_IMPL, "audit_implementation"),
    })
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(event["event_id"] == RESULT_EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": RESULT_EVENT,
            "claim_id": OLD_CLAIM,
            "test_type": "capability",
            "stage": "complete",
            "verdict": "null",
            "failure_kind": "scientific_null",
            "family_ids": [
                "direct_four_closer_type_substitution",
                "completed_then_reopened_four_closer_order",
                "pending_type_preserved_surface_paraphrase",
                "pending_type_preserved_distance_shift",
                "pending_type_preserved_nonopener_punctuation",
            ],
            "site_id": "attention13.head8.output.final_position",
            "split_plan_id": "pending_opener_unique_joint_split_r543_v2",
            "evaluation_role": "FIT_SELECT_only_terminal_audit",
            "metrics": [
                {"name": "four_closer_native_capability", "estimate": False, "ci95": None,
                 "bar": ">=75% native correctness for base and donor in every ordered pair"},
                {"name": "failed_native_closer_values", "estimate": ["}"], "ci95": None,
                 "bar": "no value-specific capability failure"},
                {"name": "attn13h8_complete_state_gate", "estimate": True, "ci95": None,
                 "bar": "both target families and all three live controls pass on FIT and SELECT"},
            ],
            "prereg_artifact_id": "r544_site_gate_preregistration",
            "result_artifact_id": "r544_site_gate_result",
            "input_artifact_ids": [
                "r543_v2_rows", "r543_v2_rows_receipt", "r544_site_gate_implementation",
                "r544_site_gate_test", "r544_site_gate_terminal_audit",
                "r544_site_gate_terminal_audit_implementation",
            ],
            "seed": 544,
            "checkpoint_sha256": result["checkpoint_weights_sha256"],
            "supersedes_event_id": PREREG_EVENT,
            "replicates_event_id": None,
            "sections": ["PENDING_OPENER_FOUR_CLOSER_SITE_GATE_RUNG544_PREREGISTRATION.md"],
            "notes": (
                "The four-value domain fails because every native cell involving the curly closer fails, while "
                "all six ordered pairs among parenthesis, square bracket, and quote are perfect on FIT/SELECT. "
                "The latter is post-outcome subgroup evidence only. attn13h8 passes every target and control "
                "complete-state gate, but no projector fit is licensed and FINAL_TEST/OOD remain unopened."
            ),
        })

    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == REJECTED_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        rejected = copy.deepcopy(previous)
        rejected.update({
            "claim_id": REJECTED_CLAIM,
            "revision": 15,
            "status": "rejected",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [RESULT_EVENT],
            "next_missing": (
                "four-value extension rejected at native capability; do not fit a four-value projector or open "
                "its FINAL_TEST/OOD rows"
            ),
        })
        rejected["causal_variable"]["id"] = "pending_opener_state_four_value_extension"
        rejected["causal_variable"]["domain"] = "parenthesis, square, curly, or quote pending state"
        append_claim_revision(TAG, rejected)

    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEXT_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == REJECTED_CLAIM)
        proposed = copy.deepcopy(previous)
        proposed.update({
            "claim_id": NEXT_CLAIM,
            "revision": 16,
            "status": "proposed",
            "supersedes": REJECTED_CLAIM,
            "next_missing": (
                "build fresh content-addressed parenthesis/square/quote groups with new templates and word pools; "
                "preregister native capability and attn13h8 full-state confirmation before any subspace fit"
            ),
        })
        proposed["causal_variable"]["id"] = "pending_opener_state_three_value_candidate"
        proposed["causal_variable"]["domain"] = (
            "candidate parenthesis, square-bracket, or quote pending state; selected after R544 and not confirmed"
        )
        proposed["alternative_explanations"].append(
            "post-outcome exclusion of curly braces; requires entirely fresh confirmation rows"
        )
        for family in proposed["counterfactual_families"]:
            family["status"] = "proposed"
            family["builder_artifact_id"] = None
            family["split_plan_id"] = None
        proposed["split_plan_ids"] = []
        append_claim_revision(TAG, proposed)

    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG,
        "rejected_claim": REJECTED_CLAIM,
        "next_claim": NEXT_CLAIM,
        "next_status": "proposed",
        "failed_value": "}",
        "projector_fit_licensed": False,
        "final_or_ood_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
