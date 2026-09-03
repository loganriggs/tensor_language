#!/usr/bin/env python3
"""Register the audited R554 capability null and held R555 audit."""

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

TAG = "task.induction.selector_payload"
OLD_CLAIM = "induction_selector_and_payload.v5"
NEW_CLAIM = "induction_selector_and_payload.v6"
R554_PLAN = "induction_selector_payload_capability.r554.preregistered.v1"
R555_PLAN = "induction_selector_payload_capability_audit.r555.preregistered.v1"
R554_EVENT = "induction_selector_payload_capability.r554.complete.null.v1"
R555_EVENT = "induction_selector_payload_capability_audit.r555.complete.held.v1"
SPLIT_ID = "induction_selector_payload_factorial_split_r552_v1"
PATHS = {
    "r554_capability_result": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_capability_rung554_results.json", "result"),
    "r555_capability_audit_result": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_capability_rung555_audit.json", "audit"),
}
FAMILIES = [
    "two_valid_sources_selector_swap", "payload_swap_match_preserved",
    "selector_payload_joint_answer_preserved", "match_break_payload_preserved",
    "irrelevant_source_edit", "copy_relation_preserved_nuisance_change",
]


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    result = json.loads((REPO / PATHS["r554_capability_result"][0]).read_text())
    audit = json.loads((REPO / PATHS["r555_capability_audit_result"][0]).read_text())
    failed = result["relation_preserving_controls"]["SELECT"]["irrelevant_source_edit"]["s1p0"]["donor"]
    assert result["pred_0_exact_instrument"] is True
    assert result["pred_a_four_cell_capability"] is True
    assert result["pred_b_relation_preserving_controls"] is False
    assert result["pred_c_selected_match_necessity_and_selectivity"] is True
    assert result["all_gates_pass"] is False
    assert failed["n_groups"] == 9 and failed["correct_fraction"] == 2 / 3
    assert failed["bootstrap95_lower_mean_margin"] <= 0
    assert audit["result_sha256"] == file_sha256(REPO / PATHS["r554_capability_result"][0])
    assert audit["terminal_decision_recomputed"] is True and audit["all_gates_pass"] is False
    assert result["forbidden_splits_opened"] == [] and audit["forbidden_splits_opened"] == []

    append_artifacts(TAG, {key: frozen(*value) for key, value in PATHS.items()})
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(event["event_id"] == R554_EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": R554_EVENT,
            "claim_id": OLD_CLAIM,
            "test_type": "capability",
            "stage": "complete",
            "verdict": "null",
            "failure_kind": "scientific_null",
            "family_ids": FAMILIES,
            "site_id": "native_final_token_logits",
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "FIT_and_SELECT_native_factorial_capability_before_site_selection",
            "metrics": [
                {"name": "four_factorial_cell_capability", "estimate": 1.0, "ci95": None,
                 "bar": "all 8 FIT/SELECT x SxP cells pass"},
                {"name": "selected_match_necessity_and_selectivity", "estimate": 1.0, "ci95": None,
                 "bar": "both split decisions pass"},
                {"name": "failed_SELECT_s1p0_irrelevant_donor_correct_fraction", "estimate": 2 / 3,
                 "ci95": None, "bar": ">=0.75"},
                {"name": "failed_SELECT_s1p0_irrelevant_donor_lower_mean_margin",
                 "estimate": failed["bootstrap95_lower_mean_margin"], "ci95": None, "bar": ">0"},
            ],
            "prereg_artifact_id": "r554_capability_preregistration",
            "result_artifact_id": "r554_capability_result",
            "input_artifact_ids": [
                "r552_factorial_rows", "r552_factorial_rows_receipt", "r553_factorial_rows_audit",
                "r554_capability_implementation", "r554_capability_test",
            ],
            "seed": 554,
            "checkpoint_sha256": result["checkpoint_weights_sha256"],
            "supersedes_event_id": R554_PLAN,
            "replicates_event_id": None,
            "sections": ["INDUCTION_SELECTOR_PAYLOAD_CAPABILITY_RUNG554_PREREGISTRATION.md"],
            "notes": "One of the preregistered SELECT invariance cells failed; no model site search is licensed.",
        })
    record = json.loads(path.read_text())
    if not any(event["event_id"] == R555_EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": R555_EVENT,
            "claim_id": OLD_CLAIM,
            "test_type": "null_control",
            "stage": "complete",
            "verdict": "held",
            "failure_kind": None,
            "family_ids": FAMILIES,
            "site_id": "r554_saved_capability_summaries",
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "CPU_receipt_and_terminal_decision_audit",
            "metrics": [
                {"name": "receipt_and_cell_coverage_exact", "estimate": 1.0, "ci95": None, "bar": "exact"},
                {"name": "terminal_null_recomputed", "estimate": 1.0, "ci95": None, "bar": "exact"},
            ],
            "prereg_artifact_id": "r555_capability_audit_preregistration",
            "result_artifact_id": "r555_capability_audit_result",
            "input_artifact_ids": ["r554_capability_result", "r555_capability_audit_implementation",
                                   "r555_capability_audit_test"],
            "seed": None,
            "checkpoint_sha256": result["checkpoint_weights_sha256"],
            "supersedes_event_id": R555_PLAN,
            "replicates_event_id": None,
            "sections": ["INDUCTION_SELECTOR_PAYLOAD_CAPABILITY_AUDIT_RUNG555_PREREGISTRATION.md"],
            "notes": audit["limitation"],
        })

    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 6,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [R554_EVENT, R555_EVENT],
            "next_missing": (
                "replace the brittle synthetic irrelevant-source construction or move to a natural-copy dataset, "
                "then freeze fresh FIT/SELECT capability bars before any factor-level model-site search; do not run R558 on R552"
            ),
        })
        append_claim_revision(TAG, claim)
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "capability_verdict": "null",
        "failed_cell_correct_fraction": failed["correct_fraction"],
        "failed_cell_lower_mean_margin": failed["bootstrap95_lower_mean_margin"],
        "r558_authorized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
