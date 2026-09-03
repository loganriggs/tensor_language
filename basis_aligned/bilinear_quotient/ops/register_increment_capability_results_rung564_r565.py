#!/usr/bin/env python3
"""Register R564's terminal FIT null and R565's independent CPU audit."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import append_artifacts, append_claim_revision, append_evidence_event, circuit_path, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

TAG = "task.increment.state"
PREREG_EVENT = "increment_native_capability.r564.preregistered.v1"
RESULT_EVENT = "increment_native_capability.r564.result.v1"
AUDIT_EVENT = "increment_native_capability_audit.r565.held.v1"
PATHS = {
    "r564_result": ("basis_aligned/bilinear_quotient/increment_native_capability_rung564_results.json", "result"),
    "r565_audit": ("basis_aligned/bilinear_quotient/increment_native_capability_rung565_audit.json", "audit"),
    "r565_script": ("basis_aligned/bilinear_quotient/ops/audit_increment_native_capability_rung565.py", "implementation"),
    "r565_test": ("basis_aligned/bilinear_quotient/ops/test_audit_increment_native_capability_rung565.py", "test"),
}
FAMILIES = [
    "digit_coherent_shift", "word_coherent_shift", "cross_format_coherent_shift",
    "incoherent_middle_number_edit", "operation_preserved_surface_edit",
    "repeated_number_numeric_control", "step_two_numeric_control",
]


def main() -> None:
    artifacts = {
        key: {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}
        for key, (path, kind) in PATHS.items()
    }
    append_artifacts(TAG, artifacts)
    record = json.loads(circuit_path(TAG).read_text())
    if not any(event["event_id"] == RESULT_EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": RESULT_EVENT, "test_type": "capability", "stage": "complete", "verdict": "null",
            "failure_kind": "scientific_null", "family_ids": FAMILIES, "site_id": None,
            "evaluation_role": "FIT_only_stopped_by_frozen_gate",
            "metrics": [
                {"name": "numeric_candidate_accuracy_and_margin", "estimate": 0.0, "ci95": None,
                 "bar": ">=0.75 correct and bootstrap lower mean margin >0 in every endpoint cell"},
                {"name": "middle_number_necessity", "estimate": 0.015625, "ci95": None,
                 "bar": ">=0.65 positive drops and bootstrap lower mean drop >0"},
                {"name": "split_opening", "estimate": 0.0, "ci95": None,
                 "bar": "SELECT only after all FIT cells pass; FINAL_TEST/OOD closed"},
            ],
            "result_artifact_id": "r564_result", "prereg_artifact_id": "r564_prereg",
            "input_artifact_ids": ["r563_rows", "r563_receipt", "r563_correction", "r564_script"],
            "split_plan_id": "increment_counterfactual_split_r563_v1", "seed": 564,
            "checkpoint_sha256": json.loads((REPO / PATHS["r564_result"][0]).read_text())["checkpoint_weights_sha256"],
            "supersedes_event_id": PREREG_EVENT, "replicates_event_id": None,
            "sections": ["polynomial_causal/INCREMENT_NATIVE_CAPABILITY_RUNG564_PREREGISTRATION.md"],
            "claim_id": "increment_state.v3",
        })
    record = json.loads(circuit_path(TAG).read_text())
    if not any(event["event_id"] == AUDIT_EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": AUDIT_EVENT, "test_type": "null_control", "stage": "complete", "verdict": "held",
            "failure_kind": None, "family_ids": FAMILIES, "site_id": None,
            "evaluation_role": "post_result_structural_audit",
            "metrics": [
                {"name": "row_statistics_recomputed", "estimate": 896, "ci95": None, "bar": "exactly 896"},
                {"name": "endpoint_cells_recomputed", "estimate": 12, "ci95": None, "bar": "exactly 12"},
                {"name": "terminal_null_reproduced", "estimate": 1.0, "ci95": None, "bar": "true"},
            ],
            "result_artifact_id": "r565_audit", "prereg_artifact_id": None,
            "input_artifact_ids": ["r563_rows", "r564_result", "r565_script", "r565_test"],
            "split_plan_id": "increment_counterfactual_split_r563_v1", "seed": 565,
            "checkpoint_sha256": None, "supersedes_event_id": None, "replicates_event_id": None,
            "sections": ["polynomial_causal/explanations/explanation_2026-09-03_1815.md"],
            "claim_id": "increment_state.v3",
        })
    record = json.loads(circuit_path(TAG).read_text())
    if not any(claim["claim_id"] == "increment_state.v4" for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == "increment_state.v3")
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": "increment_state.v4", "revision": 4, "status": "proposed", "supersedes": "increment_state.v3",
            "evidence_event_ids": previous["evidence_event_ids"] + [RESULT_EVENT, AUDIT_EVENT],
            "next_missing": (
                "use an explicitly exploratory development pool to identify natural prompt forms that elicit +1, "
                "number-word +1, copy, and +2; then freeze new group-disjoint rows before any component localization. "
                "Do not apply the legacy L8H7/L8H3/MLP8-14 story to R563"
            ),
        })
        for family in claim["counterfactual_families"]:
            family["status"] = "validated" if family["family_id"] == "repeated_number_numeric_control" else "failed"
        append_claim_revision(TAG, claim)
    final = json.loads(circuit_path(TAG).read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({"tag": TAG, "claim_id": "increment_state.v4", "r564": "null", "r565": "held"}, indent=2))


if __name__ == "__main__":
    main()
