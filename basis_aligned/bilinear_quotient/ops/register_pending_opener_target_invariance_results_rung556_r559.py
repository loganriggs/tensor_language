#!/usr/bin/env python3
"""Register the R556 selective-linear-subspace null and R559 structural audit."""

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
OLD_CLAIM = "pending_opener_state.v24"
NEW_CLAIM = "pending_opener_state.v25"
PLAN = "pending_opener_target_invariance_das.r556.preregistered.v1"
RESULT_EVENT = "pending_opener_target_invariance_das.r556.complete.null.v1"
AUDIT_EVENT = "pending_opener_target_invariance_das_audit.r559.complete.held.v1"
SPLIT_ID = "pending_opener_three_value_fresh_split_r545_v1"
PATHS = {
    "r556_target_invariance_das_result": (
        "basis_aligned/bilinear_quotient/pending_opener_target_invariance_das_rung556_results.json", "result"),
    "r556_target_invariance_das_projectors": (
        "basis_aligned/bilinear_quotient/pending_opener_target_invariance_das_rung556_projectors.pt", "bundle"),
    "r559_target_invariance_audit_spec": (
        "basis_aligned/polynomial_causal/PENDING_OPENER_TARGET_INVARIANCE_DAS_AUDIT_RUNG559.md", "audit_spec"),
    "r559_target_invariance_audit_implementation": (
        "basis_aligned/bilinear_quotient/ops/pending_opener_target_invariance_das_audit_rung559.py", "audit_implementation"),
    "r559_target_invariance_audit_test": (
        "basis_aligned/bilinear_quotient/ops/test_pending_opener_target_invariance_das_audit_rung559.py", "test"),
    "r559_target_invariance_audit_result": (
        "basis_aligned/bilinear_quotient/pending_opener_target_invariance_das_rung559_audit.json", "audit"),
}
FAMILIES = [
    "direct_three_value_type_substitution", "completed_then_reopened_three_value_order",
    "pending_type_preserved_surface_rewrite", "pending_type_preserved_distance_extension",
    "pending_type_preserved_nonopener_punctuation",
]


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    result = json.loads((REPO / PATHS["r556_target_invariance_das_result"][0]).read_text())
    audit = json.loads((REPO / PATHS["r559_target_invariance_audit_result"][0]).read_text())
    assert result["pred_a_exact_instrument"] is True
    assert result["pred_b_stable_selective_projector_exists"] is False
    assert result["strong_null"] is True and result["selected_rank"] is None
    assert audit["strong_null_recomputed"] is True and audit["selected_rank"] is None
    assert audit["result_sha256"] == file_sha256(REPO / PATHS["r556_target_invariance_das_result"][0])
    assert audit["bundle_sha256"] == file_sha256(REPO / PATHS["r556_target_invariance_das_projectors"][0])
    assert result["forbidden_splits_opened"] == []

    append_artifacts(TAG, {key: frozen(*value) for key, value in PATHS.items()})
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(event["event_id"] == RESULT_EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": RESULT_EVENT,
            "claim_id": OLD_CLAIM,
            "test_type": "das_interchange",
            "stage": "complete",
            "verdict": "null",
            "failure_kind": "scientific_null",
            "family_ids": FAMILIES,
            "site_id": "attention13.head8.output.final_position",
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "FIT_joint_target_invariance_optimization_SELECT_capacity_selection",
            "metrics": [
                {"name": "fits_passing_all_target_cells_at_dimensions_2_to_16", "estimate": 12,
                 "ci95": None, "bar": "reported; not sufficient without controls"},
                {"name": "fits_passing_all_target_and_control_cells", "estimate": 0,
                 "ci95": None, "bar": ">=2/3 seeds at one dimension"},
                {"name": "dimensions_passing_random_subspace_bar", "estimate": 4,
                 "ci95": None, "bar": "5/5"},
                {"name": "selected_dimension", "estimate": None, "ci95": None,
                 "bar": "smallest stable selective dimension"},
            ],
            "prereg_artifact_id": "r556_target_invariance_das_preregistration",
            "result_artifact_id": "r556_target_invariance_das_result",
            "input_artifact_ids": [
                "r545_three_value_rows", "r545_three_value_rows_receipt",
                "r546_three_value_confirmation_result", "r548_three_value_confirmation_audit",
                "r556_target_invariance_das_implementation", "r556_target_invariance_das_test",
                "r556_target_invariance_das_projectors",
            ],
            "seed": 556,
            "checkpoint_sha256": result["checkpoint_weights_sha256"],
            "supersedes_event_id": PLAN,
            "replicates_event_id": None,
            "sections": ["PENDING_OPENER_TARGET_INVARIANCE_DAS_RUNG556_PREREGISTRATION.md"],
            "notes": (
                "Target recovery was strong from dimension 2 upward, but every learned projector failed registered "
                "answer-preserving controls. This is a selectivity null, not a claim that the head lacks causal effect."
            ),
        })
    record = json.loads(path.read_text())
    if not any(event["event_id"] == AUDIT_EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": AUDIT_EVENT,
            "claim_id": OLD_CLAIM,
            "test_type": "null_control",
            "stage": "complete",
            "verdict": "held",
            "failure_kind": None,
            "family_ids": FAMILIES,
            "site_id": "r556_saved_row_statistics",
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "post_result_CPU_structural_audit",
            "metrics": [
                {"name": "target_cells_recomputed", "estimate": audit["target_cells_recomputed"],
                 "ci95": None, "bar": "60/60"},
                {"name": "control_cells_recomputed", "estimate": audit["control_cells_recomputed"],
                 "ci95": None, "bar": "90/90"},
                {"name": "terminal_null_recomputed", "estimate": 1.0, "ci95": None, "bar": "exact"},
            ],
            "prereg_artifact_id": "r559_target_invariance_audit_spec",
            "result_artifact_id": "r559_target_invariance_audit_result",
            "input_artifact_ids": [
                "r556_target_invariance_das_result", "r556_target_invariance_das_projectors",
                "r559_target_invariance_audit_implementation", "r559_target_invariance_audit_test",
            ],
            "seed": None,
            "checkpoint_sha256": result["checkpoint_weights_sha256"],
            "supersedes_event_id": None,
            "replicates_event_id": RESULT_EVENT,
            "sections": ["PENDING_OPENER_TARGET_INVARIANCE_DAS_AUDIT_RUNG559.md"],
            "notes": audit["limitation"],
        })

    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 25,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [RESULT_EVENT, AUDIT_EVENT],
            "next_missing": (
                "test a preregistered nonlinear or upstream factorized representation on the same R545 families; "
                "do not extend the L13H8 linear-subspace dimension sweep"
            ),
        })
        append_claim_revision(TAG, claim)
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "verdict": "selective_linear_subspace_null",
        "target_cells_recomputed": audit["target_cells_recomputed"],
        "control_cells_recomputed": audit["control_cells_recomputed"],
        "selected_rank": None,
    }, indent=2))


if __name__ == "__main__":
    main()
