#!/usr/bin/env python3
"""Register the audited R549 downstream-atlas null and R551 readout guard."""

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
OLD_CLAIM = "pending_opener_state.v22"
NEW_CLAIM = "pending_opener_state.v23"
R549_PREREG = "pending_opener_downstream_response_atlas.r549.preregistered.v1"
R551_PREREG = "pending_opener_downstream_readout_guard.r551.preregistered.v1"
R549_EVENT = "pending_opener_downstream_response_atlas.r549.complete.null.v1"
R551_EVENT = "pending_opener_downstream_readout_guard.r551.complete.null.v1"
SPLIT_ID = "pending_opener_three_value_fresh_split_r545_v1"
PATHS = {
    "r549_downstream_response_atlas_result": (
        "basis_aligned/bilinear_quotient/pending_opener_downstream_response_atlas_rung549_results.json", "result"),
    "r549_downstream_response_atlas_bundle": (
        "basis_aligned/bilinear_quotient/pending_opener_downstream_response_atlas_rung549_vectors.pt",
        "sufficient_statistics"),
    "r550_downstream_response_atlas_audit": (
        "basis_aligned/bilinear_quotient/pending_opener_downstream_response_atlas_rung550_audit.json", "audit"),
    "r550_downstream_response_atlas_audit_implementation": (
        "basis_aligned/bilinear_quotient/ops/pending_opener_downstream_response_atlas_rung550_audit.py",
        "audit_implementation"),
    "r550_downstream_response_atlas_audit_test": (
        "basis_aligned/bilinear_quotient/ops/test_pending_opener_downstream_response_atlas_rung550_audit.py", "test"),
    "r551_downstream_readout_guard_result": (
        "basis_aligned/bilinear_quotient/pending_opener_downstream_readout_independence_rung551_results.json",
        "result"),
}
ALL_FAMILIES = [
    "direct_three_value_type_substitution",
    "completed_then_reopened_three_value_order",
    "pending_type_preserved_surface_rewrite",
    "pending_type_preserved_distance_extension",
    "pending_type_preserved_nonopener_punctuation",
]
TARGET_FAMILIES = ALL_FAMILIES[:2]


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    atlas = json.loads((REPO / PATHS["r549_downstream_response_atlas_result"][0]).read_text())
    audit = json.loads((REPO / PATHS["r550_downstream_response_atlas_audit"][0]).read_text())
    guard = json.loads((REPO / PATHS["r551_downstream_readout_guard_result"][0]).read_text())
    assert atlas["pred_a_exact_instrument"] is True
    assert atlas["pred_b_fit_selects_candidate"] is True
    assert atlas["pred_c_selected_candidate_validates"] is False
    assert atlas["strong_null"] is True and atlas["selected_candidate"] == "mlp15_write"
    assert atlas["model_forwards"] == 204 and atlas["forbidden_splits_opened"] == []
    assert audit["status"] == "terminal_audit_complete"
    assert audit["independent_fit_and_select_recomputation"] is True
    assert audit["selection_depends_only_on_fit"] is True
    assert audit["selected_candidate_validates"] is False
    assert audit["result_sha256"] == file_sha256(REPO / PATHS["r549_downstream_response_atlas_result"][0])
    assert audit["bundle_sha256"] == file_sha256(REPO / PATHS["r549_downstream_response_atlas_bundle"][0])
    assert guard["pred_a_result_and_bundle_bound"] is True
    assert guard["pred_b_r549_pairwise_readout_diagnostic_reproduced"] is True
    assert guard["pred_c_distinct_downstream_target"] is False and guard["strong_null"] is True
    assert guard["forbidden_splits_opened"] == []

    append_artifacts(TAG, {key: frozen(*value) for key, value in PATHS.items()})
    path = circuit_path(TAG)
    selected = atlas["metrics"][atlas["selected_candidate"]]
    events = [
        {
            "event_id": R549_EVENT,
            "claim_id": "pending_opener_state.v20",
            "test_type": "cross_family_transfer",
            "stage": "complete",
            "verdict": "null",
            "failure_kind": "scientific_null",
            "family_ids": ALL_FAMILIES,
            "site_id": "attention13.head8.output.final_position_to_41_later_component_writes",
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "FIT_selection_SELECT_validation_downstream_response_screen",
            "metrics": [
                {"name": "fit_leave_family_out_transition_accuracy", "estimate": [
                    selected["fit"]["direct_templates_classify_order_accuracy"],
                    selected["fit"]["order_templates_classify_direct_accuracy"],
                ], "ci95": None, "bar": ">=0.50 in both directions for FIT eligibility"},
                {"name": "fit_answer_preserving_template_cosine",
                 "estimate": selected["fit"]["control_median_max_absolute_template_cosine"],
                 "ci95": None, "bar": "median maximum absolute cosine <=0.40"},
                {"name": "select_leave_family_out_transition_accuracy", "estimate": [
                    selected["select"]["direct_templates_classify_order_accuracy"],
                    selected["select"]["order_templates_classify_direct_accuracy"],
                ], "ci95": None, "bar": ">=0.50 in both directions for the FIT-selected candidate"},
                {"name": "select_answer_preserving_template_cosine",
                 "estimate": selected["select"]["control_median_max_absolute_template_cosine"],
                 "ci95": None, "bar": "median maximum absolute cosine <=0.35 for the FIT-selected candidate"},
                {"name": "select_patch_to_natural_response_norm_ratio",
                 "estimate": selected["select"]["median_patch_to_natural_response_norm_ratio"],
                 "ci95": None, "bar": "median >=0.05 for the FIT-selected candidate"},
            ],
            "prereg_artifact_id": "r549_downstream_response_atlas_preregistration",
            "result_artifact_id": "r549_downstream_response_atlas_result",
            "input_artifact_ids": [
                "r545_three_value_rows", "r545_three_value_rows_receipt",
                "r546_three_value_confirmation_result", "r548_three_value_confirmation_audit",
                "r549_downstream_response_atlas_implementation", "r549_downstream_response_atlas_test",
                "r549_downstream_response_atlas_bundle", "r550_downstream_response_atlas_audit",
                "r550_downstream_response_atlas_audit_implementation", "r550_downstream_response_atlas_audit_test",
            ],
            "seed": None,
            "checkpoint_sha256": atlas["checkpoint_weights_sha256"],
            "supersedes_event_id": R549_PREREG,
            "replicates_event_id": None,
            "sections": ["PENDING_OPENER_DOWNSTREAM_RESPONSE_ATLAS_RUNG549_PREREGISTRATION.md"],
            "notes": (
                "FIT selected MLP15 write from four eligible candidates. Both cross-construction accuracies were "
                "1.0 on SELECT and response size was substantial, but answer-preserving alignment was 0.36095, "
                "above the frozen 0.35 maximum. R550 independently reproduced selection and every metric."
            ),
        },
        {
            "event_id": R551_EVENT,
            "claim_id": "pending_opener_state.v21",
            "test_type": "null_control",
            "stage": "complete",
            "verdict": "null",
            "failure_kind": "scientific_null",
            "family_ids": TARGET_FAMILIES,
            "site_id": "r549_selected_later_write_vs_direct_closer_readout_span",
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "FIT_only_pre_outcome_interpretation_guard",
            "metrics": [
                {"name": "median_transition_template_norm_fraction_in_closer_readout_span",
                 "estimate": guard["median_readout_span_fraction"], "ci95": None,
                 "bar": "<=0.50 after R549 SELECT validation"},
                {"name": "r549_pairwise_readout_cosine_recomputation",
                 "estimate": guard["recomputed_median_max_pairwise_readout_cosine"], "ci95": None,
                 "bar": "exactly reproduce the saved pairwise diagnostic within 2e-7"},
            ],
            "prereg_artifact_id": "r551_downstream_readout_guard_preregistration",
            "result_artifact_id": "r551_downstream_readout_guard_result",
            "input_artifact_ids": [
                "r549_downstream_response_atlas_preregistration", "r549_downstream_response_atlas_implementation",
                "r549_downstream_response_atlas_result", "r549_downstream_response_atlas_bundle",
                "r551_downstream_readout_guard_implementation", "r551_downstream_readout_guard_test",
            ],
            "seed": None,
            "checkpoint_sha256": guard["checkpoint_weights_sha256"],
            "supersedes_event_id": R551_PREREG,
            "replicates_event_id": None,
            "sections": ["PENDING_OPENER_DOWNSTREAM_READOUT_INDEPENDENCE_RUNG551_PREREGISTRATION.md"],
            "notes": (
                "The MLP15 templates themselves are mostly outside the closer-readout span (median 0.0609), but "
                "the guard requires an R549 SELECT-valid candidate. That prerequisite failed, so no independent "
                "second DAS target is licensed."
            ),
        },
    ]
    for event in events:
        record = json.loads(path.read_text())
        if not any(old["event_id"] == event["event_id"] for old in record["evidence_events"]):
            append_evidence_event(TAG, event)

    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 23,
            "status": "site_live",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [R549_EVENT, R551_EVENT],
            "next_missing": (
                "preregister an L13H8 endpoint-plus-invariance interchange that rewards both answer-changing "
                "families and directly penalizes all three live answer-preserving families; do not use R549 as an "
                "independent downstream target and keep FINAL_TEST/OOD closed"
            ),
        })
        append_claim_revision(TAG, claim)

    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG,
        "claim_id": NEW_CLAIM,
        "status": "site_live",
        "r549_event": R549_EVENT,
        "r551_event": R551_EVENT,
        "selected_candidate": atlas["selected_candidate"],
        "selected_candidate_validates": False,
        "final_or_ood_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
