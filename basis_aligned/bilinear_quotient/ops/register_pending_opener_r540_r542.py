#!/usr/bin/env python3
"""Register R540's selectivity null and R542's split-integrity correction."""

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
OLD_CLAIM = "pending_opener_state.v10"
NEW_CLAIM = "pending_opener_state.v11"
PREREG_EVENT = "pending_opener_cross_family_das.r540.preregistered.v1"
COMPLETE_EVENT = "pending_opener_cross_family_das.r540.complete.v1"
AUDIT_EVENT = "pending_opener_split_integrity.r542.invalid_statistical_unit.v1"

R540_RESULT = "basis_aligned/bilinear_quotient/pending_opener_cross_family_das_rung540_results.json"
R540_BUNDLE = "basis_aligned/bilinear_quotient/pending_opener_cross_family_das_rung540_bundle.pt"
R542_RESULT = "basis_aligned/bilinear_quotient/pending_opener_split_integrity_rung542_results.json"
R542_IMPL = "basis_aligned/bilinear_quotient/ops/pending_opener_split_integrity_rung542.py"
R542_TEST = "basis_aligned/bilinear_quotient/ops/test_pending_opener_split_integrity_rung542.py"


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    r540 = json.loads((REPO / R540_RESULT).read_text())
    r542 = json.loads((REPO / R542_RESULT).read_text())
    assert r540["pred_a_exact_instrument"] is True
    assert r540["pred_b_two_way_cross_family_transfer"] is False
    assert r540["pred_d_controls_selective_and_random_below_bar"] is False
    assert r540["strong_null"] is True and r540["selected_rank"] is None
    assert r540["forbidden_splits_opened"] == []
    assert r542["pred_a_exact_cross_split_isolation"] is True
    assert r542["pred_b_no_within_split_pseudoreplication"] is False
    assert r542["pred_c_r538_site_decision_survives_unique_prompt_rescore"] is True
    assert r542["pred_d_r539_control_liveness_survives_unique_prompt_rescore"] is True
    assert r542["pred_e_r540_null_survives_unique_prompt_rescore"] is True

    artifacts = {
        "r540_das_result": frozen(R540_RESULT, "result"),
        "r540_das_bundle": frozen(R540_BUNDLE, "sufficient_statistics"),
        "r542_split_integrity_result": frozen(R542_RESULT, "audit"),
        "r542_split_integrity_implementation": frozen(R542_IMPL, "audit_implementation"),
        "r542_split_integrity_test": frozen(R542_TEST, "test"),
    }
    append_artifacts(TAG, artifacts)
    path = circuit_path(TAG)
    record = json.loads(path.read_text())

    if not any(item["event_id"] == COMPLETE_EVENT for item in record["evidence_events"]):
        audit_summary = r542["r540_unique_prompt_rescore"]["summary"]
        fits = r542["r540_unique_prompt_rescore"]["fits"]
        target_cells = [
            fits[str(rank)][source][str(seed)]["targets"][family][direction]
            for rank in (1, 2, 4, 8, 16)
            for source in ("direct", "structural", "joint")
            for seed in (0, 1, 2)
            for family in ("opener_type_substitution", "closed_then_reopened_type")
            for direction in ("base_to_donor", "donor_to_base")
        ]
        control_cells = [
            fits[str(rank)][source][str(seed)]["controls"][family][direction]
            for rank in (1, 2, 4, 8, 16)
            for source in ("direct", "structural", "joint")
            for seed in (0, 1, 2)
            for family in ("pending_state_preserved_surface_edit", "nonopener_punctuation_substitution")
            for direction in ("base_to_donor", "donor_to_base")
        ]
        append_evidence_event(TAG, {
            "event_id": COMPLETE_EVENT,
            "claim_id": "pending_opener_state.v9",
            "test_type": "cross_family_transfer",
            "stage": "complete",
            "verdict": "null",
            "failure_kind": "scientific_null",
            "family_ids": [
                "opener_type_substitution", "closed_then_reopened_type",
                "pending_state_preserved_surface_edit", "nonopener_punctuation_substitution",
            ],
            "site_id": "residual.block8.entry.final_position",
            "split_plan_id": "pending_opener_joint_split_r537_v1",
            "evaluation_role": "FIT_optimize_SELECT_choose_unique_prompt_rescore",
            "metrics": [
                {
                    "name": "two_way_cross_family_normalized_recovery",
                    "estimate": min(item["median"] for item in target_cells),
                    "ci95": [min(item["bootstrap95_lower_mean"] for item in target_cells), None],
                    "bar": "median>=0.50, bootstrap lower>0, positive fraction>=0.75 in every target cell",
                },
                {
                    "name": "answer_preserving_control_leakage",
                    "estimate": {
                        "cells_passing": audit_summary["control_cells_passed"],
                        "cells_total": audit_summary["control_cells_total"],
                        "maximum_mean_absolute_logit": max(item["mean_absolute"] for item in control_cells),
                        "maximum_fraction_of_full": max(item["fraction_of_full"] for item in control_cells),
                    },
                    "ci95": None,
                    "bar": "mean absolute<=0.10 logit and <=0.25 of full-state effect in every control cell",
                },
                {
                    "name": "operational_response_equivalence",
                    "estimate": None,
                    "ci95": None,
                    "bar": "response cosine>=0.90 and RMS difference<=0.15 across training sources",
                },
            ],
            "prereg_artifact_id": "r540_das_preregistration",
            "result_artifact_id": "r540_das_result",
            "input_artifact_ids": [
                "r537_rows", "r537_controls", "r538_site_result_v2", "r539_control_result",
                "r540_das_implementation", "r540_das_bundle", "r542_split_integrity_result",
            ],
            "seed": 540,
            "checkpoint_sha256": r540["checkpoint_weights_sha256"],
            "supersedes_event_id": PREREG_EVENT,
            "replicates_event_id": None,
            "sections": ["PENDING_OPENER_CROSS_FAMILY_DAS_RUNG540_PREREGISTRATION.md"],
            "notes": (
                "All 180 deduplicated SELECT target cells passed, but only 3/180 answer-preserving "
                "control cells passed. No rank was eligible. Exact prompt deduplication in R542 leaves "
                "this decision unchanged; FINAL_TEST and OOD were not opened."
            ),
        })

    record = json.loads(path.read_text())
    if not any(item["event_id"] == AUDIT_EVENT for item in record["evidence_events"]):
        unique_fractions = [
            cell["effective_fraction"]
            for family in r542["main_dataset"]["cells"].values()
            for cell in family.values()
        ]
        append_evidence_event(TAG, {
            "event_id": AUDIT_EVENT,
            "claim_id": "pending_opener_state.v10",
            "test_type": "seed_stability",
            "stage": "complete",
            "verdict": "invalid",
            "failure_kind": "invalid_instrument",
            "family_ids": [
                "opener_type_substitution", "closed_then_reopened_type",
                "pending_state_preserved_surface_edit", "nonopener_punctuation_substitution",
            ],
            "site_id": "residual.block8.entry.final_position",
            "split_plan_id": "pending_opener_joint_split_r537_v1",
            "evaluation_role": "post_result_exact_stimulus_audit",
            "metrics": [
                {"name": "exact_cross_split_prompt_overlap", "estimate": 0, "ci95": None, "bar": "0"},
                {"name": "minimum_unique_prompt_fraction", "estimate": min(unique_fractions),
                 "ci95": None, "bar": "1.0 for nominal independent rows"},
                {"name": "deduplicated_decisions_preserved", "estimate": 3 / 3,
                 "ci95": None, "bar": "R538, R539, and R540 decisions all unchanged"},
            ],
            "prereg_artifact_id": None,
            "result_artifact_id": "r542_split_integrity_result",
            "input_artifact_ids": [
                "r537_rows", "r537_controls", "r538_site_result_v2", "r539_control_result",
                "r540_das_result", "r542_split_integrity_implementation", "r542_split_integrity_test",
            ],
            "seed": 542,
            "checkpoint_sha256": None,
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": [],
            "notes": (
                "No exact token sequence crosses FIT/SELECT/FINAL/OOD, but deterministic cycles repeat "
                "identical prompt pairs within each split. Report unique-prompt n, not group_id n. "
                "The saved R538--R540 decisions survive deduplication, but unopened splits are retired "
                "and future rows require content-addressed groups."
            ),
        })

    record = json.loads(path.read_text())
    if not any(item["claim_id"] == NEW_CLAIM for item in record["claims"]):
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 11,
            "status": "site_live",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [COMPLETE_EVENT, AUDIT_EVENT],
            "next_missing": (
                "replace the cyclic R537 stimuli with unique content-addressed counterfactual groups; "
                "then fit a contrastive shared-plus-private intervention that is rewarded for both target "
                "families and explicitly penalized on both live answer-preserving families"
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
        "r540_event": COMPLETE_EVENT,
        "r542_event": AUDIT_EVENT,
        "selected_rank": None,
        "final_or_ood_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
