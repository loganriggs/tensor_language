#!/usr/bin/env python3
"""Register R546's independently audited fresh capability and complete-state confirmation."""

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
OLD_CLAIM, NEW_CLAIM = "pending_opener_state.v19", "pending_opener_state.v20"
PREREG_EVENT = "pending_opener_three_value_confirmation.r546.preregistered.v1"
CAPABILITY_EVENT = "pending_opener_three_value_confirmation.r546.fresh_capability.held.v1"
SITE_EVENT = "pending_opener_three_value_confirmation.r546.l13h8_site.held.v1"
PATHS = {
    "r546_three_value_confirmation_result": (
        "basis_aligned/bilinear_quotient/pending_opener_three_value_confirmation_rung546_results.json", "result"),
    "r548_three_value_confirmation_audit": (
        "basis_aligned/bilinear_quotient/pending_opener_three_value_confirmation_rung548_audit.json", "audit"),
    "r548_three_value_confirmation_audit_implementation": (
        "basis_aligned/bilinear_quotient/ops/pending_opener_three_value_confirmation_rung548_audit.py",
        "audit_implementation"),
    "r548_three_value_confirmation_audit_test": (
        "basis_aligned/bilinear_quotient/ops/test_pending_opener_three_value_confirmation_rung548_audit.py", "test"),
}


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    result = json.loads((REPO / PATHS["r546_three_value_confirmation_result"][0]).read_text())
    audit = json.loads((REPO / PATHS["r548_three_value_confirmation_audit"][0]).read_text())
    assert result["model_forwards"] == 204 and result["model_backwards"] == 0
    assert result["evaluated_splits"] == ["FIT", "SELECT"] and result["forbidden_splits_opened"] == []
    assert result["all_gates_pass"] is True
    assert audit["all_gates_held"] is True
    assert audit["independent_summary_recomputation_exact"] is True
    assert audit["complete_row_identity_and_cell_counts"] is True

    append_artifacts(TAG, {key: frozen(*value) for key, value in PATHS.items()})
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    families = [family["family_id"] for family in next(
        claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)["counterfactual_families"]]
    common = {
        "claim_id": OLD_CLAIM,
        "stage": "complete",
        "verdict": "held",
        "failure_kind": None,
        "family_ids": families,
        "site_id": "attention13.head8.output.final_position",
        "split_plan_id": "pending_opener_three_value_fresh_split_r545_v1",
        "evaluation_role": "fresh_FIT_SELECT_confirmation_terminal_audit",
        "prereg_artifact_id": "r546_three_value_confirmation_preregistration",
        "result_artifact_id": "r546_three_value_confirmation_result",
        "input_artifact_ids": [
            "r545_three_value_rows", "r545_three_value_rows_receipt",
            "r546_three_value_confirmation_implementation", "r546_three_value_confirmation_test",
            "r548_three_value_confirmation_audit", "r548_three_value_confirmation_audit_implementation",
            "r548_three_value_confirmation_audit_test",
        ],
        "seed": 546,
        "checkpoint_sha256": result["checkpoint_weights_sha256"],
        "replicates_event_id": None,
        "sections": ["PENDING_OPENER_THREE_VALUE_CONFIRMATION_RUNG546_PREREGISTRATION.md"],
    }
    events = [
        {
            **common,
            "event_id": CAPABILITY_EVENT,
            "test_type": "capability",
            "metrics": [
                {"name": "native_correct_fraction_minimum_across_24_target_pair_cells",
                 "estimate": min(
                     min(cell.values())
                     for cell in audit["capability_by_ordered_pair"].values()
                 ), "ci95": None, "bar": ">=0.75 for base and donor in every ordered-pair cell"},
                {"name": "all_three_answer_preserving_families_natively_capable",
                 "estimate": True, "ci95": None, "bar": ">=0.75 correct on both sides in FIT and SELECT"},
            ],
            "supersedes_event_id": None,
            "notes": (
                "Fresh parenthesis/square/quote capability is perfect in all 24 ordered-pair target cells. "
                "This confirms the three-value domain selected after the independent four-value curly-brace null."
            ),
        },
        {
            **common,
            "event_id": SITE_EVENT,
            "test_type": "full_swap_ceiling",
            "metrics": [
                {"name": "minimum_target_bootstrap95_lower_mean_logit_change",
                 "estimate": audit["site_gate_minima"]["target_bootstrap_lower_mean"],
                 "ci95": None, "bar": ">0 in every family/direction/split, with >=70% positive rows"},
                {"name": "minimum_target_positive_fraction",
                 "estimate": audit["site_gate_minima"]["target_positive_fraction"],
                 "ci95": None, "bar": ">=0.70 in every family/direction/split"},
                {"name": "minimum_control_bootstrap95_lower_absolute_logit_change",
                 "estimate": audit["site_gate_minima"]["control_bootstrap_lower_abs_change"],
                 "ci95": None, "bar": ">0.03 in every control/direction/split"},
                {"name": "minimum_control_full_vocabulary_logit_rms",
                 "estimate": audit["site_gate_minima"]["control_full_vocabulary_logit_rms"],
                 "ci95": None, "bar": ">0.01 in every control/direction/split"},
            ],
            "supersedes_event_id": PREREG_EVENT,
            "notes": (
                "The complete L13H8 state is a confirmed causal site on fresh rows: every target cell transfers "
                "and all three answer-preserving families have enough full-state effect to detect later leakage. "
                "This does not yet identify a selective subspace."
            ),
        },
    ]
    for item in events:
        record = json.loads(path.read_text())
        if not any(old["event_id"] == item["event_id"] for old in record["evidence_events"]):
            append_evidence_event(TAG, item)

    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 20,
            "status": "site_live",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [CAPABILITY_EVENT, SITE_EVENT],
            "next_missing": (
                "freeze a FIT/SELECT-only downstream-response atlas for the two answer-changing and three "
                "answer-preserving families, then preregister a multi-output selective interchange at L13H8; "
                "FINAL_TEST/OOD remain unopened"
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
        "capability_event": CAPABILITY_EVENT,
        "site_event": SITE_EVENT,
        "final_or_ood_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
