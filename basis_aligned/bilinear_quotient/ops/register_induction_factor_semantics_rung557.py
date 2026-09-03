#!/usr/bin/env python3
"""Register the held R557 selector-versus-payload intervention semantics check."""

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
OLD_CLAIM = "induction_selector_and_payload.v4"
NEW_CLAIM = "induction_selector_and_payload.v5"
EVENT = "induction_factor_intervention_semantics.r557.held.v1"
SPLIT_ID = "induction_selector_payload_factorial_split_r552_v1"
PATHS = {
    "r557_factor_semantics_preregistration": (
        "basis_aligned/polynomial_causal/INDUCTION_FACTOR_INTERVENTION_SEMANTICS_RUNG557_PREREGISTRATION.md",
        "preregistration",
    ),
    "r557_factor_semantics_implementation": (
        "basis_aligned/bilinear_quotient/ops/induction_factor_intervention_semantics_rung557.py",
        "implementation",
    ),
    "r557_factor_semantics_test": (
        "basis_aligned/bilinear_quotient/ops/test_induction_factor_intervention_semantics_rung557.py",
        "test",
    ),
    "r557_factor_semantics_result": (
        "basis_aligned/bilinear_quotient/induction_factor_intervention_semantics_rung557_results.json",
        "result",
    ),
}
FAMILIES = [
    "two_valid_sources_selector_swap",
    "payload_swap_match_preserved",
    "selector_payload_joint_answer_preserved",
    "match_break_payload_preserved",
    "irrelevant_source_edit",
]


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    result = json.loads((REPO / PATHS["r557_factor_semantics_result"][0]).read_text())
    assert result["all_checks_pass"] is True
    assert result["model_loaded"] is False
    assert result["model_forwards"] == 0 and result["model_backwards"] == 0
    assert result["outcomes_opened"] == []
    assert result["factorial_condition_checks"] == 720
    assert result["direction_checks"] == {
        "two_valid_sources_selector_swap": 720,
        "payload_swap_match_preserved": 720,
        "selector_payload_joint_answer_preserved": 720,
        "match_break_payload_preserved": 180,
        "irrelevant_source_edit": 180,
    }

    append_artifacts(TAG, {key: frozen(*value) for key, value in PATHS.items()})
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(old["event_id"] == EVENT for old in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": EVENT,
            "claim_id": OLD_CLAIM,
            "test_type": "null_control",
            "stage": "complete",
            "verdict": "held",
            "failure_kind": None,
            "family_ids": FAMILIES,
            "site_id": "r552_discrete_equality_score_and_payload_factors",
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "outcome_blind_intervention_semantics_before_model_site_search",
            "metrics": [
                {"name": "factorial_fetch_exact", "estimate": 720, "ci95": None, "bar": "720/720"},
                {"name": "selector_score_transplants_exact", "estimate": 720, "ci95": None, "bar": "720/720"},
                {"name": "payload_value_transplants_exact", "estimate": 720, "ci95": None, "bar": "720/720"},
                {"name": "joint_answer_preserving_transplants_exact", "estimate": 720, "ci95": None, "bar": "720/720"},
                {"name": "necessity_and_irrelevant_controls_exact", "estimate": 360, "ci95": None, "bar": "360/360"},
            ],
            "prereg_artifact_id": "r557_factor_semantics_preregistration",
            "result_artifact_id": "r557_factor_semantics_result",
            "input_artifact_ids": [
                "r552_factorial_rows", "r552_factorial_rows_receipt", "r553_factorial_rows_audit",
                "r557_factor_semantics_implementation", "r557_factor_semantics_test",
            ],
            "seed": 557,
            "checkpoint_sha256": None,
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["INDUCTION_FACTOR_INTERVENTION_SEMANTICS_RUNG557_PREREGISTRATION.md"],
            "notes": (
                "This validates the meaning of separate selector-score and payload-value interventions on all "
                "frozen rows. It is not evidence that any model site implements those factors."
            ),
        })

    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 5,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [EVENT],
            "next_missing": (
                "execute and audit R554; if native factorial capability holds, preregister a model-facing "
                "selector-score versus payload-value interchange screen below the attention-head boundary"
            ),
        })
        append_claim_revision(TAG, claim)

    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG,
        "claim_id": NEW_CLAIM,
        "event": EVENT,
        "checks": 3240,
        "model_forwards": 0,
        "status": "held intervention semantics; model implementation not yet tested",
    }, indent=2))


if __name__ == "__main__":
    main()
