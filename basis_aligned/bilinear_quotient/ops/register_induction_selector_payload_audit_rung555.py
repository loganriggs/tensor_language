#!/usr/bin/env python3
"""Register the pre-outcome R555 receipt audit for R554."""

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
OLD_CLAIM = "induction_selector_and_payload.v3"
NEW_CLAIM = "induction_selector_and_payload.v4"
EVENT = "induction_selector_payload_capability_audit.r555.preregistered.v1"
SPLIT_ID = "induction_selector_payload_factorial_split_r552_v1"
PATHS = {
    "r555_capability_audit_preregistration": (
        "basis_aligned/polynomial_causal/INDUCTION_SELECTOR_PAYLOAD_CAPABILITY_AUDIT_RUNG555_PREREGISTRATION.md",
        "preregistration"),
    "r555_capability_audit_implementation": (
        "basis_aligned/bilinear_quotient/ops/induction_selector_payload_capability_audit_rung555.py",
        "audit_implementation"),
    "r555_capability_audit_test": (
        "basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_capability_audit_rung555.py", "test"),
}
FAMILIES = [
    "two_valid_sources_selector_swap", "payload_swap_match_preserved",
    "selector_payload_joint_answer_preserved", "match_break_payload_preserved",
    "irrelevant_source_edit", "copy_relation_preserved_nuisance_change",
]


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    append_artifacts(TAG, {key: frozen(*value) for key, value in PATHS.items()})
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(old["event_id"] == EVENT for old in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": EVENT,
            "claim_id": OLD_CLAIM,
            "test_type": "null_control",
            "stage": "preregistered",
            "verdict": "inconclusive",
            "failure_kind": None,
            "family_ids": FAMILIES,
            "site_id": "r554_saved_capability_summaries",
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "pre_outcome_CPU_receipt_and_terminal_decision_audit",
            "metrics": [
                {"name": "execution_and_authority_exact", "estimate": None, "ci95": None,
                 "bar": "checkpoint/input hashes, 864 sequences, 27 forwards, and FIT/SELECT opening exact"},
                {"name": "required_cell_coverage_exact", "estimate": None, "ci95": None,
                 "bar": "all registered factorial, invariance, and necessity summary cells present once"},
                {"name": "terminal_decision_recomputed", "estimate": None, "ci95": None,
                 "bar": "every leaf and prediction flag equals the independently applied frozen inequality"},
            ],
            "prereg_artifact_id": "r555_capability_audit_preregistration",
            "result_artifact_id": None,
            "input_artifact_ids": [
                "r552_factorial_rows", "r552_factorial_rows_receipt", "r553_factorial_rows_audit",
                "r554_capability_preregistration", "r554_capability_implementation", "r554_capability_test",
                "r555_capability_audit_implementation", "r555_capability_audit_test",
            ],
            "seed": None,
            "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["INDUCTION_SELECTOR_PAYLOAD_CAPABILITY_AUDIT_RUNG555_PREREGISTRATION.md"],
            "notes": (
                "Frozen before R554 outcomes. This recomputes decisions from saved group summaries; it does not "
                "claim an independent bootstrap recomputation because R554 does not save raw group margins."
            ),
        })

    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 4,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [EVENT],
            "next_missing": (
                "execute R554 and immediately apply the pre-outcome R555 receipt/decision audit; only after an "
                "audited held capability result preregister separate selector and payload complete-state ceilings"
            ),
        })
        append_claim_revision(TAG, claim)

    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({"tag": TAG, "claim_id": NEW_CLAIM, "event": EVENT,
                      "model_outcomes_opened": False}, indent=2))


if __name__ == "__main__":
    main()
