#!/usr/bin/env python3
"""Register the frozen R554 induction capability screen before execution."""

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
OLD_CLAIM = "induction_selector_and_payload.v2"
NEW_CLAIM = "induction_selector_and_payload.v3"
EVENT = "induction_selector_payload_capability.r554.preregistered.v1"
SPLIT_ID = "induction_selector_payload_factorial_split_r552_v1"
PATHS = {
    "r554_capability_preregistration": (
        "basis_aligned/polynomial_causal/INDUCTION_SELECTOR_PAYLOAD_CAPABILITY_RUNG554_PREREGISTRATION.md",
        "preregistration",
    ),
    "r554_capability_implementation": (
        "basis_aligned/bilinear_quotient/ops/induction_selector_payload_capability_rung554.py",
        "implementation",
    ),
    "r554_capability_test": (
        "basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_capability_rung554.py",
        "test",
    ),
}
FAMILIES = [
    "two_valid_sources_selector_swap",
    "payload_swap_match_preserved",
    "selector_payload_joint_answer_preserved",
    "match_break_payload_preserved",
    "irrelevant_source_edit",
    "copy_relation_preserved_nuisance_change",
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
            "test_type": "capability",
            "stage": "preregistered",
            "verdict": "inconclusive",
            "failure_kind": None,
            "family_ids": FAMILIES,
            "site_id": "native_final_token_logits",
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "FIT_and_SELECT_native_factorial_capability_before_site_selection",
            "metrics": [
                {"name": "four_factorial_cell_accuracy", "estimate": None, "ci95": None,
                 "bar": ">=0.75 in every FIT/SELECT x SxP cell; bootstrap lower mean margin >0"},
                {"name": "relation_preserving_endpoint_accuracy", "estimate": None, "ci95": None,
                 "bar": ">=0.75 in every FIT/SELECT x variant x endpoint cell; bootstrap lower mean margin >0"},
                {"name": "selected_match_necessity", "estimate": None, "ci95": None,
                 "bar": ">=0.70 positive rows and bootstrap lower mean margin reduction >0 in both splits"},
                {"name": "selected_vs_irrelevant_source_edit", "estimate": None, "ci95": None,
                 "bar": "bootstrap lower paired mean selective gap >0 in both splits"},
            ],
            "prereg_artifact_id": "r554_capability_preregistration",
            "result_artifact_id": None,
            "input_artifact_ids": [
                "r552_factorial_rows", "r552_factorial_rows_receipt", "r553_factorial_rows_audit",
                "r554_capability_implementation", "r554_capability_test",
            ],
            "seed": 554,
            "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["INDUCTION_SELECTOR_PAYLOAD_CAPABILITY_RUNG554_PREREGISTRATION.md"],
            "notes": (
                "Exactly 864 unique FIT/SELECT sequences and 27 native forwards. No site, subspace, rank, or "
                "regularization selection; FINAL_TEST/OOD stay closed."
            ),
        })

    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 3,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [EVENT],
            "next_missing": (
                "execute and independently audit the frozen R554 FIT/SELECT native-capability screen; only after a "
                "held result preregister separate selector and payload/write complete-state ceilings"
            ),
        })
        append_claim_revision(TAG, claim)

    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG,
        "claim_id": NEW_CLAIM,
        "status": "specified",
        "event": EVENT,
        "planned_forwards": 27,
        "model_outcomes_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
