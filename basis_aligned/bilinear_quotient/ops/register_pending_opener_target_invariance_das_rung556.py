#!/usr/bin/env python3
"""Register R556 before any optimization or model call."""

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
OLD_CLAIM = "pending_opener_state.v23"
NEW_CLAIM = "pending_opener_state.v24"
EVENT = "pending_opener_target_invariance_das.r556.preregistered.v1"
SPLIT_ID = "pending_opener_three_value_fresh_split_r545_v1"
PATHS = {
    "r556_target_invariance_das_preregistration": (
        "basis_aligned/polynomial_causal/PENDING_OPENER_TARGET_INVARIANCE_DAS_RUNG556_PREREGISTRATION.md",
        "preregistration"),
    "r556_target_invariance_das_implementation": (
        "basis_aligned/bilinear_quotient/ops/pending_opener_target_invariance_das_rung556.py", "implementation"),
    "r556_target_invariance_das_test": (
        "basis_aligned/bilinear_quotient/ops/test_pending_opener_target_invariance_das_rung556.py", "test"),
}
FAMILIES = [
    "direct_three_value_type_substitution", "completed_then_reopened_three_value_order",
    "pending_type_preserved_surface_rewrite", "pending_type_preserved_distance_extension",
    "pending_type_preserved_nonopener_punctuation",
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
            "test_type": "das_interchange",
            "stage": "preregistered",
            "verdict": "inconclusive",
            "failure_kind": None,
            "family_ids": FAMILIES,
            "site_id": "attention13.head8.output.final_position",
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "FIT_joint_target_invariance_optimization_SELECT_capacity_selection",
            "metrics": [
                {"name": "target_normalized_recovery", "estimate": None, "ci95": None,
                 "bar": "median>=0.50, bootstrap lower mean>0, and positive fraction>=0.75 in every SELECT target cell"},
                {"name": "control_closer_margin_change", "estimate": None, "ci95": None,
                 "bar": "mean absolute<=0.10 logit and <=0.25 of complete-head effect in every SELECT control cell"},
                {"name": "control_full_vocabulary_change", "estimate": None, "ci95": None,
                 "bar": "mean logit RMS <=0.25 of complete-head RMS in every SELECT control cell"},
                {"name": "seed_stability", "estimate": None, "ci95": None,
                 "bar": ">=2/3 seeds pass every target and control cell at the selected smallest capacity"},
                {"name": "random_subspace_target_recovery", "estimate": None, "ci95": None,
                 "bar": "five dimension-matched controls average <0.10 at every tested capacity"},
            ],
            "prereg_artifact_id": "r556_target_invariance_das_preregistration",
            "result_artifact_id": None,
            "input_artifact_ids": [
                "r545_three_value_rows", "r545_three_value_rows_receipt",
                "r546_three_value_confirmation_result", "r548_three_value_confirmation_audit",
                "r556_target_invariance_das_implementation", "r556_target_invariance_das_test",
            ],
            "seed": 556,
            "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["PENDING_OPENER_TARGET_INVARIANCE_DAS_RUNG556_PREREGISTRATION.md"],
            "notes": (
                "Unlike R540, all three answer-preserving families enter the FIT objective through normalized "
                "full-vocabulary invariance loss. Capacity is selected by causal SELECT bars, not reconstruction."
            ),
        })

    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 24,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [EVENT],
            "next_missing": (
                "execute and independently audit the frozen R556 FIT-trained target-plus-invariance interchange; "
                "keep FINAL_TEST/OOD closed and do not extend capacity after a null"
            ),
        })
        append_claim_revision(TAG, claim)

    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({"tag": TAG, "claim_id": NEW_CLAIM, "event": EVENT,
                      "fits": 15, "model_outcomes_opened": False}, indent=2))


if __name__ == "__main__":
    main()
