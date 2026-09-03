#!/usr/bin/env python3
"""Supersede R543v1 with delimiter-balanced, still outcome-free v2 rows."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    append_claim_revision, append_evidence_event, circuit_path,
    file_sha256, rebuild_registry_v2, validate_v2,
)

TAG = "task.bracket.pending_opener"
OLD_CLAIM, NEW_CLAIM = "pending_opener_state.v12", "pending_opener_state.v13"
EVENT = "pending_opener_rows.r543.v1.invalid_unbalanced_delimiter_pairs"
SPLIT_ID = "pending_opener_unique_joint_split_r543_v2"
PATHS = {
    "r543_v2_rows": ("basis_aligned/bilinear_quotient/pending_opener_unique_rows_rung543_v2.json", "dataset"),
    "r543_v2_rows_receipt": ("basis_aligned/bilinear_quotient/pending_opener_unique_rows_rung543_v2_receipt.json", "split"),
    "r543_v2_rows_builder": ("basis_aligned/bilinear_quotient/ops/pending_opener_unique_rows_rung543_v2.py", "builder"),
    "r543_v2_rows_test": ("basis_aligned/bilinear_quotient/ops/test_pending_opener_unique_rows_rung543_v2.py", "test"),
    "r543_v2_correction": ("basis_aligned/polynomial_causal/PENDING_OPENER_UNIQUE_FOUR_CLOSER_ROWS_RUNG543_V2_CORRECTION.md", "correction"),
}


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    artifacts = {key: frozen(*value) for key, value in PATHS.items()}
    rows = json.loads((REPO / PATHS["r543_v2_rows"][0]).read_text())
    receipt = json.loads((REPO / PATHS["r543_v2_rows_receipt"][0]).read_text())
    assert rows["outcomes_opened"] == [] and rows["model_loaded"] is False
    assert receipt["unique_prompt_pair_count"] == 1200
    split = {
        "split_plan_id": SPLIT_ID,
        "unit": "content-addressed semantic group shared across five families and balanced across 12 ordered delimiter pairs",
        "partition_artifact_id": "r543_v2_rows_receipt",
        "builder_artifact_id": "r543_v2_rows_builder",
        "seed": 543,
        "groups": {"FIT": 96, "SELECT": 48, "FINAL_TEST": 48, "OOD": 48},
        "leakage_group_keys": [
            "exact token sequence", "exact oriented prompt pair", "prefix pool", "word pool",
            "ordered delimiter pair", "template",
        ],
        "sealed_before_outcomes": True,
        "sealed_at": "2026-09-03T15:25:00Z",
    }
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(item["event_id"] == EVENT for item in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": EVENT,
            "claim_id": OLD_CLAIM,
            "test_type": "null_control",
            "stage": "invalid",
            "verdict": "invalid",
            "failure_kind": "invalid_instrument",
            "family_ids": [
                "direct_four_closer_type_substitution",
                "completed_then_reopened_four_closer_order",
            ],
            "site_id": None,
            "split_plan_id": "pending_opener_unique_joint_split_r543_v1",
            "evaluation_role": "pre_outcome_delimiter_balance_audit",
            "metrics": [
                {"name": "minimum_ordered_delimiter_pair_count", "estimate": 0,
                 "ci95": None, "bar": ">=4 in every split"},
                {"name": "model_outcomes_opened_before_correction", "estimate": 0,
                 "ci95": None, "bar": "0"},
            ],
            "prereg_artifact_id": "r543_unique_rows_preregistration",
            "result_artifact_id": "r543_unique_rows_receipt",
            "input_artifact_ids": ["r543_unique_rows", "r543_unique_rows_builder", "r543_unique_rows_test"],
            "seed": 543,
            "checkpoint_sha256": None,
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["PENDING_OPENER_UNIQUE_FOUR_CLOSER_ROWS_RUNG543_V2_CORRECTION.md"],
            "notes": "V1 was globally unique but did not balance the 12 ordered closer pairs. It was superseded before any model outcome.",
        })
    record = json.loads(path.read_text())
    if not any(item["claim_id"] == NEW_CLAIM for item in record["claims"]):
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 13,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [EVENT],
            "split_plan_ids": [SPLIT_ID],
            "next_missing": (
                "run FIT/SELECT-only balanced four-closer capability and complete-state site ceilings; "
                "then compare ordinary and endpoint-readout-deflated contrastive DAS without opening FINAL_TEST/OOD"
            ),
        })
        for family in claim["counterfactual_families"]:
            family["split_plan_id"] = SPLIT_ID
            family["builder_artifact_id"] = "r543_v2_rows_builder"
        append_claim_revision(TAG, claim, artifacts=artifacts, split_plans=[split])
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "status": "specified",
        "v1_event": EVENT, "v2_rows_sha256": artifacts["r543_v2_rows"]["sha256"],
        "outcomes_opened": [],
    }, indent=2))


if __name__ == "__main__":
    main()
