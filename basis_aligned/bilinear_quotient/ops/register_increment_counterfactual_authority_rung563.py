#!/usr/bin/env python3
"""Register R563 and preserve R562 v1 as a rejected pre-outcome construction."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import append_claim_revision, circuit_path, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

TAG = "task.increment.state"
OLD_CLAIM, NEW_CLAIM = "increment_state.v2", "increment_state.v3"
SPLIT_ID = "increment_counterfactual_split_r563_v1"
PATHS = {
    "r563_rows": ("basis_aligned/bilinear_quotient/increment_counterfactual_authority_rung563.json", "dataset"),
    "r563_receipt": ("basis_aligned/bilinear_quotient/increment_counterfactual_authority_rung563_receipt.json", "split"),
    "r563_builder": ("basis_aligned/bilinear_quotient/ops/increment_counterfactual_authority_rung563.py", "builder"),
    "r563_test": ("basis_aligned/bilinear_quotient/ops/test_increment_counterfactual_authority_rung563.py", "test"),
    "r563_correction": ("basis_aligned/polynomial_causal/INCREMENT_COUNTERFACTUAL_AUTHORITY_RUNG563_CORRECTION.md", "preregistration"),
}


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    artifacts = {key: frozen(*value) for key, value in PATHS.items()}
    rows = json.loads((REPO / PATHS["r563_rows"][0]).read_text())
    receipt = json.loads((REPO / PATHS["r563_receipt"][0]).read_text())
    assert rows["row_count"] == 1120 and rows["outcomes_opened"] == []
    assert rows["family_revealing_prompt_labels"] is False
    assert receipt["unique_prompt_pair_count"] == 1120
    record = json.loads(circuit_path(TAG).read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 3,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "split_plan_ids": [SPLIT_ID],
            "next_missing": (
                "preregister FIT/SELECT native capability on all seven natural-prompt R563 families; require digit "
                "and number-word success, cross-format transfer, middle-edit sensitivity, and stability on both "
                "nonincrement controls before reopening L8H7/L8H3 or MLP8-14 localization"
            ),
        })
        for family in claim["counterfactual_families"]:
            family["builder_artifact_id"] = "r563_builder"
            family["split_plan_id"] = SPLIT_ID
        split = {
            "split_plan_id": SPLIT_ID,
            "unit": "content-addressed semantic group shared across seven natural-prompt families",
            "partition_artifact_id": "r563_receipt",
            "builder_artifact_id": "r563_builder",
            "seed": 563,
            "groups": {"FIT": 64, "SELECT": 32, "FINAL_TEST": 32, "OOD": 32},
            "leakage_group_keys": ["semantic group", "oriented start values", "lexical pool", "prompt lead", "all derived rows"],
            "sealed_before_outcomes": True,
            "sealed_at": "2026-09-03T17:58:00Z",
        }
        append_claim_revision(TAG, claim, artifacts=artifacts, split_plans=[split])
    final = json.loads(circuit_path(TAG).read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "status": "specified",
        "rows_sha256": artifacts["r563_rows"]["sha256"], "groups": 160,
        "rows": 1120, "outcomes_opened": [], "r562_v1_reused": False,
    }, indent=2))


if __name__ == "__main__":
    main()
