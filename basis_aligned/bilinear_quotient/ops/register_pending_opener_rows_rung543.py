#!/usr/bin/env python3
"""Register R543's unique four-closer rows as the active bracket authority."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    append_claim_revision, circuit_path, file_sha256, rebuild_registry_v2, validate_v2,
)

TAG = "task.bracket.pending_opener"
OLD_CLAIM = "pending_opener_state.v11"
NEW_CLAIM = "pending_opener_state.v12"
ROWS = "basis_aligned/bilinear_quotient/pending_opener_unique_rows_rung543.json"
RECEIPT = "basis_aligned/bilinear_quotient/pending_opener_unique_rows_rung543_receipt.json"
BUILDER = "basis_aligned/bilinear_quotient/ops/pending_opener_unique_rows_rung543.py"
TEST = "basis_aligned/bilinear_quotient/ops/test_pending_opener_unique_rows_rung543.py"
PREREG = "basis_aligned/polynomial_causal/PENDING_OPENER_UNIQUE_FOUR_CLOSER_ROWS_RUNG543_PREREGISTRATION.md"
SPLIT_ID = "pending_opener_unique_joint_split_r543_v1"


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def family(family_id: str, role: str, changes: list[str], holds: list[str], controls: list[str]) -> dict:
    return {
        "family_id": family_id,
        "role": role,
        "changes": changes,
        "holds_fixed": holds,
        "builder_artifact_id": "r543_unique_rows_builder",
        "control_ids": controls,
        "split_plan_id": SPLIT_ID,
        "status": "frozen",
    }


def main() -> None:
    rows = json.loads((REPO / ROWS).read_text())
    receipt = json.loads((REPO / RECEIPT).read_text())
    assert rows["status"] == "rows_frozen_outcomes_unopened"
    assert rows["outcomes_opened"] == []
    assert receipt["unique_prompt_pair_count"] == receipt["row_count"] == 1200
    assert receipt["unique_token_sequence_count"] == 2400
    artifacts = {
        "r543_unique_rows": frozen(ROWS, "dataset"),
        "r543_unique_rows_receipt": frozen(RECEIPT, "split"),
        "r543_unique_rows_builder": frozen(BUILDER, "builder"),
        "r543_unique_rows_test": frozen(TEST, "test"),
        "r543_unique_rows_preregistration": frozen(PREREG, "preregistration"),
    }
    split = {
        "split_plan_id": SPLIT_ID,
        "unit": "content-addressed semantic group shared across all five counterfactual families",
        "partition_artifact_id": "r543_unique_rows_receipt",
        "builder_artifact_id": "r543_unique_rows_builder",
        "seed": 543,
        "groups": {"FIT": 96, "SELECT": 48, "FINAL_TEST": 48, "OOD": 48},
        "leakage_group_keys": [
            "exact token sequence", "exact oriented prompt pair", "prefix pool", "word pool",
            "delimiter pair", "template",
        ],
        "sealed_before_outcomes": True,
        "sealed_at": "2026-09-03T15:22:00Z",
    }
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(item["claim_id"] == NEW_CLAIM for item in record["claims"]):
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 12,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "counterfactual_families": [
                family(
                    "direct_four_closer_type_substitution", "interchange",
                    ["pending delimiter type and correct next closer", "replace exactly one opener token among parenthesis, square, curly, and quote"],
                    ["all ordinary words", "sequence length", "opener position", "body text"],
                    ["answer-preserving comma/colon edit", "wrong closer", "endpoint-readout span"],
                ),
                family(
                    "completed_then_reopened_four_closer_order", "interchange",
                    ["pending delimiter type and correct next closer", "complete one type and open another, then reverse their roles"],
                    ["ordinary-word multiset", "sequence length where tokenization permits", "final lexical suffix"],
                    ["same-order paraphrase", "wrong closer", "endpoint-readout span"],
                ),
                family(
                    "pending_type_preserved_surface_paraphrase", "invariance",
                    ["surface scaffold and lexical order", "rewrite the content while retaining one pending delimiter type"],
                    ["pending delimiter type", "correct closer", "semantic group"],
                    ["held-out lexical pool", "complete-state ceiling", "random subspace"],
                ),
                family(
                    "pending_type_preserved_distance_shift", "invariance",
                    ["opener-to-final distance", "insert neutral material after the same opener"],
                    ["pending delimiter type", "correct closer", "content words"],
                    ["short versus long", "complete-state ceiling", "random subspace"],
                ),
                family(
                    "pending_type_preserved_nonopener_punctuation", "invariance",
                    ["one comma token to one colon token before the opener"],
                    ["pending delimiter type", "correct closer", "length", "all tokens after the edit"],
                    ["wrong closer", "complete-state ceiling", "random subspace"],
                ),
            ],
            "split_plan_ids": [SPLIT_ID],
            "next_missing": (
                "run FIT/SELECT-only four-closer capability and complete-state site ceilings; "
                "then compare ordinary and endpoint-readout-deflated contrastive DAS without opening FINAL_TEST/OOD"
            ),
        })
        for site in claim["candidate_sites"]:
            site["ceiling_event_ids"] = []
        append_claim_revision(TAG, claim, artifacts=artifacts, split_plans=[split])
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG,
        "claim_id": NEW_CLAIM,
        "status": "specified",
        "rows": receipt["row_count"],
        "unique_sequences": receipt["unique_token_sequence_count"],
        "outcomes_opened": [],
    }, indent=2))


if __name__ == "__main__":
    main()
