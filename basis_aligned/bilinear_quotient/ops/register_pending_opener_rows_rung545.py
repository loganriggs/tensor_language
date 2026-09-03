#!/usr/bin/env python3
"""Register R545's fresh, outcome-free three-value row authority."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import append_claim_revision, circuit_path, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

TAG = "task.bracket.pending_opener"
OLD_CLAIM, NEW_CLAIM = "pending_opener_state.v16", "pending_opener_state.v17"
SPLIT_ID = "pending_opener_three_value_fresh_split_r545_v1"
PATHS = {
    "r545_three_value_rows": ("basis_aligned/bilinear_quotient/pending_opener_three_value_fresh_rows_rung545.json", "dataset"),
    "r545_three_value_rows_receipt": ("basis_aligned/bilinear_quotient/pending_opener_three_value_fresh_rows_rung545_receipt.json", "split"),
    "r545_three_value_rows_builder": ("basis_aligned/bilinear_quotient/ops/pending_opener_three_value_fresh_rows_rung545.py", "builder"),
    "r545_three_value_rows_test": ("basis_aligned/bilinear_quotient/ops/test_pending_opener_three_value_fresh_rows_rung545.py", "test"),
    "r545_three_value_rows_preregistration": ("basis_aligned/polynomial_causal/PENDING_OPENER_THREE_VALUE_FRESH_ROWS_RUNG545_PREREGISTRATION.md", "preregistration"),
}


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def family(family_id: str, role: str, changes: list[str], holds_fixed: list[str], controls: list[str]) -> dict:
    return {
        "family_id": family_id, "role": role, "changes": changes, "holds_fixed": holds_fixed,
        "builder_artifact_id": "r545_three_value_rows_builder", "control_ids": controls,
        "split_plan_id": SPLIT_ID, "status": "frozen",
    }


def main() -> None:
    artifacts = {key: frozen(*value) for key, value in PATHS.items()}
    rows = json.loads((REPO / PATHS["r545_three_value_rows"][0]).read_text())
    receipt = json.loads((REPO / PATHS["r545_three_value_rows_receipt"][0]).read_text())
    assert rows["outcomes_opened"] == [] and rows["model_loaded"] is False
    assert rows["row_count"] == 900 and rows["group_count"] == 180
    assert receipt["unique_prompt_pair_count"] == 900 and receipt["unique_token_sequence_count"] == 1800
    split = {
        "split_plan_id": SPLIT_ID,
        "unit": "fresh content-addressed semantic group shared across five families and six ordered delimiter pairs",
        "partition_artifact_id": "r545_three_value_rows_receipt",
        "builder_artifact_id": "r545_three_value_rows_builder",
        "seed": 545,
        "groups": {"FIT": 72, "SELECT": 36, "FINAL_TEST": 36, "OOD": 36},
        "leakage_group_keys": [
            "exact token sequence", "exact oriented prompt pair", "prefix pool", "word pool",
            "ordered delimiter pair", "template", "all R543 prefixes and words",
        ],
        "sealed_before_outcomes": True,
        "sealed_at": "2026-09-03T15:38:00Z",
    }
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM, "revision": 17, "status": "specified", "supersedes": OLD_CLAIM,
            "counterfactual_families": [
                family(
                    "direct_three_value_type_substitution", "interchange",
                    ["pending delimiter type and correct closer", "one opener token among parenthesis, square, and quote"],
                    ["ordinary words", "sequence length", "opener position", "body order"],
                    ["wrong closer", "three answer-preserving families", "endpoint-gradient overlap"],
                ),
                family(
                    "completed_then_reopened_three_value_order", "interchange",
                    ["which of two delimiter types remains pending", "correct closer"],
                    ["ordinary-word multiset", "sequence length", "final lexical suffix", "one completed span"],
                    ["direct construction", "same-variable surface rewrite", "wrong closer"],
                ),
                family(
                    "pending_type_preserved_surface_rewrite", "invariance",
                    ["surface wording", "ordinary-word order"],
                    ["pending delimiter type", "correct closer", "semantic group"],
                    ["complete-state effect", "held-out lexical pool", "random subspace"],
                ),
                family(
                    "pending_type_preserved_distance_extension", "invariance",
                    ["opener-to-final distance", "neutral intervening clause"],
                    ["pending delimiter type", "correct closer", "core content words"],
                    ["complete-state effect", "short versus long", "random subspace"],
                ),
                family(
                    "pending_type_preserved_nonopener_punctuation", "invariance",
                    ["one comma token to one colon token before the opener"],
                    ["pending delimiter type", "correct closer", "length", "all tokens after the edit"],
                    ["complete-state effect", "wrong closer", "random subspace"],
                ),
            ],
            "split_plan_ids": [SPLIT_ID],
            "next_missing": (
                "run a preregistered FIT/SELECT native-capability and L13H8 complete-state confirmation on R545; "
                "do not fit a projector or open FINAL_TEST/OOD until that result is audited"
            ),
        })
        append_claim_revision(TAG, claim, artifacts=artifacts, split_plans=[split])
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "status": "specified",
        "rows_sha256": artifacts["r545_three_value_rows"]["sha256"],
        "groups": 180, "rows": 900, "outcomes_opened": [],
    }, indent=2))


if __name__ == "__main__":
    main()
