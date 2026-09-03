#!/usr/bin/env python3
"""Register the outcome-blind R552 induction rows and independent R553 audit."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import append_claim_revision, circuit_path, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

TAG = "task.induction.selector_payload"
OLD_CLAIM = "induction_selector_and_payload.v1"
NEW_CLAIM = "induction_selector_and_payload.v2"
SPLIT_ID = "induction_selector_payload_factorial_split_r552_v1"
PATHS = {
    "r552_factorial_preregistration": (
        "basis_aligned/polynomial_causal/INDUCTION_SELECTOR_PAYLOAD_FACTORIAL_ROWS_RUNG552_PREREGISTRATION.md",
        "preregistration",
    ),
    "r552_factorial_rows": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_factorial_rows_rung552.json",
        "dataset",
    ),
    "r552_factorial_rows_receipt": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_factorial_rows_rung552_receipt.json",
        "split",
    ),
    "r552_factorial_rows_builder": (
        "basis_aligned/bilinear_quotient/ops/induction_selector_payload_factorial_rows_rung552.py",
        "builder",
    ),
    "r552_factorial_rows_test": (
        "basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_factorial_rows_rung552.py",
        "test",
    ),
    "r553_factorial_rows_audit": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_factorial_rows_rung553_audit.json",
        "audit",
    ),
    "r553_factorial_rows_audit_implementation": (
        "basis_aligned/bilinear_quotient/ops/induction_selector_payload_factorial_rows_rung553_audit.py",
        "implementation",
    ),
    "r553_factorial_rows_audit_test": (
        "basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_factorial_rows_rung553_audit.py",
        "test",
    ),
}


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def family(
    family_id: str,
    role: str,
    changes: list[str],
    holds_fixed: list[str],
    controls: list[str],
) -> dict:
    return {
        "family_id": family_id,
        "role": role,
        "changes": changes,
        "holds_fixed": holds_fixed,
        "builder_artifact_id": "r552_factorial_rows_builder",
        "control_ids": controls,
        "split_plan_id": SPLIT_ID,
        "status": "frozen",
    }


def main() -> None:
    rows = json.loads((REPO / PATHS["r552_factorial_rows"][0]).read_text())
    receipt = json.loads((REPO / PATHS["r552_factorial_rows_receipt"][0]).read_text())
    audit = json.loads((REPO / PATHS["r553_factorial_rows_audit"][0]).read_text())
    assert rows["model_loaded"] is False and rows["outcomes_opened"] == []
    assert receipt["model_forwards"] == 0 and receipt["model_backwards"] == 0
    assert receipt["group_count"] == 180 and receipt["row_count"] == 1800
    assert receipt["factorial_condition_count"] == 720
    assert receipt["unique_prompt_sequence_count"] == 1440
    assert receipt["rows_sha256"] == file_sha256(REPO / PATHS["r552_factorial_rows"][0])
    assert audit["status"] == "terminal_audit_complete"
    assert audit["all_token_level_factorial_checks_pass"] is True
    assert audit["actual_variable_tokens_disjoint_across_splits"] is True
    assert audit["prompt_sequences_never_cross_groups"] is True
    assert audit["outcomes_opened"] == []

    artifacts = {key: frozen(*value) for key, value in PATHS.items()}
    split = {
        "split_plan_id": SPLIT_ID,
        "unit": "one two-source selector-by-payload factorial group, shared across every derived edge and control",
        "partition_artifact_id": "r552_factorial_rows_receipt",
        "builder_artifact_id": "r552_factorial_rows_builder",
        "seed": 552,
        "groups": {"FIT": 72, "SELECT": 36, "FINAL_TEST": 36, "OOD": 36},
        "leakage_group_keys": [
            "semantic group",
            "exact token sequence",
            "exact prompt-answer pair",
            "variable token bank",
            "prefix and layout template",
            "all four factorial conditions and their derived controls",
        ],
        "sealed_before_outcomes": True,
        "sealed_at": "2026-09-03T16:21:00Z",
    }

    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 2,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "counterfactual_families": [
                family(
                    "two_valid_sources_selector_swap",
                    "interchange",
                    ["the final query changes from source A to source C", "the correct payload changes from B to D"],
                    ["both source-to-payload pairs", "payload assignment", "length", "source positions"],
                    ["joint answer-preserving diagonal", "irrelevant-source edit", "answer-only direction"],
                ),
                family(
                    "payload_swap_match_preserved",
                    "interchange",
                    ["the followers of fixed sources A and C are exchanged", "the correct payload changes"],
                    ["source/query equality", "query token", "source positions", "prompt length"],
                    ["joint answer-preserving diagonal", "filler/lag change", "answer-only direction"],
                ),
                family(
                    "selector_payload_joint_answer_preserved",
                    "invariance",
                    ["both the queried source and the payload assignment"],
                    ["correct answer token", "two-source relation", "source positions", "payload-token set"],
                    ["both single-factor edges", "full-state response", "answer-only direction"],
                ),
                family(
                    "match_break_payload_preserved",
                    "necessity",
                    ["the earlier selected source is replaced by a decoy, removing only its query match"],
                    ["selected follower token", "final query", "original answer token", "sequence length"],
                    ["irrelevant-source edit", "offset-matched source", "token derangement"],
                ),
                family(
                    "irrelevant_source_edit",
                    "invariance",
                    ["the unselected source token is replaced by a decoy"],
                    ["selected equality edge", "selected follower", "query", "correct answer"],
                    ["match-breaking edit", "same edit at selected source", "answer-only direction"],
                ),
                family(
                    "copy_relation_preserved_nuisance_change",
                    "invariance",
                    ["filler tokens or source-to-query lag"],
                    ["query", "payload assignment", "selected source-to-payload relation", "correct answer"],
                    ["held-out lag patterns", "held-out code/trace layouts", "matched token banks"],
                ),
                copy.deepcopy(next(
                    item for item in previous["counterfactual_families"]
                    if item["family_id"] == "natural_pair_interchange"
                )),
            ],
            "split_plan_ids": [SPLIT_ID, "joint_split_v1"],
            "next_missing": (
                "preregister and run a FIT/SELECT-only native-capability screen on the frozen 2x2 factorial rows; "
                "only if capability holds, measure separate selector and payload/write complete-state ceilings; "
                "FINAL_TEST/OOD remain unopened"
            ),
        })
        append_claim_revision(TAG, claim, artifacts=artifacts, split_plans=[split])

    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG,
        "claim_id": NEW_CLAIM,
        "status": "specified",
        "rows_sha256": artifacts["r552_factorial_rows"]["sha256"],
        "audit_sha256": artifacts["r553_factorial_rows_audit"]["sha256"],
        "groups": 180,
        "rows": 1800,
        "model_forwards": 0,
        "outcomes_opened": [],
    }, indent=2))


if __name__ == "__main__":
    main()
