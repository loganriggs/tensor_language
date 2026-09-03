#!/usr/bin/env python3
"""Register R562's fresh, outcome-free increment/successor authority."""

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
OLD_CLAIM, NEW_CLAIM = "increment_state.v1", "increment_state.v2"
SPLIT_ID = "increment_counterfactual_split_r562_v1"
PATHS = {
    "r562_rows": ("basis_aligned/bilinear_quotient/increment_counterfactual_authority_rung562.json", "dataset"),
    "r562_receipt": ("basis_aligned/bilinear_quotient/increment_counterfactual_authority_rung562_receipt.json", "split"),
    "r562_builder": ("basis_aligned/bilinear_quotient/ops/increment_counterfactual_authority_rung562.py", "builder"),
    "r562_test": ("basis_aligned/bilinear_quotient/ops/test_increment_counterfactual_authority_rung562.py", "test"),
    "r562_prereg": ("basis_aligned/polynomial_causal/INCREMENT_COUNTERFACTUAL_AUTHORITY_RUNG562_PREREGISTRATION.md", "preregistration"),
}


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def family(family_id: str, role: str, changes: list[str], holds: list[str], controls: list[str]) -> dict:
    return {
        "family_id": family_id, "role": role, "changes": changes, "holds_fixed": holds,
        "builder_artifact_id": "r562_builder", "control_ids": controls,
        "split_plan_id": SPLIT_ID, "status": "frozen",
    }


def main() -> None:
    artifacts = {key: frozen(*value) for key, value in PATHS.items()}
    rows = json.loads((REPO / PATHS["r562_rows"][0]).read_text())
    receipt = json.loads((REPO / PATHS["r562_receipt"][0]).read_text())
    assert rows["row_count"] == 1120 and rows["group_count"] == 160
    assert rows["outcomes_opened"] == [] and rows["model_loaded"] is False
    assert receipt["all_groups_have_all_families"] and receipt["all_answer_endpoints_single_token"]
    split = {
        "split_plan_id": SPLIT_ID,
        "unit": "content-addressed semantic group shared across seven target, necessity, and control families",
        "partition_artifact_id": "r562_receipt",
        "builder_artifact_id": "r562_builder",
        "seed": 562,
        "groups": {"FIT": 64, "SELECT": 32, "FINAL_TEST": 32, "OOD": 32},
        "leakage_group_keys": [
            "semantic group", "oriented start values", "lexical pool", "prompt lead",
            "surface style", "all seven derived families",
        ],
        "sealed_before_outcomes": True,
        "sealed_at": "2026-09-03T17:55:00Z",
    }
    record = json.loads(circuit_path(TAG).read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 2,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "counterfactual_families": [
                family("digit_coherent_shift", "interchange",
                       ["numeric state and correct next value", "coherent shift of a digit-written +1 sequence"],
                       ["+1 operation", "prompt format", "word triplet"],
                       ["repeated-number control", "step-two control", "surface rewrite"]),
                family("word_coherent_shift", "interchange",
                       ["numeric state and correct next value", "coherent shift of a number-word +1 sequence"],
                       ["+1 operation", "prompt format", "word triplet"],
                       ["digit transfer", "repeated-number control", "step-two control"]),
                family("cross_format_coherent_shift", "interchange",
                       ["numeric state", "digit versus number-word representation", "correct next token"],
                       ["+1 operation", "semantic group", "word triplet"],
                       ["within-format shifts", "surface rewrite", "nonincrement numeric controls"]),
                family("incoherent_middle_number_edit", "necessity",
                       ["only the middle number and therefore evidence for a coherent +1 relation"],
                       ["first number", "final observed number", "registered expected answer"],
                       ["surface rewrite", "coherent shift", "nonincrement numeric controls"]),
                family("operation_preserved_surface_edit", "invariance",
                       ["word order", "separators", "surface template"],
                       ["all numeric values", "+1 operation", "correct answer"],
                       ["coherent state shift", "middle-number break", "held-out lexical pools"]),
                family("repeated_number_numeric_control", "invariance",
                       ["word order", "separators", "surface template"],
                       ["repeated numeric state", "copy/repeat rule", "correct answer"],
                       ["+1 target rows", "step-two rows", "held-out lexical pools"]),
                family("step_two_numeric_control", "invariance",
                       ["word order", "separators", "surface template"],
                       ["numeric state", "+2 operation", "correct answer"],
                       ["+1 target rows", "repeated-number rows", "held-out number pools"]),
            ],
            "split_plan_ids": [SPLIT_ID],
            "next_missing": (
                "preregister FIT/SELECT native capability on all seven R562 families; require digit and number-word "
                "success, cross-format transfer, middle-edit sensitivity, and stability on both nonincrement controls "
                "before reopening L8H7/L8H3 or MLP8-14 localization"
            ),
        })
        append_claim_revision(TAG, claim, artifacts=artifacts, split_plans=[split])
    final = json.loads(circuit_path(TAG).read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "status": "specified",
        "rows_sha256": artifacts["r562_rows"]["sha256"], "groups": 160,
        "rows": 1120, "outcomes_opened": [],
    }, indent=2))


if __name__ == "__main__":
    main()
