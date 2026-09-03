#!/usr/bin/env python3
"""Register the outcome-blind R578 induction rows in the canonical circuit record."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import _atomic_json, _lock, circuit_path, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402


TAG = "task.induction.selector_payload"
OLD_CLAIM = "induction_selector_and_payload.v6"
NEW_CLAIM = "induction_selector_and_payload.v7"
SPLIT_ID = "induction_selector_payload_three_source_split_r578_v1"
PATHS = {
    "r578_three_source_preregistration": (
        "basis_aligned/polynomial_causal/INDUCTION_SELECTOR_PAYLOAD_THREE_SOURCE_ROWS_RUNG578_PREREGISTRATION.md",
        "preregistration"),
    "r578_three_source_rows": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_three_source_rows_rung578.json",
        "dataset"),
    "r578_three_source_receipt": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_three_source_rows_rung578_receipt.json",
        "split"),
    "r578_three_source_builder": (
        "basis_aligned/bilinear_quotient/ops/induction_selector_payload_three_source_rows_rung578.py",
        "builder"),
    "r578_three_source_test": (
        "basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_three_source_rows_rung578.py",
        "test"),
}


def artifact(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def family(family_id: str, role: str, changes: list[str], holds: list[str], controls: list[str]) -> dict:
    return {"family_id": family_id, "role": role, "changes": changes, "holds_fixed": holds,
            "builder_artifact_id": "r578_three_source_builder", "control_ids": controls,
            "split_plan_id": SPLIT_ID, "status": "frozen"}


def main() -> None:
    rows = json.loads((REPO / PATHS["r578_three_source_rows"][0]).read_text())
    receipt = json.loads((REPO / PATHS["r578_three_source_receipt"][0]).read_text())
    assert rows["model_loaded"] is False and rows["outcomes_opened"] == []
    assert rows["group_count"] == 180 and rows["row_count"] == 5400
    assert receipt["group_count"] == 180 and receipt["factorial_condition_count"] == 720
    assert receipt["unique_prompt_sequence_count"] == 5040
    assert receipt["rows_sha256"] == file_sha256(REPO / PATHS["r578_three_source_rows"][0])
    assert receipt["model_forwards"] == receipt["model_backwards"] == 0
    assert receipt["outcomes_opened"] == []
    artifacts = {key: artifact(*value) for key, value in PATHS.items()}
    split = {
        "split_plan_id": SPLIT_ID,
        "unit": "one complete three-pair selector-by-payload group with all factorial cells and controls",
        "partition_artifact_id": "r578_three_source_receipt",
        "builder_artifact_id": "r578_three_source_builder",
        "seed": 578,
        "groups": {"FIT": 72, "SELECT": 36, "FINAL_TEST": 36, "OOD": 36},
        "leakage_group_keys": [
            "semantic group", "sampled token block", "exact token sequence", "exact prompt-answer pair",
            "all four selector-by-payload conditions", "all derived controls", "prefix and layout family",
        ],
        "sealed_before_outcomes": True,
        "sealed_at": "2026-09-03T18:40:00Z",
    }
    path = circuit_path(TAG)
    with _lock("registry"):
        record = json.loads(path.read_text())
        if any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
            validate_v2(record)
            print(json.dumps({"status": "already_registered", "claim_id": NEW_CLAIM}, indent=2))
            return
        for artifact_id, value in artifacts.items():
            if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
                raise ValueError(f"artifact collision: {artifact_id}")
            record["artifacts"][artifact_id] = value
        if any(item["split_plan_id"] == SPLIT_ID for item in record["split_plans"]):
            raise ValueError(f"split collision: {SPLIT_ID}")
        record["split_plans"].append(split)
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        natural = deepcopy(next(item for item in previous["counterfactual_families"]
                                if item["family_id"] == "natural_pair_interchange"))
        claim = deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 7,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "counterfactual_families": [
                family("two_valid_sources_selector_swap", "interchange",
                       ["the final query selects A rather than C or vice versa", "the correct payload changes"],
                       ["all three source-payload pairs", "payload assignment", "filler and pair positions"],
                       ["selector_payload_joint_answer_preserved", "irrelevant_source_edit", "copy_relation_preserved_nuisance_change"]),
                family("payload_swap_match_preserved", "interchange",
                       ["the B and D payload assignments to A and C are exchanged", "the correct payload changes"],
                       ["query and selector", "source identities and positions", "neutral X-to-E pair"],
                       ["selector_payload_joint_answer_preserved", "irrelevant_payload_edit", "copy_relation_preserved_nuisance_change"]),
                family("selector_payload_joint_answer_preserved", "invariance",
                       ["selector and target payload assignment change together"],
                       ["correct answer", "source and payload token sets", "neutral pair and filler"],
                       ["two_valid_sources_selector_swap", "payload_swap_match_preserved", "match_break_payload_preserved"]),
                family("match_break_payload_preserved", "necessity",
                       ["the selected earlier source is replaced, breaking its only query match"],
                       ["query token", "both target payload tokens", "contrast and neutral pairs"],
                       ["irrelevant_source_edit", "contrast_target_source_edit", "irrelevant_payload_edit"]),
                family("irrelevant_source_edit", "invariance",
                       ["the endpoint-neutral X source is replaced by a decoy"],
                       ["both target pairs", "query and answers", "neutral E payload"],
                       ["match_break_payload_preserved", "irrelevant_payload_edit", "contrast_target_source_edit"]),
                family("irrelevant_payload_edit", "invariance",
                       ["the endpoint-neutral E payload is replaced by a decoy"],
                       ["both target pairs", "query and answers", "neutral X source"],
                       ["match_break_payload_preserved", "irrelevant_source_edit", "contrast_target_source_edit"]),
                family("contrast_target_source_edit", "invariance",
                       ["the unselected target source immediately before the competing payload is replaced"],
                       ["selected target pair", "query", "both answer-token identities"],
                       ["match_break_payload_preserved", "irrelevant_source_edit", "irrelevant_payload_edit"]),
                family("copy_relation_preserved_nuisance_change", "invariance",
                       ["filler identities or the source-to-query lag"],
                       ["all three source-payload pairs", "selector and payload assignment", "correct answer"],
                       ["irrelevant_source_edit", "irrelevant_payload_edit", "match_break_payload_preserved"]),
                natural,
            ],
            "split_plan_ids": [SPLIT_ID, "joint_split_v1"],
            "next_missing": (
                "implement and preregister the R580 FIT/SELECT native-capability screen on the frozen three-source "
                "rows, saving row-level group measurements for an independent audit; do not reopen R552 or run "
                "factor-level model-site searches before capability holds"),
        })
        record["claims"].append(claim)
        validate_v2(record)
        _atomic_json(path, record)
    rebuild_registry_v2()
    print(json.dumps({"status": "registered", "claim_id": NEW_CLAIM,
                      "split_plan_id": SPLIT_ID, "groups": 180, "rows": 5400,
                      "model_forwards": 0, "outcomes_opened": []}, indent=2))


if __name__ == "__main__":
    main()
