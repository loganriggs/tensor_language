#!/usr/bin/env python3
"""Register R567 as two canonical circuits and close the broad increment parent."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import append_artifacts, append_claim_revision, circuit_path, file_sha256, rebuild_registry_v2, validate_v2, write_behavior_circuit  # noqa: E402

PATHS = {
    "r567_rows": ("basis_aligned/bilinear_quotient/increment_two_hypothesis_rows_rung567.json", "dataset"),
    "r567_receipt": ("basis_aligned/bilinear_quotient/increment_two_hypothesis_rows_rung567_receipt.json", "split"),
    "r567_builder": ("basis_aligned/bilinear_quotient/ops/increment_two_hypothesis_rows_rung567.py", "builder"),
    "r567_test": ("basis_aligned/bilinear_quotient/ops/test_increment_two_hypothesis_rows_rung567.py", "test"),
    "r567_prereg": ("basis_aligned/polynomial_causal/INCREMENT_TWO_HYPOTHESIS_FRESH_FREEZE_RUNG567.md", "preregistration"),
    "r566_development": ("basis_aligned/bilinear_quotient/increment_format_exploration_rung566_results.json", "exploratory_result"),
}


def artifacts() -> dict:
    return {key: {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}
            for key, (path, kind) in PATHS.items()}


def family(family_id: str, role: str, changes: list[str], held: list[str], controls: list[str], split_id: str) -> dict:
    return {"family_id": family_id, "role": role, "changes": changes, "holds_fixed": held,
            "builder_artifact_id": "r567_builder", "control_ids": controls,
            "split_plan_id": split_id, "status": "frozen"}


def split(split_id: str, hypothesis: str) -> dict:
    return {
        "split_plan_id": split_id,
        "unit": f"fresh content-addressed {hypothesis} semantic group with every derived family",
        "partition_artifact_id": "r567_receipt", "builder_artifact_id": "r567_builder", "seed": 567,
        "groups": {"FIT": 32, "SELECT": 16, "FINAL_TEST": 16, "OOD": 16},
        "leakage_group_keys": ["exact prompt pair", "R566 development prompt", "start-value pool", "content-word pool", "all derived families"],
        "sealed_before_outcomes": True, "sealed_at": "2026-09-03T18:40:00Z",
    }


def list_record() -> dict:
    split_id = "numbered_list_successor_split_r567_v1"
    return {
        "schema_version": 2, "tag": "task.numbered_list.index_successor",
        "identity": {"kind": "behavior_circuit", "instance": None, "identity_artifact_id": "r567_rows",
                     "aliases": ["numbered_list_successor", "legacy_increment_list"]},
        "claims": [{
            "claim_id": "numbered_list_index_successor.v1", "revision": 1, "status": "specified", "supersedes": None,
            "causal_variable": {"id": "list_index_state", "domain": "integer labels in numbered lists",
                "read": "the final visible list label and list structure", "operation": "advance the list index by one",
                "write": "evidence for the next digit label", "endpoint": "next-list-label numeric-candidate logit margin"},
            "alternative_explanations": ["general arithmetic progression", "copy the final label", "memorized prompt positions"],
            "counterfactual_families": [
                family("list_two_line_state_shift", "interchange", ["two-line list state and next label"], ["list length", "content nouns", "step of one"], ["three-line transfer", "repeated labels", "step-two conflict"], split_id),
                family("list_three_line_state_shift", "interchange", ["three-line list state and next label"], ["list length", "content nouns", "step of one"], ["two-line transfer", "middle break", "step-two conflict"], split_id),
                family("list_surface_preserved", "invariance", ["content nouns and their order"], ["all labels", "list length", "next label"], ["state shifts", "held-out nouns", "middle break"], split_id),
                family("list_middle_index_break", "necessity", ["middle label and coherence of the visible +1 run"], ["first label", "final label", "registered successor"], ["surface edit", "state shift", "repeated labels"], split_id),
                family("list_repeated_index_control", "invariance", ["content nouns and their order"], ["repeated label", "copy rule", "registered answer"], ["+1 rows", "step-two conflict", "held-out nouns"], split_id),
                family("list_step_two_conflict", "invariance", ["content nouns and their order"], ["step-two visible labels", "last-label successor answer", "list format"], ["arithmetic +2 answer", "+1 rows", "copy rows"], split_id),
            ],
            "candidate_sites": [
                {"site_id": "l8h7_l8h3_value_paths", "tensor_path": "L8H7/L8H3 exact value contributions",
                 "shape": ["batch", "position", "projected value"], "intervention": "transplant exact selected-source value contributions", "ceiling_event_ids": []},
                {"site_id": "mlp8_14_successor_writes", "tensor_path": "MLP8 through MLP14 product/write terms",
                 "shape": ["batch", "position", "residual width"], "intervention": "replace exact source-conditioned product terms or complete writes", "ceiling_event_ids": []},
            ],
            "split_plan_ids": [split_id], "evidence_event_ids": [], "translation_ids": [],
            "next_missing": "run a preregistered FIT/SELECT native gate on all six R567 list families before retesting the legacy L8H7/L8H3 and MLP8-14 hypothesis",
        }],
        "split_plans": [split(split_id, "numbered-list")], "evidence_events": [], "translations": [],
        "artifacts": artifacts(), "provenance": {"rung": 567, "split_from": "task.increment.state"},
    }


def sequence_record() -> dict:
    split_id = "numeric_sequence_continuation_split_r567_v1"
    return {
        "schema_version": 2, "tag": "task.numeric_sequence.continuation",
        "identity": {"kind": "behavior_circuit", "instance": None, "identity_artifact_id": "r567_rows",
                     "aliases": ["numeric_sequence_continuation", "digit_word_sequence"]},
        "claims": [{
            "claim_id": "numeric_sequence_continuation.v1", "revision": 1, "status": "specified", "supersedes": None,
            "causal_variable": {"id": "numeric_sequence_state_and_rule", "domain": "comma-separated digit or number-word sequences",
                "read": "observed numeric values and their relation", "operation": "predict the next value under +1 or copy",
                "write": "evidence for the next digit or number-word token", "endpoint": "representation-matched numeric-candidate logit margin"},
            "alternative_explanations": ["last-value successor without reading the relation", "format-specific memorization", "generic numeric-token prior"],
            "counterfactual_families": [
                family("sequence_digit_state_shift", "interchange", ["digit sequence state and next value"], ["digit representation", "+1 relation", "prompt form"], ["word transfer", "copy", "+2 conflict"], split_id),
                family("sequence_word_state_shift", "interchange", ["number-word sequence state and next value"], ["word representation", "+1 relation", "prompt form"], ["digit transfer", "copy", "+2 conflict"], split_id),
                family("sequence_cross_format_shift", "interchange", ["numeric state", "digit versus word representation", "next token"], ["+1 relation", "semantic group", "content word"], ["within-format shifts", "surface edits", "+2 conflict"], split_id),
                family("sequence_digit_surface_preserved", "invariance", ["content word and prompt wording"], ["digit values", "+1 relation", "answer"], ["state shift", "middle break", "word surface"], split_id),
                family("sequence_word_surface_preserved", "invariance", ["content word and prompt wording"], ["number-word values", "+1 relation", "answer"], ["state shift", "middle break", "digit surface"], split_id),
                family("sequence_middle_value_break", "necessity", ["middle value and evidence for the +1 relation"], ["first value", "final value", "registered answer"], ["surface edit", "copy", "last-value successor alternative"], split_id),
                family("sequence_digit_copy_control", "invariance", ["content word and prompt wording"], ["repeated digit", "copy rule", "answer"], ["+1 rows", "word copy", "+2 conflict"], split_id),
                family("sequence_word_copy_control", "invariance", ["content word and prompt wording"], ["repeated number word", "copy rule", "answer"], ["+1 rows", "digit copy", "+2 conflict"], split_id),
                family("sequence_step_two_conflict", "invariance", ["surface wording only"], ["step-two values", "arithmetic and last-successor candidates", "digit representation"], ["+1 rows", "copy rows", "native full-vocabulary baseline"], split_id),
            ],
            "candidate_sites": [
                {"site_id": "cross_format_state_interface", "tensor_path": "unlocalized digit/word state interface",
                 "shape": ["batch", "position", "site-specific feature"], "intervention": "complete-state interchange before fitting any subspace", "ceiling_event_ids": []},
                {"site_id": "successor_output_terms", "tensor_path": "downstream MLP product/write terms",
                 "shape": ["batch", "position", "residual width"], "intervention": "test downstream-equivalent exact product terms across digit and word inputs", "ceiling_event_ids": []},
            ],
            "split_plan_ids": [split_id], "evidence_event_ids": [], "translation_ids": [],
            "next_missing": "run a preregistered FIT/SELECT native gate that scores digit, word, cross-format, middle-break, copy, and +2-conflict cells separately",
        }],
        "split_plans": [split(split_id, "numeric-sequence")], "evidence_events": [], "translations": [],
        "artifacts": artifacts(), "provenance": {"rung": 567, "split_from": "task.increment.state"},
    }


def main() -> None:
    rows = json.loads((REPO / PATHS["r567_rows"][0]).read_text())
    receipt = json.loads((REPO / PATHS["r567_receipt"][0]).read_text())
    assert rows["outcomes_opened"] == [] and rows["row_count"] == 1200
    assert receipt["development_sequence_overlap"] == 0
    for record in (list_record(), sequence_record()):
        path = circuit_path(record["tag"])
        if not path.exists():
            write_behavior_circuit(record)
        else:
            validate_v2(json.loads(path.read_text()))

    parent_tag = "task.increment.state"
    append_artifacts(parent_tag, artifacts())
    parent = json.loads(circuit_path(parent_tag).read_text())
    if not any(claim["claim_id"] == "increment_state.v5" for claim in parent["claims"]):
        previous = next(claim for claim in parent["claims"] if claim["claim_id"] == "increment_state.v4")
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": "increment_state.v5", "revision": 5, "status": "rejected", "supersedes": "increment_state.v4",
            "next_missing": ("broad increment identity split after R566; do not run more experiments under this tag. "
                             "Use task.numbered_list.index_successor and task.numeric_sequence.continuation"),
        })
        append_claim_revision(parent_tag, claim)
    for tag in (parent_tag, "task.numbered_list.index_successor", "task.numeric_sequence.continuation"):
        validate_v2(json.loads(circuit_path(tag).read_text()))
    rebuild_registry_v2()
    print(json.dumps({"rung": 567, "parent": "rejected_and_split",
                      "children": ["task.numbered_list.index_successor", "task.numeric_sequence.continuation"],
                      "rows": 1200, "outcomes_opened": []}, indent=2))


if __name__ == "__main__":
    main()
