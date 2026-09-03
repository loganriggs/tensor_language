#!/usr/bin/env python3
"""Register R576 result and its independent R579 audit in both canonical records."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    _atomic_json, _lock, circuit_path, design_key, execution_key,
    file_sha256, rebuild_registry_v2, validate_v2,
)


LIST_TAG = "task.numbered_list.index_successor"
SEQUENCE_TAG = "task.numeric_sequence.continuation"
LIST_OLD = "numbered_list_index_successor.v8"
LIST_NEW = "numbered_list_index_successor.v9"
SEQUENCE_OLD = "numeric_sequence_continuation.v4"
SEQUENCE_NEW = "numeric_sequence_continuation.v5"
COMPILE_OPEN = "numbered_list_cached_value_weights.r576.preregistered.v1"
REMOVAL_OPEN = "numbered_list_cached_value_removal.r576.preregistered.v1"
REUSE_OPEN = "numeric_sequence_cached_value_reuse.r576.preregistered.v1"
COMPILE_DONE = "numbered_list_cached_value_weights.r576.complete.held.v1"
REMOVAL_DONE = "numbered_list_cached_value_removal.r576.complete.null.v1"
AUDIT_DONE = "numbered_list_cached_value_removal_audit.r579.complete.held.v1"
REUSE_DONE = "numeric_sequence_cached_value_reuse.r576.complete.null.v1"
RESULT = BQ / "numbered_list_cached_value_weight_removal_rung576_results.json"
AUDIT = BQ / "numbered_list_cached_value_weight_removal_rung579_audit.json"
NEW_ARTIFACTS = {
    "r576_result": ("basis_aligned/bilinear_quotient/numbered_list_cached_value_weight_removal_rung576_results.json", "result"),
    "r579_audit_prereg": ("basis_aligned/polynomial_causal/NUMBERED_LIST_CACHED_VALUE_REMOVAL_AUDIT_RUNG579.md", "preregistration"),
    "r579_audit_script": ("basis_aligned/bilinear_quotient/ops/audit_cached_value_weight_removal_rung579.py", "audit_implementation"),
    "r579_audit_test": ("basis_aligned/bilinear_quotient/ops/test_audit_cached_value_weight_removal_rung579.py", "test"),
    "r579_audit": ("basis_aligned/bilinear_quotient/numbered_list_cached_value_weight_removal_rung579_audit.json", "audit"),
}


def artifact(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def bind(record: dict, event: dict) -> dict:
    event["design_key"] = design_key(record, event)
    event["execution_key"] = execution_key(record, event)
    return event


def append_artifacts(record: dict) -> None:
    for artifact_id, spec in NEW_ARTIFACTS.items():
        value = artifact(*spec)
        if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
            raise ValueError(f"artifact collision: {artifact_id}")
        record["artifacts"][artifact_id] = value


def complete_from_open(record: dict, open_id: str, done_id: str, claim_id: str,
                       verdict: str, failure_kind: str | None, notes: str) -> dict:
    event = deepcopy(next(item for item in record["evidence_events"] if item["event_id"] == open_id))
    event.update({"event_id": done_id, "claim_id": claim_id, "stage": "complete", "verdict": verdict,
                  "failure_kind": failure_kind, "result_artifact_id": "r576_result",
                  "supersedes_event_id": open_id, "notes": notes})
    event["input_artifact_ids"] = list(dict.fromkeys([
        *event["input_artifact_ids"], "r576_result", "r579_audit"]))
    return bind(record, event)


def main() -> None:
    result = json.loads(RESULT.read_text())
    audit = json.loads(AUDIT.read_text())
    assert result["pred_a_exact_weight_compilation"] is True
    assert result["all_required_gates_pass"] is False
    assert result["decision"] == "removal_or_selectivity_null"
    assert result["evaluated_splits"] == ["FIT"] and result["model_forwards"] == 123
    assert result["fit_report"]["list_necessity_pass"] is True
    assert result["fit_report"]["active_copy_controls_pass"] is False
    assert result["fit_report"]["all_sequence_successor_cells_pass"] is True
    assert audit["all_checks_pass"] is True and audit["copy_control_nonzero_term_fraction"] == 1.0
    list_path, sequence_path = circuit_path(LIST_TAG), circuit_path(SEQUENCE_TAG)
    with _lock("registry"):
        list_record = json.loads(list_path.read_text())
        sequence_record = json.loads(sequence_path.read_text())
        if any(event["event_id"] == REMOVAL_DONE for event in list_record["evidence_events"]):
            assert any(event["event_id"] == REUSE_DONE for event in sequence_record["evidence_events"])
            validate_v2(list_record); validate_v2(sequence_record)
            print(json.dumps({"status": "already_registered"}, indent=2))
            return
        append_artifacts(list_record); append_artifacts(sequence_record)

        old_list = next(claim for claim in list_record["claims"] if claim["claim_id"] == LIST_OLD)
        list_claim = deepcopy(old_list)
        list_claim.update({
            "claim_id": LIST_NEW,
            "revision": 9,
            "supersedes": LIST_OLD,
            "status": "weights_translated",
            "evidence_event_ids": [*old_list["evidence_event_ids"], COMPILE_DONE, REMOVAL_DONE, AUDIT_DONE],
            "next_missing": (
                "the final-label cached-value term is exact and necessary but not selectively removable: split its "
                "downstream-read or source/action contributions using active repeated-list and word-copy controls; "
                "do not weaken the frozen collateral thresholds or repeat whole-term deletion"),
        })
        for site in list_claim["candidate_sites"]:
            if site["site_id"] == "final_label_l0_value_through_l8h3_h7":
                site["ceiling_event_ids"] = list(dict.fromkeys([
                    *site["ceiling_event_ids"], COMPILE_DONE, REMOVAL_DONE]))
        list_record["claims"].append(list_claim)
        compile_event = complete_from_open(
            list_record, COMPILE_OPEN, COMPILE_DONE, LIST_NEW, "held", None,
            "Direct weight computation matched the activation transplant; maximum relative squared logit error 1.32e-12.")
        removal_event = complete_from_open(
            list_record, REMOVAL_OPEN, REMOVAL_DONE, LIST_NEW, "null", "scientific_null",
            "All ten FIT list necessity cells passed, but active repeated-list controls changed at 0.49-0.51 of the target margin/RMS scales and word-copy RMS reached 0.26-0.32; SELECT stayed closed.")
        audit_event = {
            "event_id": AUDIT_DONE, "claim_id": LIST_NEW, "test_type": "null_control",
            "stage": "complete", "verdict": "held", "failure_kind": None,
            "family_ids": [family["family_id"] for family in list_claim["counterfactual_families"]],
            "site_id": "final_label_l0_value_through_l8h3_h7",
            "split_plan_id": "numbered_list_successor_split_r567_v1",
            "evaluation_role": "independent_model_free_row_metric_bootstrap_split_and_verdict_audit",
            "metrics": [
                {"name": "all_saved_decisions_recomputed", "estimate": 1.0, "ci95": None,
                 "bar": "every frozen hash, row set, bootstrap, scale, split, price, and terminal decision matches"},
                {"name": "active_copy_rows", "estimate": 192, "ci95": None,
                 "bar": "all saved copy-control intervention terms are nonzero"},
            ],
            "prereg_artifact_id": "r579_audit_prereg", "result_artifact_id": "r579_audit",
            "input_artifact_ids": ["r576_result", "r579_audit_script", "r579_audit_test"],
            "seed": 576, "checkpoint_sha256": None, "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["polynomial_causal/NUMBERED_LIST_CACHED_VALUE_REMOVAL_AUDIT_RUNG579.md"],
            "notes": "Zero model calls; 20/20 independent checks held.",
        }
        list_record["evidence_events"].extend([
            compile_event, removal_event, bind(list_record, audit_event)])
        validate_v2(list_record)

        old_sequence = next(claim for claim in sequence_record["claims"] if claim["claim_id"] == SEQUENCE_OLD)
        sequence_claim = deepcopy(old_sequence)
        sequence_claim.update({
            "claim_id": SEQUENCE_NEW,
            "revision": 5,
            "supersedes": SEQUENCE_OLD,
            "status": "specified",
            "evidence_event_ids": [*old_sequence["evidence_event_ids"], REUSE_DONE],
            "next_missing": (
                "run R577 to distinguish complete relation state and first/middle score/cached/own-value factors; "
                "R576 showed strong +1 dependence on the final cached-value term but failed selective reuse because "
                "number-word copy logits changed too broadly"),
        })
        sequence_record["claims"].append(sequence_claim)
        reuse_event = complete_from_open(
            sequence_record, REUSE_OPEN, REUSE_DONE, SEQUENCE_NEW, "null", "scientific_null",
            "All six FIT digit/word/cross-format +1 necessity cells passed, but number-word copy full-vocabulary RMS exceeded the frozen selectivity bound; SELECT stayed closed.")
        sequence_record["evidence_events"].append(reuse_event)
        validate_v2(sequence_record)
        _atomic_json(list_path, list_record)
        _atomic_json(sequence_path, sequence_record)
    rebuild_registry_v2()
    print(json.dumps({"status": "registered", "list_claim": LIST_NEW,
                      "sequence_claim": SEQUENCE_NEW,
                      "events": [COMPILE_DONE, REMOVAL_DONE, AUDIT_DONE, REUSE_DONE],
                      "result_sha256": file_sha256(RESULT), "audit_sha256": file_sha256(AUDIT)}, indent=2))


if __name__ == "__main__":
    main()
