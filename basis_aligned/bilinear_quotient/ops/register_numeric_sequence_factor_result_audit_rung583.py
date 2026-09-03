#!/usr/bin/env python3
"""Register the R577 complete-state null and held independent R583 audit."""

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


TAG = "task.numeric_sequence.continuation"
OLD_CLAIM = "numeric_sequence_continuation.v5"
NEW_CLAIM = "numeric_sequence_continuation.v6"
OPEN_EVENT = "numeric_sequence_complete_state_factor_localization.r577.preregistered.v1"
DONE_EVENT = "numeric_sequence_complete_state_factor_localization.r577.complete.null.v1"
AUDIT_EVENT = "numeric_sequence_factor_localization_audit.r583.complete.held.v1"
SITE_ID = "numeric_final_query_site_and_factor_ladder_r577"
SPLIT_ID = "numeric_sequence_continuation_split_r567_v1"
RESULT = BQ / "numeric_sequence_complete_state_factor_localization_rung577_results.json"
AUDIT = BQ / "numeric_sequence_factor_localization_audit_rung583.json"
ARTIFACTS = {
    "r577_factor_result": (
        "basis_aligned/bilinear_quotient/numeric_sequence_complete_state_factor_localization_rung577_results.json",
        "result",
    ),
    "r577_factor_runlog": (
        "basis_aligned/bilinear_quotient/runlogs/numeric_sequence_complete_state_factor_localization_rung577.log",
        "runlog",
    ),
    "r583_audit_preregistration": (
        "basis_aligned/polynomial_causal/NUMERIC_SEQUENCE_FACTOR_LOCALIZATION_AUDIT_RUNG583_PREREGISTRATION.md",
        "preregistration",
    ),
    "r583_audit_implementation": (
        "basis_aligned/bilinear_quotient/ops/audit_numeric_sequence_factor_localization_rung583.py",
        "audit_implementation",
    ),
    "r583_audit_test": (
        "basis_aligned/bilinear_quotient/ops/test_audit_numeric_sequence_factor_localization_rung583.py",
        "test",
    ),
    "r583_audit_dryrun": (
        "basis_aligned/bilinear_quotient/numeric_sequence_factor_localization_audit_rung583_dryrun.json",
        "dryrun_receipt",
    ),
    "r583_audit_result": (
        "basis_aligned/bilinear_quotient/numeric_sequence_factor_localization_audit_rung583.json",
        "audit",
    ),
}


def artifact(path: str, kind: str) -> dict:
    return {
        "path": path,
        "sha256": file_sha256(REPO / path),
        "kind": kind,
        "status": "frozen",
    }


def bind(record: dict, event: dict) -> dict:
    event["design_key"] = design_key(record, event)
    event["execution_key"] = execution_key(record, event)
    return event


def main() -> None:
    result = json.loads(RESULT.read_text())
    audit = json.loads(AUDIT.read_text())
    assert result["decision"] == "complete_state_site_null"
    assert result["pred_a_exact_replay_and_semantic_factor_algebra"] is True
    assert result["pred_b_complete_state_site_holds_fit_and_select"] is False
    assert result["site_choice"]["eligible_arms"] == []
    assert result["factor_choice"] is None
    assert result["evaluated_splits"] == ["FIT"]
    assert result["execution_price"]["observed_forwards"] == 205
    assert result["execution_price"]["model_backwards"] == 0
    assert audit["audit_verdict"] == "held_independent_audit"
    assert audit["audit_failures"] == []
    assert audit["independently_recomputed_scientific_decision"] == result["decision"]
    assert audit["bootstrap_cell_count"] == 56
    assert audit["knowledge_packet"]["all_control_interventions_strictly_nonzero"] is True
    assert audit["result_receipt_present"] is False

    path = circuit_path(TAG)
    with _lock("registry"):
        record = json.loads(path.read_text())
        if any(event["event_id"] == DONE_EVENT for event in record["evidence_events"]):
            assert any(event["event_id"] == AUDIT_EVENT for event in record["evidence_events"])
            validate_v2(record)
            print(json.dumps({"status": "already_registered", "claim_id": NEW_CLAIM}, indent=2))
            return
        for artifact_id, spec in ARTIFACTS.items():
            value = artifact(*spec)
            if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
                raise ValueError(f"artifact collision: {artifact_id}")
            record["artifacts"][artifact_id] = value

        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 6,
            "supersedes": OLD_CLAIM,
            "status": "specified",
            "evidence_event_ids": [*previous["evidence_event_ids"], DONE_EVENT, AUDIT_EVENT],
            "next_missing": (
                "R577 proves that complete final-query head/state swaps are broad rather than selective: split the "
                "transported numeric carrier by source/action or downstream bilinear use, with active surface/copy "
                "controls; do not repeat larger residual swaps or open R577's factor stage"
            ),
        })
        for site in claim["candidate_sites"]:
            if site["site_id"] == SITE_ID:
                site["ceiling_event_ids"] = list(dict.fromkeys([
                    *site["ceiling_event_ids"], DONE_EVENT, AUDIT_EVENT,
                ]))
        record["claims"].append(claim)

        open_event = deepcopy(next(
            event for event in record["evidence_events"] if event["event_id"] == OPEN_EVENT
        ))
        open_event.update({
            "event_id": DONE_EVENT,
            "claim_id": NEW_CLAIM,
            "stage": "complete",
            "verdict": "null",
            "failure_kind": "scientific_null",
            "result_artifact_id": "r577_factor_result",
            "input_artifact_ids": list(dict.fromkeys([
                *open_event["input_artifact_ids"], "r577_factor_result", "r577_factor_runlog",
            ])),
            "supersedes_event_id": OPEN_EVENT,
            "notes": (
                "Exact replay/algebra held, but none of seven complete final-query sites passed target, relation, "
                "and active-control gates together. SELECT and the finer factor stage stayed closed; 205 forwards, "
                "zero backwards."
            ),
        })
        open_event["metrics"] = [
            {"name": "exact_replay_and_attention_factor_algebra", "estimate": 1.0, "ci95": None,
             "bar": "native replay, source sum, and cached/own split errors <=1e-10"},
            {"name": "complete_state_sites_eligible", "estimate": 0, "ci95": None,
             "bar": "at least one site passes every FIT target, relation, and active-control gate"},
            {"name": "a8_h73_target_and_relation_cells", "estimate": 8, "ci95": None,
             "bar": "8/8 target/relation direction cells pass; reported but cannot rescue 7/10 controls"},
            {"name": "execution_envelope", "estimate": 205, "ci95": None,
             "bar": "<=652 forwards, zero backwards, FIT only, FINAL_TEST/OOD closed"},
        ]

        audit_event = {
            "event_id": AUDIT_EVENT,
            "claim_id": NEW_CLAIM,
            "test_type": "null_control",
            "stage": "complete",
            "verdict": "held",
            "failure_kind": None,
            "family_ids": [family["family_id"] for family in claim["counterfactual_families"]],
            "site_id": SITE_ID,
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "independent_model_free_raw_row_bootstrap_liveness_and_terminal_null_audit",
            "metrics": [
                {"name": "seeded_bootstrap_cells_recomputed", "estimate": 56, "ci95": None,
                 "bar": "all 56 saved 2,000-draw bootstrap cells match"},
                {"name": "active_control_interventions", "estimate": 1.0, "ci95": None,
                 "bar": "every saved control intervention is strictly nonzero and each median liveness gate passes"},
                {"name": "terminal_decision_recomputed", "estimate": 1.0, "ci95": None,
                 "bar": "complete_state_site_null, no eligible site, SELECT/factor stage closed, 205 forwards"},
            ],
            "prereg_artifact_id": "r583_audit_preregistration",
            "result_artifact_id": "r583_audit_result",
            "input_artifact_ids": [
                "r577_factor_result", "r577_factor_runlog", "r583_audit_implementation",
                "r583_audit_test", "r583_audit_dryrun",
            ],
            "seed": 583,
            "checkpoint_sha256": result["checkpoint_weights_sha256"],
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": [
                "polynomial_causal/NUMERIC_SEQUENCE_FACTOR_LOCALIZATION_AUDIT_RUNG583_PREREGISTRATION.md"
            ],
            "notes": (
                "Audit held with zero model calls. No contemporaneous R577 result receipt exists; the audit instead "
                "binds exact result bytes, runlog, append-only completion record, checkpoint, and code authorities."
            ),
        }
        record["evidence_events"].extend([
            bind(record, open_event), bind(record, audit_event),
        ])
        validate_v2(record)
        _atomic_json(path, record)
    rebuild_registry_v2()
    print(json.dumps({
        "status": "registered",
        "claim_id": NEW_CLAIM,
        "events": [DONE_EVENT, AUDIT_EVENT],
        "result_sha256": file_sha256(RESULT),
        "audit_sha256": file_sha256(AUDIT),
    }, indent=2))


if __name__ == "__main__":
    main()
