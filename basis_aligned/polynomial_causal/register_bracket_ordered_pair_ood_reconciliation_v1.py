#!/usr/bin/env python3
"""Reconcile passing bracket ordered-pair OOD receipts into the task authority."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BQ = REPO / "basis_aligned/bilinear_quotient"
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    _atomic_json, _lock, circuit_path, design_key, execution_key,
    file_sha256, rebuild_registry_v2, validate_v2,
)


TAG = "task.bracket_pending_opener"
OLD_CLAIM = "pending_opener_state.v29"
NEW_CLAIM = "pending_opener_state.v30"
EVENTS = (
    "pending_opener.ordered_pair_program.ood_capability.held.v1",
    "pending_opener.ordered_pair_program.ood_validation.held.v1",
    "pending_opener.ordered_pair_scalar_feasibility.held.v1",
)
ARTIFACTS = {
    "ordered_pair_program_builder": ("basis_aligned/bilinear_quotient/ops/circuit_fast_screen_candidate_bracket_l13h8_ordered_pair_displacement_program.py", "builder"),
    "ordered_pair_program_artifact": ("basis_aligned/bilinear_quotient/circuits/fast_screens/bracket_l13h8_ordered_pair_displacement_artifact_v1.json", "extracted_program"),
    "ordered_pair_ood_capability_runner": ("basis_aligned/bilinear_quotient/ops/run_bracket_l13h8_ordered_pair_program_ood_capability.py", "runner"),
    "ordered_pair_ood_capability_result": ("basis_aligned/bilinear_quotient/circuits/fast_screens/bracket_l13h8_ordered_pair_program_ood_capability_v1_result.json", "result"),
    "ordered_pair_ood_validation_runner": ("basis_aligned/bilinear_quotient/ops/run_bracket_l13h8_ordered_pair_displacement_program_ood_validation.py", "runner"),
    "ordered_pair_ood_validation_result": ("basis_aligned/bilinear_quotient/circuits/followups/bracket_l13h8_ordered_pair_displacement_program_ood_validation_v1_result.json", "result"),
    "ordered_pair_scalar_prior_art": ("basis_aligned/bilinear_quotient/circuits/prior_art/bracket_ordered_pair_suffix_free_scalar_feasibility_v1.json", "prior_art"),
    "ordered_pair_scalar_runner": ("basis_aligned/bilinear_quotient/ops/audit_bracket_ordered_pair_suffix_free_scalar_feasibility_v1.py", "audit_implementation"),
    "ordered_pair_scalar_result": ("basis_aligned/bilinear_quotient/circuits/followups/bracket_ordered_pair_suffix_free_scalar_feasibility_v1_result.json", "audit"),
}


def artifact(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def bound(record, event):
    event["design_key"] = design_key(record, event)
    event["execution_key"] = execution_key(record, event)
    return event


def event(event_id, test_type, metrics, result_artifact_id, site_id=None, notes=""):
    return {
        "event_id": event_id,
        "claim_id": NEW_CLAIM,
        "test_type": test_type,
        "stage": "complete",
        "verdict": "held",
        "failure_kind": None,
        "family_ids": [
            "direct_three_value_type_substitution",
            "completed_then_reopened_three_value_order",
            "pending_type_preserved_surface_rewrite",
            "pending_type_preserved_distance_extension",
            "pending_type_preserved_nonopener_punctuation",
        ],
        "site_id": site_id,
        "split_plan_id": "pending_opener_three_value_fresh_split_r545_v1",
        "evaluation_role": "OOD_reconciliation_from_committed_primary_receipt",
        "metrics": metrics,
        "prereg_artifact_id": None,
        "result_artifact_id": result_artifact_id,
        "input_artifact_ids": list(ARTIFACTS),
        "seed": None,
        "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "supersedes_event_id": None,
        "replicates_event_id": None,
        "sections": [ARTIFACTS[result_artifact_id][0]],
        "notes": notes,
    }


def main():
    capability = json.loads((BQ / "circuits/fast_screens/bracket_l13h8_ordered_pair_program_ood_capability_v1_result.json").read_text())
    validation = json.loads((BQ / "circuits/followups/bracket_l13h8_ordered_pair_displacement_program_ood_validation_v1_result.json").read_text())
    scalar = json.loads((BQ / "circuits/followups/bracket_ordered_pair_suffix_free_scalar_feasibility_v1_result.json").read_text())
    assert capability["terminal"] == "capability_pass" and all(capability["predictions"].values())
    assert validation["terminal"] == "program_screen" and all(validation["score"]["predictions"].values())
    assert scalar["terminal"] == "feasibility_screen" and all(scalar["score"]["predictions"].values())

    path = circuit_path(TAG)
    existing = json.loads(path.read_text())
    if all(any(item["event_id"] == key for item in existing["evidence_events"]) for key in EVENTS):
        validate_v2(existing); rebuild_registry_v2()
        print(json.dumps({"status": "already_registered", "claim_id": NEW_CLAIM}, indent=2)); return

    with _lock("registry"):
        record = json.loads(path.read_text())
        if any(any(item["event_id"] == key for item in record["evidence_events"]) for key in EVENTS):
            raise RuntimeError("partial reconciliation already exists")
        for artifact_id, spec in ARTIFACTS.items():
            value = artifact(*spec)
            if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
                raise ValueError(f"artifact collision: {artifact_id}")
            record["artifacts"][artifact_id] = value

        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 30,
            "supersedes": OLD_CLAIM,
            "status": "weights_translated",
            "evidence_event_ids": [*previous["evidence_event_ids"], *EVENTS],
            "next_missing": (
                "OOD capability and fixed six-vector ordered-pair displacement transfer already pass on two "
                "frozen constructions; a retrospective six-scalar effect table also transfers between them. "
                "Selective pair-centered necessity remains null. Next freeze a third structurally distinct, "
                "native-capable pending-opener construction and prospectively test the unchanged ordered-pair "
                "program plus the six-scalar effect prediction. Do not refit vectors or rescue distance bars."
            ),
        })
        claim["causal_variable"] = {
            **claim["causal_variable"],
            "operation": (
                "write closer-type evidence through the exact L13H8 source term; on frozen OOD rows an unchanged "
                "six-vector ordered-pair displacement program approximates exact donor-term interchange"
            ),
        }
        claim["candidate_sites"].append({
            "site_id": "attention13.head8.ordered_pair_displacement_program.final_query",
            "tensor_path": "six fixed 1152-dimensional ordered-pair displacement vectors at the exact L13H8 pending-opener term",
            "shape": [6, 1152],
            "intervention": "dispatch one immutable vector by recipient and donor closer type; zero-dispatch invariance controls",
            "ceiling_event_ids": [EVENTS[1]],
        })
        record["claims"].append(claim)

        cap_min = min(value["accuracy"] for value in capability["cell_reports"].values())
        score = validation["score"]
        scalar_score = scalar["score"]
        new_events = [
            event(EVENTS[0], "capability", [
                {"name": "minimum_ood_cell_accuracy", "estimate": cap_min, "ci95": None, "bar": ">=0.75 each of 18 cells"},
                {"name": "ood_endpoints", "estimate": 360.0, "ci95": None, "bar": "=360"},
            ], "ordered_pair_ood_capability_result", notes="Capability-only gate opens the already frozen OOD rows."),
            event(EVENTS[1], "ood", [
                {"name": "overall_effect_cosine", "estimate": score["overall"]["cosine"], "ci95": None, "bar": ">=0.65"},
                {"name": "overall_relative_l2_error", "estimate": score["overall"]["relative_l2_error"], "ci95": None, "bar": "<=0.90"},
                {"name": "overall_sign_agreement", "estimate": score["overall"]["sign_agreement"], "ci95": None, "bar": ">=0.75"},
                {"name": "control_max_absolute_logit_change", "estimate": score["control_max_absolute_logit_change"], "ci95": None, "bar": "<=1e-4"},
            ], "ordered_pair_ood_validation_result", "attention13.head8.ordered_pair_displacement_program.final_query", "All six ordered pairs and both target constructions recur; controls zero-dispatch exactly."),
            event(EVENTS[2], "cross_family_transfer", [
                {"name": "six_scalar_cross_family_cosine", "estimate": scalar_score["overall"]["cosine"], "ci95": None, "bar": ">=0.80"},
                {"name": "six_scalar_cross_family_relative_l2", "estimate": scalar_score["overall"]["relative_l2_error"], "ci95": None, "bar": "<=0.60"},
                {"name": "six_scalar_cross_family_sign_agreement", "estimate": scalar_score["overall"]["sign_agreement"], "ci95": None, "bar": ">=0.90"},
            ], "ordered_pair_scalar_result", "attention13.head8.ordered_pair_displacement_program.final_query", "Retrospective two-construction feasibility only; prospective third construction remains required."),
        ]
        for item in new_events:
            record["evidence_events"].append(bound(record, item))
        validate_v2(record)
        _atomic_json(path, record)
    rebuild_registry_v2()
    validate_v2(json.loads(path.read_text()))
    print(json.dumps({"status": "registered", "claim_id": NEW_CLAIM, "events": EVENTS}, indent=2))


if __name__ == "__main__":
    main()
