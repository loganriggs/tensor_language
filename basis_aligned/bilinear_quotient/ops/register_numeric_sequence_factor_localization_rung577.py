#!/usr/bin/env python3
"""Register reviewed R577 numeric-sequence site/factor localization before execution."""

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
OLD_CLAIM = "numeric_sequence_continuation.v3"
NEW_CLAIM = "numeric_sequence_continuation.v4"
EVENT_ID = "numeric_sequence_complete_state_factor_localization.r577.preregistered.v1"
SPLIT_ID = "numeric_sequence_continuation_split_r567_v1"
SITE_ID = "numeric_final_query_site_and_factor_ladder_r577"
PATHS = {
    "r577_semantic_positions": (
        "basis_aligned/bilinear_quotient/numeric_sequence_semantic_positions_rung577.json", "semantic_audit"),
    "r577_semantic_positions_builder": (
        "basis_aligned/bilinear_quotient/ops/numeric_sequence_semantic_positions_rung577.py", "builder"),
    "r577_semantic_positions_test": (
        "basis_aligned/bilinear_quotient/ops/test_numeric_sequence_semantic_positions_rung577.py", "test"),
    "r577_factor_preregistration": (
        "basis_aligned/polynomial_causal/NUMERIC_SEQUENCE_COMPLETE_STATE_FACTOR_LOCALIZATION_RUNG577_PREREGISTRATION.md",
        "preregistration"),
    "r577_factor_implementation": (
        "basis_aligned/bilinear_quotient/ops/numeric_sequence_complete_state_factor_localization_rung577.py",
        "implementation"),
    "r577_factor_test": (
        "basis_aligned/bilinear_quotient/ops/test_numeric_sequence_complete_state_factor_localization_rung577.py",
        "test"),
}


def artifact(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    position_data = json.loads((REPO / PATHS["r577_semantic_positions"][0]).read_text())
    assert position_data["row_count"] == 432
    assert position_data["r575_endpoint_mappings_reproduced"] == 480
    assert position_data["model_loaded"] is False and position_data["model_forwards"] == 0
    assert position_data["outcomes_opened"] == []
    artifacts = {key: artifact(*value) for key, value in PATHS.items()}
    path = circuit_path(TAG)
    with _lock("registry"):
        record = json.loads(path.read_text())
        if any(event["event_id"] == EVENT_ID for event in record["evidence_events"]):
            validate_v2(record)
            print(json.dumps({"status": "already_registered", "event_id": EVENT_ID}, indent=2))
            return
        for artifact_id, value in artifacts.items():
            if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
                raise ValueError(f"artifact collision: {artifact_id}")
            record["artifacts"][artifact_id] = value
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 4,
            "supersedes": OLD_CLAIM,
            "status": "specified",
            "evidence_event_ids": [*previous["evidence_event_ids"], EVENT_ID],
            "next_missing": (
                "run and independently audit R576 exact final-source cached-value removal plus R577 complete-state "
                "and semantic-factor localization; preserve both results because R576 tests shared final-value "
                "transport while R577 isolates relation-dependent first/middle contributions"),
        })
        site = {
            "site_id": SITE_ID,
            "tensor_path": (
                "final-query L8H7/H3 or all-head output; post-attention/MLP residual boundaries through MLP14; "
                "exact L8H7/H3 score, cached-value, own-value, and joint semantic-source terms"),
            "shape": ["batch", "final query", "site-specific 1152 residual or head-factor tensor"],
            "intervention": (
                "semantic base/donor interchange at the complete final-query boundary or exact first/middle/final "
                "attention source terms"),
            "ceiling_event_ids": [EVENT_ID],
        }
        if not any(item["site_id"] == SITE_ID for item in claim["candidate_sites"]):
            claim["candidate_sites"].append(site)
        record["claims"].append(claim)
        families = [family["family_id"] for family in claim["counterfactual_families"]]
        event = {
            "event_id": EVENT_ID,
            "claim_id": NEW_CLAIM,
            "test_type": "composition",
            "stage": "preregistered",
            "verdict": "inconclusive",
            "failure_kind": None,
            "family_ids": families,
            "site_id": SITE_ID,
            "split_plan_id": SPLIT_ID,
            "evaluation_role": "FIT_selection_then_conditional_SELECT_complete_state_and_exact_semantic_factor_localization",
            "metrics": [
                {"name": "exact_replay_and_attention_factor_algebra", "estimate": None, "ci95": None,
                 "bar": "native replay, head-source sum, and cached/own split relative squared errors <=1e-10"},
                {"name": "selected_complete_state_site", "estimate": None, "ci95": None,
                 "bar": "first FIT-passing site also passes every target, relation, and active-control gate on SELECT"},
                {"name": "l8h7_h3_shared_carrier", "estimate": None, "ci95": None,
                 "bar": "complete L8H7/H3 is the selected FIT site and holds on SELECT"},
                {"name": "selected_exact_semantic_factor", "estimate": None, "ci95": None,
                 "bar": "structurally simplest FIT-passing factor holds on SELECT; later factors cannot rescue it"},
                {"name": "execution_envelope", "estimate": None, "ci95": None,
                 "bar": "<=652 forwards, zero backwards/fitted vectors/weight updates, FINAL_TEST/OOD closed"},
            ],
            "prereg_artifact_id": "r577_factor_preregistration",
            "result_artifact_id": None,
            "input_artifact_ids": [
                "r567_rows", "r567_receipt", "r569_r570_result", "r571_audit",
                "r575_numeric_positions", "r577_semantic_positions",
                "r577_semantic_positions_builder", "r577_semantic_positions_test",
                "r577_factor_implementation", "r577_factor_test",
            ],
            "seed": 577,
            "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": [
                "polynomial_causal/NUMERIC_SEQUENCE_COMPLETE_STATE_FACTOR_LOCALIZATION_RUNG577_PREREGISTRATION.md"],
            "notes": (
                "R576 final-source cached value is a fixed external comparator and is not duplicated. Every "
                "FIT-eligible exact factor is characterized on SELECT, but only the frozen first factor can promote."),
        }
        event["design_key"] = design_key(record, event)
        event["execution_key"] = execution_key(record, event)
        record["evidence_events"].append(event)
        validate_v2(record)
        _atomic_json(path, record)
    rebuild_registry_v2()
    print(json.dumps({"status": "registered", "claim_id": NEW_CLAIM, "event_id": EVENT_ID,
                      "maximum_forwards": 652, "model_outcomes_opened": False}, indent=2))


if __name__ == "__main__":
    main()
