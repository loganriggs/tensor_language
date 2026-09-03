#!/usr/bin/env python3
"""Atomically register held R573 v2 evidence and its R574 CPU audit."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


BQ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    _atomic_json, _lock, circuit_path, design_key, execution_key,
    rebuild_registry_v2, validate_v2,
)


TAG = "task.numbered_list.index_successor"
CLAIM_ID = "numbered_list_index_successor.v7"
RESULT_ID = "numbered_list_label_factor.r573.v2.complete.held.v1"
AUDIT_ID = "numbered_list_label_factor_audit.r574.complete.held.v1"


def artifact(path: str, sha256: str, kind: str) -> dict:
    return {"path": path, "sha256": sha256, "kind": kind, "status": "frozen"}


def bind(record: dict, event: dict) -> dict:
    event["design_key"] = design_key(record, event)
    event["execution_key"] = execution_key(record, event)
    return event


def main() -> None:
    path = circuit_path(TAG)
    with _lock("registry"):
        record = json.loads(path.read_text())
        if any(event["event_id"] == AUDIT_ID for event in record["evidence_events"]):
            validate_v2(record)
            print(json.dumps({"status": "already_registered", "claim_id": CLAIM_ID}, indent=2))
            return
        record["artifacts"].update({
            "r573_v2_result": artifact(
                "basis_aligned/bilinear_quotient/numbered_list_factor_localization_rung573_v2_results.json",
                "052930b8b9086e8b7606e3d05929f521f468c04427be8d1182720f1772ee43ec", "result"),
            "r574_audit": artifact(
                "basis_aligned/bilinear_quotient/numbered_list_factor_localization_rung574_audit.json",
                "3d6580ee1a4f1bb77c07e4ee2b404bc23dc70f733db31425bc5da2a11a25a04e", "audit"),
            "r574_audit_script": artifact(
                "basis_aligned/bilinear_quotient/ops/audit_numbered_list_factor_rung574.py",
                "fc5abf51e24d016cab9ca1d690291346f847f7d5ab473ce6cafa4e70955427bc", "implementation"),
        })
        prior = next(claim for claim in record["claims"] if claim["claim_id"] == "numbered_list_index_successor.v6")
        complete = bind(record, {
            "event_id": RESULT_ID, "test_type": "composition", "stage": "complete", "verdict": "held",
            "failure_kind": None,
            "family_ids": [family["family_id"] for family in prior["counterfactual_families"]],
            "site_id": "l8h7_l8h3_value_paths", "evaluation_role": "price_only_replay_FIT_then_conditional_SELECT",
            "metrics": [
                {"name": "complete_two_head_ceiling_minimum_positive_fraction", "estimate": 1.0,
                 "ci95": [3.7861832082271576, None],
                 "bar": ">=0.75 donor-direction rows and bootstrap lower mean effect >0 in every target cell"},
                {"name": "selected_factor_minimum_mean_recovery", "estimate": 0.954336459476474,
                 "ci95": [4.161793579161167, None],
                 "bar": "mean and median cell recovery >=0.5, >=0.75 positive rows, bootstrap lower mean effect >0"},
                {"name": "selected_control_maximum_normalized_effect", "estimate": 0.0, "ci95": None,
                 "bar": "answer-margin and full-vocabulary effects <=0.25 of frozen FIT target scales; answer preserved >=0.75"},
                {"name": "split_and_price", "estimate": 280, "ci95": None,
                 "bar": "SELECT only after FIT selection; <=280 forwards; zero backwards; FINAL_TEST/OOD closed"},
            ],
            "result_artifact_id": "r573_v2_result", "prereg_artifact_id": "r573_v2_amendment",
            "input_artifact_ids": ["r567_rows", "r567_receipt", "r569_r570_result", "r572_result",
                                   "r573_positions", "r573_script", "r573_test", "r573_v1_invalid_receipt",
                                   "r573_v2_script"],
            "split_plan_id": "numbered_list_successor_split_r567_v1", "seed": 573,
            "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            "supersedes_event_id": "numbered_list_label_factor.r573.v2.preregistered.v1",
            "replicates_event_id": None,
            "sections": ["bilinear_quotient/numbered_list_factor_localization_rung573_v2_results.json"],
            "claim_id": "numbered_list_index_successor.v6",
        })
        record["evidence_events"].append(complete)
        claim = deepcopy(prior)
        claim.update({"claim_id": CLAIM_ID, "revision": 7, "status": "activation_identified",
                      "supersedes": "numbered_list_index_successor.v6",
                      "next_missing": "compile the final-label layer-0 cached-value path through L8H7/L8H3 into weights, then test necessity/removal with active unrelated-behavior collateral controls before OOD"})
        claim["evidence_event_ids"] = [*prior["evidence_event_ids"], RESULT_ID, AUDIT_ID]
        site = next(item for item in claim["candidate_sites"] if item["site_id"] == "l8h7_l8h3_value_paths")
        site["ceiling_event_ids"] = [*site["ceiling_event_ids"], RESULT_ID]
        record["claims"].append(claim)
        audit = bind(record, {
            "event_id": AUDIT_ID, "test_type": "null_control", "stage": "complete", "verdict": "held",
            "failure_kind": None,
            "family_ids": [family["family_id"] for family in claim["counterfactual_families"]],
            "site_id": "l8h7_l8h3_value_paths", "evaluation_role": "post_result_independent_CPU_recomputation",
            "metrics": [
                {"name": "raw_cells_recomputed", "estimate": 120, "ci95": None,
                 "bar": "all saved row sets, cell summaries, decisions, hashes, split rules, and price agree"},
                {"name": "minimum_selected_target_effect", "estimate": 3.5816450119018555, "ci95": None,
                 "bar": ">0 on every FIT and SELECT target direction"},
                {"name": "selected_control_intervention_nontriviality", "estimate": 0.0, "ci95": None,
                 "bar": "declared limitation: controls are exact no-ops because final label identity is fixed"},
            ],
            "result_artifact_id": "r574_audit", "prereg_artifact_id": None,
            "input_artifact_ids": ["r573_v2_result", "r574_audit_script", "r567_rows", "r573_positions"],
            "split_plan_id": "numbered_list_successor_split_r567_v1", "seed": 574,
            "checkpoint_sha256": None, "supersedes_event_id": None, "replicates_event_id": None,
            "sections": ["bilinear_quotient/numbered_list_factor_localization_rung574_audit.json"],
            "claim_id": CLAIM_ID,
        })
        record["evidence_events"].append(audit)
        validate_v2(record)
        _atomic_json(path, record)
    rebuild_registry_v2()
    print(json.dumps({"status": "registered", "claim_id": CLAIM_ID,
                      "result_event": RESULT_ID, "audit_event": AUDIT_ID}, indent=2))


if __name__ == "__main__":
    main()
