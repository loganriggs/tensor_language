#!/usr/bin/env python3
"""Atomically register the frozen R573 claim revision and open event, CPU only."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    _atomic_json,
    _lock,
    circuit_path,
    design_key,
    execution_key,
    rebuild_registry_v2,
    validate_v2,
)


TAG = "task.numbered_list.index_successor"
CLAIM_ID = "numbered_list_index_successor.v5"
EVENT_ID = "numbered_list_label_factor.r573.preregistered.v1"


def artifact(path: str, sha256: str, kind: str) -> dict:
    return {"path": path, "sha256": sha256, "kind": kind, "status": "frozen"}


def main() -> None:
    path = circuit_path(TAG)
    with _lock("registry"):
        record = json.loads(path.read_text())
        if any(claim["claim_id"] == CLAIM_ID for claim in record["claims"]):
            assert any(event["event_id"] == EVENT_ID for event in record["evidence_events"])
            validate_v2(record)
            print(json.dumps({"status": "already_registered", "claim_id": CLAIM_ID,
                              "event_id": EVENT_ID}, indent=2))
            return
        additions = {
            "r573_positions": artifact(
                "basis_aligned/bilinear_quotient/numbered_list_semantic_positions_rung573.json",
                "b4f4f8e58c03deb3a015141656572dcf1fd0fe0c12027c358b441459c245c16b", "semantic_audit"),
            "r573_prereg": artifact(
                "basis_aligned/polynomial_causal/NUMBERED_LIST_FACTOR_LOCALIZATION_RUNG573_PREREGISTRATION.md",
                "12e06f11a0865396a85fb3b58c6b2fa23bb90ea28f80d94de88796a0f13d9365", "preregistration"),
            "r573_script": artifact(
                "basis_aligned/bilinear_quotient/ops/numbered_list_factor_localization_rung573.py",
                "5723e42e2a5f72a4ddab7a20b631e18e0b6d28875ff53f3db2d37d1845d6e076", "implementation"),
            "r573_test": artifact(
                "basis_aligned/bilinear_quotient/ops/test_numbered_list_factor_localization_rung573.py",
                "cb7a4c4d768d8ca1046edfce3d3c39e66d7fe0c21c0cd57916d832c567ab18d6", "test"),
        }
        record["artifacts"].update(additions)
        prior = next(claim for claim in record["claims"]
                     if claim["claim_id"] == "numbered_list_index_successor.v4")
        claim = deepcopy(prior)
        claim.update({"claim_id": CLAIM_ID, "revision": 5,
                      "supersedes": "numbered_list_index_successor.v4",
                      "next_missing": "run the open R573 exact L8H7/L8H3 label-factor FIT gate; only a passing frozen arm may open SELECT"})
        claim["evidence_event_ids"] = [*prior["evidence_event_ids"], EVENT_ID]
        site = next(item for item in claim["candidate_sites"]
                    if item["site_id"] == "l8h7_l8h3_value_paths")
        site["ceiling_event_ids"] = [*site["ceiling_event_ids"], EVENT_ID]
        record["claims"].append(claim)
        event = {
            "event_id": EVENT_ID,
            "test_type": "composition",
            "stage": "preregistered",
            "verdict": "inconclusive",
            "failure_kind": None,
            "family_ids": list(r573_families(claim)),
            "site_id": "l8h7_l8h3_value_paths",
            "evaluation_role": "FIT_selects_one_exact_factor_then_conditional_SELECT",
            "metrics": [
                {"name": "complete_two_head_ceiling", "estimate": None, "ci95": None,
                 "bar": ">=0.75 donor-direction rows and bootstrap lower mean effect >0 in every target cell"},
                {"name": "exact_factor_target_recovery", "estimate": None, "ci95": None,
                 "bar": "mean and median cell recovery >=0.5, >=0.75 positive rows, bootstrap lower mean effect >0"},
                {"name": "answer_preserving_control_effect", "estimate": None, "ci95": None,
                 "bar": "answer-margin and full-vocabulary effects <=0.25 of frozen FIT target scales; answer preserved >=0.75"},
                {"name": "split_and_price", "estimate": None, "ci95": None,
                 "bar": "SELECT only after FIT selection; <=278 forwards; zero backwards; FINAL_TEST/OOD closed"},
            ],
            "result_artifact_id": None,
            "prereg_artifact_id": "r573_prereg",
            "input_artifact_ids": ["r567_rows", "r567_receipt", "r569_r570_result", "r572_result",
                                   "r573_positions", "r573_script", "r573_test"],
            "split_plan_id": "numbered_list_successor_split_r567_v1",
            "seed": 573,
            "checkpoint_sha256": None,
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["polynomial_causal/NUMBERED_LIST_FACTOR_LOCALIZATION_RUNG573_PREREGISTRATION.md"],
            "claim_id": CLAIM_ID,
        }
        event["design_key"] = design_key(record, event)
        event["execution_key"] = execution_key(record, event)
        record["evidence_events"].append(event)
        validate_v2(record)
        _atomic_json(path, record)
    rebuild_registry_v2()
    print(json.dumps({"status": "registered", "claim_id": CLAIM_ID,
                      "event_id": EVENT_ID, "gpu_used": False}, indent=2))


def r573_families(claim: dict):
    wanted = set(("list_two_line_state_shift", "list_three_line_state_shift",
                  "list_surface_preserved", "list_middle_index_break",
                  "list_repeated_index_control", "list_step_two_conflict"))
    observed = [family["family_id"] for family in claim["counterfactual_families"]
                if family["family_id"] in wanted]
    assert set(observed) == wanted
    return observed


if __name__ == "__main__":
    main()
