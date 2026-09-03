#!/usr/bin/env python3
"""Atomically invalidate R573 v1 and register its price-only v2 repair, CPU only."""

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
CLAIM_ID = "numbered_list_index_successor.v6"
INVALID_ID = "numbered_list_label_factor.r573.v1.invalid_price"
V2_ID = "numbered_list_label_factor.r573.v2.preregistered.v1"


def artifact(path: str, sha256: str, kind: str) -> dict:
    return {"path": path, "sha256": sha256, "kind": kind, "status": "frozen"}


def metrics(price: int) -> list[dict]:
    return [
        {"name": "complete_two_head_ceiling", "estimate": None, "ci95": None,
         "bar": ">=0.75 donor-direction rows and bootstrap lower mean effect >0 in every target cell"},
        {"name": "exact_factor_target_recovery", "estimate": None, "ci95": None,
         "bar": "mean and median cell recovery >=0.5, >=0.75 positive rows, bootstrap lower mean effect >0"},
        {"name": "answer_preserving_control_effect", "estimate": None, "ci95": None,
         "bar": "answer-margin and full-vocabulary effects <=0.25 of frozen FIT target scales; answer preserved >=0.75"},
        {"name": "split_and_price", "estimate": None, "ci95": None,
         "bar": f"SELECT only after FIT selection; <={price} forwards; zero backwards; FINAL_TEST/OOD closed"},
    ]


def bind(record: dict, event: dict) -> dict:
    event["design_key"] = design_key(record, event)
    event["execution_key"] = execution_key(record, event)
    return event


def main() -> None:
    path = circuit_path(TAG)
    with _lock("registry"):
        record = json.loads(path.read_text())
        if any(event["event_id"] == V2_ID for event in record["evidence_events"]):
            validate_v2(record)
            print(json.dumps({"status": "already_registered", "claim_id": CLAIM_ID,
                              "event_id": V2_ID}, indent=2))
            return
        record["artifacts"].update({
            "r573_v1_invalid_receipt": artifact(
                "basis_aligned/bilinear_quotient/numbered_list_factor_localization_rung573_v1_invalid_receipt.json",
                "48ae3d2d366b06134851e31585268a0409389ebc7461e7c58c614fa7252ada21", "invalid_receipt"),
            "r573_v2_amendment": artifact(
                "basis_aligned/polynomial_causal/NUMBERED_LIST_FACTOR_LOCALIZATION_RUNG573_V2_IMPLEMENTATION_AMENDMENT.md",
                "27729a0e1405221f989ad0f6b9fef5d2f797c137fbe71a044148e9a5e3e0b4d7", "implementation_amendment"),
            "r573_v2_script": artifact(
                "basis_aligned/bilinear_quotient/ops/numbered_list_factor_localization_rung573_v2.py",
                "a0a4e6b6e654aaff4dcfdcf9ad96f55216014720f7cd2ca4731902b2f5ac071d", "implementation"),
        })
        prior = next(claim for claim in record["claims"] if claim["claim_id"] == "numbered_list_index_successor.v5")
        invalid = bind(record, {
            "event_id": INVALID_ID, "test_type": "composition", "stage": "invalid", "verdict": "invalid",
            "failure_kind": "implementation_failure",
            "family_ids": [family["family_id"] for family in prior["counterfactual_families"]],
            "site_id": "l8h7_l8h3_value_paths", "evaluation_role": "FIT_then_conditional_SELECT_invalid_price",
            "metrics": metrics(278), "result_artifact_id": "r573_v1_invalid_receipt",
            "prereg_artifact_id": "r573_prereg",
            "input_artifact_ids": ["r567_rows", "r567_receipt", "r569_r570_result", "r572_result",
                                   "r573_positions", "r573_script", "r573_test"],
            "split_plan_id": "numbered_list_successor_split_r567_v1", "seed": 573,
            "checkpoint_sha256": None,
            "supersedes_event_id": "numbered_list_label_factor.r573.preregistered.v1",
            "replicates_event_id": None,
            "sections": ["bilinear_quotient/numbered_list_factor_localization_rung573_v1_invalid_receipt.json"],
            "claim_id": "numbered_list_index_successor.v5",
        })
        record["evidence_events"].append(invalid)
        claim = deepcopy(prior)
        claim.update({"claim_id": CLAIM_ID, "revision": 6, "supersedes": "numbered_list_index_successor.v5",
                      "next_missing": "run the registered R573 v2 price-only repair; do not represent it as an independent blind replication"})
        claim["evidence_event_ids"] = [*prior["evidence_event_ids"], INVALID_ID, V2_ID]
        site = next(item for item in claim["candidate_sites"] if item["site_id"] == "l8h7_l8h3_value_paths")
        site["ceiling_event_ids"] = [*site["ceiling_event_ids"], INVALID_ID, V2_ID]
        record["claims"].append(claim)
        v2 = bind(record, {
            "event_id": V2_ID, "test_type": "composition", "stage": "preregistered", "verdict": "inconclusive",
            "failure_kind": None,
            "family_ids": [family["family_id"] for family in claim["counterfactual_families"]],
            "site_id": "l8h7_l8h3_value_paths", "evaluation_role": "price_only_replay_FIT_then_conditional_SELECT",
            "metrics": metrics(280), "result_artifact_id": None,
            "prereg_artifact_id": "r573_v2_amendment",
            "input_artifact_ids": ["r567_rows", "r567_receipt", "r569_r570_result", "r572_result",
                                   "r573_positions", "r573_script", "r573_test", "r573_v1_invalid_receipt",
                                   "r573_v2_script"],
            "split_plan_id": "numbered_list_successor_split_r567_v1", "seed": 573,
            "checkpoint_sha256": None, "supersedes_event_id": INVALID_ID, "replicates_event_id": None,
            "sections": ["polynomial_causal/NUMBERED_LIST_FACTOR_LOCALIZATION_RUNG573_V2_IMPLEMENTATION_AMENDMENT.md"],
            "claim_id": CLAIM_ID,
        })
        record["evidence_events"].append(v2)
        validate_v2(record)
        _atomic_json(path, record)
    rebuild_registry_v2()
    print(json.dumps({"status": "registered", "invalid_event": INVALID_ID,
                      "claim_id": CLAIM_ID, "open_event": V2_ID, "gpu_used": False}, indent=2))


if __name__ == "__main__":
    main()
