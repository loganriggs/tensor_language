#!/usr/bin/env python3
"""Register the audited native L11H3 response-coordinate discovery result."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
import circuit_registry_v2 as registry


PATH = BQ / "circuits/task_subject_verb_number_agreement.json"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
FILES = {
    "subject_response_proxy_prereg_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_DONOR_FREE_HEAD_RESPONSE_PROXY_V1_PREREGISTRATION.md", "854e8941cafed428163eadd70f6e098dac2b850347b71efdeae8ba4ca49c3871", "preregistration"),
    "subject_response_proxy_runner_v1": ("basis_aligned/bilinear_quotient/ops/run_subject_number_donor_free_head_response_proxy_v1.py", "2e41c0745f90499678a47f38252ade2506c191cfec4f21e0d20c4df155e677ad", "runner"),
    "subject_response_proxy_invalid_v1": ("basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_donor_free_head_response_proxy_v1_result.json", "adc7f7c4afc47ac2759f11a717440e4006b97f1f6d3fa845b9438da723efd850", "invalid_result"),
    "subject_response_proxy_correction_v2": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_DONOR_FREE_HEAD_RESPONSE_PROXY_V2_CORRECTION.md", "6ee2533d1e764221f1ac6fafe5204f08179115ac9013a89b1607008aabbae9ec", "correction"),
    "subject_response_proxy_binding_v2": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_DONOR_FREE_HEAD_RESPONSE_PROXY_V2_BINDING.json", "f768c2519106c55f1030cff9986600b6e8c7542e998490702f1d03f91fc0fdf9", "binding"),
    "subject_response_proxy_runner_v2": ("basis_aligned/bilinear_quotient/ops/run_subject_number_donor_free_head_response_proxy_v2.py", "3673726941b3e3ece759611b23a5dff85f70ec2d5f0234c5581374b008374ab5", "runner"),
    "subject_response_proxy_result_v2": ("basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_donor_free_head_response_proxy_v2_result.json", "73cb7bdb4f981cf3d20e5f7e7f2810341179643919b350cf105be92384844edb", "result"),
}


def metric(name, estimate, bar):
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def build():
    record = json.loads(PATH.read_text())
    old = record["claims"][-1]
    if old["claim_id"] != "grammatical_subject_number.v32":
        raise ValueError("subject authority moved")
    for artifact_id, (relative, expected, kind) in FILES.items():
        if hashlib.sha256((REPO / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("artifact drift " + artifact_id)
        record["artifacts"][artifact_id] = {"path": relative, "sha256": expected,
                                           "kind": kind, "status": "frozen"}
    event_id = "subject_number_donor_free_head_response_proxy.v2.complete.null"
    claim = copy.deepcopy(old)
    claim.update(claim_id="grammatical_subject_number.v33", revision=33,
                 supersedes=old["claim_id"], status="weights_translated")
    claim["evidence_event_ids"] = claim["evidence_event_ids"] + [event_id]
    claim["next_missing"] = ("A two-vector cross-construction MLP6/7 prototype predicts the exact L11H3 response scalar well (cosine .96590, relative L2 .28804), but errors amplify through the required native-coordinate interaction: coefficient relative L2 .51585, only .09346 better than the .60931 native baseline and .13896 worse than the .37689 exact-response oracle. Do not expand direction prototypes into lexical/background tables. Next test a vector-valued recipient-side predictive state, then require fresh-authority causal substitution.")
    record["claims"].append(claim)
    event = {
        "event_id": event_id, "claim_id": claim["claim_id"],
        "test_type": "cross_family_transfer", "stage": "complete", "verdict": "null",
        "failure_kind": "scientific_null", "family_ids": [],
        "site_id": "L11H3.projected_write.direction_cardinality_rank1_program",
        "split_plan_id": None, "evaluation_role": "opened_cross_construction_donor_free_proxy",
        "metrics": [
            metric("exact_response_metric_replay_error", 3.236673262740908e-09, "<=1e-8"),
            metric("response_proxy_cosine", .9658983061002184, ">=.85"),
            metric("response_proxy_relative_l2", .2880428105400366, "<=.50"),
            metric("proxy_program_relative_l2", .5158500295544283, "<=.45"),
            metric("proxy_program_improvement_over_native", .09345726721082104, ">=.10"),
            metric("proxy_program_degradation_from_oracle", .13895640238908552, "<=.10"),
            metric("offline_head_function_row_evaluations", 1536, "must equal 1536"),
        ],
        "prereg_artifact_id": "subject_response_proxy_prereg_v1",
        "result_artifact_id": "subject_response_proxy_result_v2",
        "input_artifact_ids": ["subject_response_proxy_runner_v1", "subject_response_proxy_invalid_v1",
                               "subject_response_proxy_correction_v2", "subject_response_proxy_binding_v2",
                               "subject_response_proxy_runner_v2"],
        "seed": None, "checkpoint_sha256": CHECKPOINT,
        "supersedes_event_id": None, "replicates_event_id": None,
        "sections": [FILES["subject_response_proxy_prereg_v1"][0],
                     FILES["subject_response_proxy_correction_v2"][0]],
        "notes": "Outcome-blind opened-authority cross-construction test with four fold-specific 1,152D direction prototypes and no fits. The held-out proxy uses no row-specific donor, logits, downstream outcomes, gradients, updates, or quantization. V1 was implementation-invalid only because a 3.24e-9 aggregate replay difference missed a 1e-10 audit; V2 changes that audit to 1e-8, still far below the 5e-5 model-closure tolerance. The response scalar passes, but the preregistered complete program fails all three coefficient gates."
    }
    event["design_key"] = registry.design_key(record, event)
    event["execution_key"] = registry.execution_key(record, event)
    record["evidence_events"].append(event)
    registry.validate_v2(record)
    return record


def main():
    value = build()
    with registry._lock("registry"):
        if json.loads(PATH.read_text())["claims"][-1]["claim_id"] != "grammatical_subject_number.v32":
            raise ValueError("subject authority moved during publication")
        registry._atomic_json(PATH, value)
    registry.rebuild_registry_v2()
    print(json.dumps({"written": str(PATH.relative_to(REPO)), "gpu_used": False,
                      "claim_id": value["claims"][-1]["claim_id"]}))


if __name__ == "__main__":
    main()
