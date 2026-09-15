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
    "subject_rank2_state_prereg_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_RANK2_RECIPIENT_STATE_RESPONSE_PROXY_V1_PREREGISTRATION.md", "44fec4c52310f7e04ba47294ec98762f4cf3fb7f177f870cc6a0854abddefbdb", "preregistration"),
    "subject_rank2_state_runner_v1_failed": ("basis_aligned/bilinear_quotient/ops/run_subject_number_rank2_recipient_state_response_proxy_v1.py", "ac6ddb40f654983b3192207ab090643e263bad8e633bef09e530521172c08a7a", "runner"),
    "subject_rank2_state_correction_v2": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_RANK2_RECIPIENT_STATE_RESPONSE_PROXY_V2_CORRECTION.md", "bec296e50312c53e791718952c87449a5f3beef5dd621a19255d9bbce61b1452", "correction"),
    "subject_rank2_state_binding_v2": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_RANK2_RECIPIENT_STATE_RESPONSE_PROXY_V2_BINDING.json", "449dbace1ba767e60f78b432548cb5ae0ed285a14735605deafcf1bc509c9966", "binding"),
    "subject_rank2_state_runner_v2": ("basis_aligned/bilinear_quotient/ops/run_subject_number_rank2_recipient_state_response_proxy_v2.py", "d69492632e22a28b819e576121343053eeccd5b732230f136140540eaeb18f79", "runner"),
    "subject_rank2_state_result_v2": ("basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_rank2_recipient_state_response_proxy_v2_result.json", "387846818d7e74d62eccba803349c67595d2ebc5bf3b8cf55a0e7572fc117141", "result"),
}


def metric(name, estimate, bar):
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def build():
    record = json.loads(PATH.read_text())
    old = record["claims"][-1]
    if old["claim_id"] != "grammatical_subject_number.v33":
        raise ValueError("subject authority moved")
    for artifact_id, (relative, expected, kind) in FILES.items():
        if hashlib.sha256((REPO / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("artifact drift " + artifact_id)
        record["artifacts"][artifact_id] = {"path": relative, "sha256": expected, "kind": kind, "status": "frozen"}
    event_id = "subject_number_rank2_recipient_state_response_proxy.v2.complete.null"
    claim = copy.deepcopy(old)
    claim.update(claim_id="grammatical_subject_number.v34", revision=34, supersedes=old["claim_id"], status="weights_translated")
    claim["evidence_event_ids"] = claim["evidence_event_ids"] + [event_id]
    claim["next_missing"] = ("Rank-2 activation/displacement PCA does not improve the donor-free L11H3 response proxy: response relative L2 .28722 versus .28804 for the mean vector, only .00083 gain, and coefficient error .51072 still misses all complete-program gates. Do not sweep activation rank on this opened authority. Next subject-number circuit hour should define a response-oriented predictive basis using the frozen head-response operator, then require fresh-authority causal substitution if it passes.")
    record["claims"].append(claim)
    event = {
        "event_id": event_id, "claim_id": claim["claim_id"], "test_type": "cross_family_transfer",
        "stage": "complete", "verdict": "null", "failure_kind": "scientific_null", "family_ids": [],
        "site_id": "L11H3.projected_write.direction_cardinality_rank1_program", "split_plan_id": None,
        "evaluation_role": "opened_cross_construction_rank2_recipient_state_proxy",
        "metrics": [
            metric("parent_replay", True, "must be true"),
            metric("mean_response_relative_l2", .2880428105400366, "reference"),
            metric("rank2_response_relative_l2", .2872163287046907, "improve by >=.05"),
            metric("rank2_response_absolute_improvement", .0008264818353458936, ">=.05"),
            metric("rank2_program_relative_l2", .5107208113072614, "<=.45"),
            metric("rank2_program_improvement_over_native", .09858648545798787, ">=.10"),
            metric("rank2_program_degradation_from_oracle", .1338271841419187, "<=.10"),
        ],
        "prereg_artifact_id": "subject_rank2_state_prereg_v1", "result_artifact_id": "subject_rank2_state_result_v2",
        "input_artifact_ids": ["subject_rank2_state_runner_v1_failed", "subject_rank2_state_correction_v2",
                               "subject_rank2_state_binding_v2", "subject_rank2_state_runner_v2"],
        "seed": None, "checkpoint_sha256": CHECKPOINT, "supersedes_event_id": None, "replicates_event_id": None,
        "sections": [FILES["subject_rank2_state_prereg_v1"][0], FILES["subject_rank2_state_correction_v2"][0]],
        "notes": "Outcome-blind opened-authority rank-2 recipient-state/displacement prediction with four fold-direction maps. No coefficient refit, logits, downstream outcomes, held-out row donor, gradients, updates, or quantization. V1 failed before scientific computation due an output-array initialization typo; V2 changes only that typo. Top-two activation and displacement PCs retain about .37-.43 and .49-.54 of training energy but add negligible response fidelity."
    }
    event["design_key"] = registry.design_key(record, event); event["execution_key"] = registry.execution_key(record, event)
    record["evidence_events"].append(event); registry.validate_v2(record); return record


def main():
    value = build()
    with registry._lock("registry"):
        if json.loads(PATH.read_text())["claims"][-1]["claim_id"] != "grammatical_subject_number.v33":
            raise ValueError("subject authority moved during publication")
        registry._atomic_json(PATH, value)
    registry.rebuild_registry_v2()
    print(json.dumps({"written": str(PATH.relative_to(REPO)), "gpu_used": False,
                      "claim_id": value["claims"][-1]["claim_id"]}))


if __name__ == "__main__":
    main()
