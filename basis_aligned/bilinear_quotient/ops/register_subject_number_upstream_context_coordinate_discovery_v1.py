#!/usr/bin/env python3
"""Register the audited upstream context-coordinate null in the subject circuit."""
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
    "subject_upstream_context_prereg_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_UPSTREAM_CONTEXT_COORDINATE_DISCOVERY_V1_PREREGISTRATION.md", "0f5636b4ed1b4877a02a6b186675e4eb346b1cff696792fdddd75e7a93e14a87", "preregistration"),
    "subject_upstream_context_binding_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_UPSTREAM_CONTEXT_COORDINATE_DISCOVERY_V1_BINDING.json", "f47105cb6f14c4913dcdc93de0a0f624a23be41c289f7e6161a72ff203a4cfe4", "binding"),
    "subject_upstream_context_runner_v1": ("basis_aligned/bilinear_quotient/ops/run_subject_number_upstream_context_coordinate_discovery_v1.py", "86d34828e96e62810525ae79f838c7996ce81532850b1af73a0bb2eff188c79e", "runner"),
    "subject_upstream_context_result_v1": ("basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_upstream_context_coordinate_discovery_v1_result.json", "573f38d7bcbf2a0e82d4987c62100ce65b42e8564b818f8b1ed166d199711a53", "result"),
}


def metric(name, estimate, bar):
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def build():
    record = json.loads(PATH.read_text())
    old = record["claims"][-1]
    if old["claim_id"] != "grammatical_subject_number.v30":
        raise ValueError("subject authority moved")
    for artifact_id, (relative, expected, kind) in FILES.items():
        if hashlib.sha256((REPO / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("artifact drift " + artifact_id)
        record["artifacts"][artifact_id] = {"path": relative, "sha256": expected,
                                               "kind": kind, "status": "frozen"}
    event_id = "subject_number_upstream_context_coordinate_discovery.v1.complete.null"
    claim = copy.deepcopy(old)
    claim.update(claim_id="grammatical_subject_number.v31", revision=31,
                 supersedes=old["claim_id"], status="weights_translated")
    claim["evidence_event_ids"] = claim["evidence_event_ids"] + [event_id]
    claim["next_missing"] = ("The native L11H3 axis coordinate plus the MLP8-input E/A/U/W secant coordinate does not predict the frozen direction/cardinality amplitude across constructions: bilinear relative L2 .55313 versus affine .61700, only .06387 improvement, and quadratic-context .54989. Do not widen the base-head singular or this secant-coordinate family. Next test an independently defined upstream causal-response or predictive-state coordinate, without fitting behavioral outcomes or using quantization.")
    record["claims"].append(claim)
    event = {
        "event_id": event_id, "claim_id": claim["claim_id"],
        "test_type": "compiled_equivalence", "stage": "complete", "verdict": "null",
        "failure_kind": "scientific_null", "family_ids": [],
        "site_id": "L11H3.projected_write.direction_cardinality_rank1_program",
        "split_plan_id": None, "evaluation_role": "opened_construction_program_label_discovery",
        "metrics": [
            metric("activation_instrument", True, "must be true"),
            metric("affine_cross_construction_relative_l2", .6169963100737825, "report"),
            metric("bilinear_cross_construction_relative_l2", .5531257055717395, "<=.40"),
            metric("bilinear_absolute_improvement", .06387060450204307, ">=.10"),
            metric("quadratic_context_cross_construction_relative_l2", .5498922482027531, "<=.40 fallback"),
            metric("scalar_OLS_fits", 9, "must equal 9"),
        ],
        "prereg_artifact_id": "subject_upstream_context_prereg_v1",
        "result_artifact_id": "subject_upstream_context_result_v1",
        "input_artifact_ids": ["subject_upstream_context_binding_v1", "subject_upstream_context_runner_v1"],
        "seed": None, "checkpoint_sha256": CHECKPOINT,
        "supersedes_event_id": None, "replicates_event_id": None,
        "sections": [FILES["subject_upstream_context_prereg_v1"][0]],
        "notes": "Outcome-blind program-label discovery on the opened 32-row authority and all 16 E/A/U/W subsets; one forward over 96 role sequences. No answer logits, behavioral effects, exact donor displacements, fresh authority, gradients, updates, or quantization."
    }
    event["design_key"] = registry.design_key(record, event)
    event["execution_key"] = registry.execution_key(record, event)
    record["evidence_events"].append(event)
    registry.validate_v2(record)
    return record


def main():
    value = build()
    with registry._lock("registry"):
        if json.loads(PATH.read_text())["claims"][-1]["claim_id"] != "grammatical_subject_number.v30":
            raise ValueError("subject authority moved during publication")
        registry._atomic_json(PATH, value)
    registry.rebuild_registry_v2()
    print(json.dumps({"written": str(PATH.relative_to(REPO)), "gpu_used": False,
                      "claim_id": value["claims"][-1]["claim_id"]}))


if __name__ == "__main__":
    main()
