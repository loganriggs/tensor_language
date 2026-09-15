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
    "subject_head_response_prereg_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_NATIVE_HEAD_RESPONSE_COORDINATE_DISCOVERY_V1_PREREGISTRATION.md", "d243d2d7642c215981b8f55fc18829547db755f13b44a7f910e2ee3bd008393b", "preregistration"),
    "subject_head_response_runner_v1_failed": ("basis_aligned/bilinear_quotient/ops/run_subject_number_native_head_response_coordinate_discovery_v1.py", "1f6c71b5655945a76de5870c2228c7945c948f38f92e7dfc4bf53cd276dc08d6", "runner"),
    "subject_head_response_correction_v2": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_NATIVE_HEAD_RESPONSE_COORDINATE_DISCOVERY_V2_CORRECTION.md", "31888ba65a24a28c4c369e04a15348f2ee1c39003ca02bbafe0a55f821d73fe6", "correction"),
    "subject_head_response_binding_v2": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_NATIVE_HEAD_RESPONSE_COORDINATE_DISCOVERY_V2_BINDING.json", "2acf93fcdc04bb6ce42f35c55b35f6edfea6ca5213ff49aa4be53df8a852e22f", "binding"),
    "subject_head_response_runner_v2": ("basis_aligned/bilinear_quotient/ops/run_subject_number_native_head_response_coordinate_discovery_v2.py", "f316c2e75ca667548ddf1e31d9cd741d80c88188cdd22310c63a31cddc95b1ac", "runner"),
    "subject_head_response_result_v2": ("basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_native_head_response_coordinate_discovery_v2_result.json", "e0b07656953e5d12a48612f0354e41acb2f01f340ff7868753525d98cbfc0de4", "result"),
}


def metric(name, estimate, bar):
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def build():
    record = json.loads(PATH.read_text())
    old = record["claims"][-1]
    if old["claim_id"] != "grammatical_subject_number.v31":
        raise ValueError("subject authority moved")
    for artifact_id, (relative, expected, kind) in FILES.items():
        if hashlib.sha256((REPO / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("artifact drift " + artifact_id)
        record["artifacts"][artifact_id] = {"path": relative, "sha256": expected,
                                           "kind": kind, "status": "frozen"}
    event_id = "subject_number_native_head_response_coordinate_discovery.v2.complete.held"
    claim = copy.deepcopy(old)
    claim.update(claim_id="grammatical_subject_number.v32", revision=32,
                 supersedes=old["claim_id"], status="weights_translated")
    claim["evidence_event_ids"] = claim["evidence_event_ids"] + [event_id]
    claim["next_missing"] = ("A donor-dependent native L11H3 response coordinate now predicts the frozen direction/cardinality amplitude across constructions only when multiplied by the native head coordinate: joint-interaction relative L2 .37689 versus native baseline .60931, while response-only .44493 and joint-additive .45149 miss. Freeze this scalar interaction and test whether its response coordinate can be predicted donor-free from recipient-side upstream state, then validate on fresh authority before claiming a native generator.")
    record["claims"].append(claim)
    event = {
        "event_id": event_id, "claim_id": claim["claim_id"],
        "test_type": "compiled_equivalence", "stage": "complete", "verdict": "held",
        "failure_kind": None, "family_ids": [],
        "site_id": "L11H3.projected_write.direction_cardinality_rank1_program",
        "split_plan_id": None, "evaluation_role": "opened_construction_program_label_discovery",
        "metrics": [
            metric("head_response_instrument", True, "must be true"),
            metric("native_baseline_cross_construction_relative_l2", .6093072967652493, "report"),
            metric("response_only_cross_construction_relative_l2", .4449342546806798, "<=.40"),
            metric("joint_additive_cross_construction_relative_l2", .4514891840077579, "<=.40"),
            metric("joint_interaction_cross_construction_relative_l2", .37689362716534275, "<=.40"),
            metric("joint_interaction_absolute_improvement", .23241366959990656, ">=.10"),
            metric("joint_interaction_cosine", .9265004678237022, "report"),
            metric("scalar_OLS_fits", 12, "must equal 12"),
        ],
        "prereg_artifact_id": "subject_head_response_prereg_v1",
        "result_artifact_id": "subject_head_response_result_v2",
        "input_artifact_ids": ["subject_head_response_runner_v1_failed",
                               "subject_head_response_correction_v2",
                               "subject_head_response_binding_v2",
                               "subject_head_response_runner_v2"],
        "seed": None, "checkpoint_sha256": CHECKPOINT,
        "supersedes_event_id": None, "replicates_event_id": None,
        "sections": [FILES["subject_head_response_prereg_v1"][0],
                     FILES["subject_head_response_correction_v2"][0]],
        "notes": "Outcome-blind opened-authority discovery on 32 rows and 16 E/A/U/W backgrounds. The coordinate is the native L11H3-axis response to an opposite-number MLP6/7 YZ donor switch. It uses no logits, downstream causal outcomes, gradients, updates, or quantization, but remains donor-dependent and is not prospective native-generation evidence. V1 failed before scientific output because of a dtype-container implementation error; V2 changes only the dtype source."
    }
    event["design_key"] = registry.design_key(record, event)
    event["execution_key"] = registry.execution_key(record, event)
    record["evidence_events"].append(event)
    registry.validate_v2(record)
    return record


def main():
    value = build()
    with registry._lock("registry"):
        if json.loads(PATH.read_text())["claims"][-1]["claim_id"] != "grammatical_subject_number.v31":
            raise ValueError("subject authority moved during publication")
        registry._atomic_json(PATH, value)
    registry.rebuild_registry_v2()
    print(json.dumps({"written": str(PATH.relative_to(REPO)), "gpu_used": False,
                      "claim_id": value["claims"][-1]["claim_id"]}))


if __name__ == "__main__":
    main()
