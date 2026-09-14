#!/usr/bin/env python3
"""Register the fresh A1/A2 native-capability null in narrative authority."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import circuit_registry_v2 as registry


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
PATH = BQ / "circuits/task_narrative_tense_past_vs_present.json"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
ARTIFACTS = {
    "newlex_a1_a2_capability_authority_v1": (
        "basis_aligned/bilinear_quotient/ops/circuit_fast_screen_candidate_narrative_tense_newlex_a1_a2_authority.py",
        "5128fe4277fe366a4b2a48b338f7fe9cfc9808236a9402bafde2c37c985d1b41", "dataset_authority"),
    "newlex_a1_a2_capability_prereg_v1": (
        "basis_aligned/bilinear_quotient/circuits/prior_art/NARRATIVE_TENSE_NEWLEX_A1_A2_NATIVE_CAPABILITY_V1_PREREGISTRATION.md",
        "39a4a69c66b24db1a3a302df26d53a52b5e7410fb169cca0ef6d4c0d28a1368d", "preregistration"),
    "newlex_a1_a2_capability_result_v1": (
        "basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_newlex_a1_a2_native_capability_v1_result.json",
        "c776b4732969820dd4dc1e28f1264aae352adb6c39c714a069b55981a32b7a39", "screen_result"),
}


def metric(name, estimate, bar):
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def build() -> dict:
    record = json.loads(PATH.read_text())
    if record["claims"][-1]["claim_id"] != "narrative_tense_at_final_position.v6":
        raise ValueError("narrative authority moved")
    for key, (relative, expected, kind) in ARTIFACTS.items():
        actual = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"artifact drift: {key}")
        record["artifacts"][key] = {"path": relative, "sha256": actual,
                                    "kind": kind, "status": "frozen"}
    claim = copy.deepcopy(record["claims"][-1])
    claim.update({"claim_id": "narrative_tense_at_final_position.v7", "revision": 7,
                  "supersedes": "narrative_tense_at_final_position.v6"})
    event_id = "narrative_tense.newlex_a1_a2_native_capability.v1.complete.null"
    claim["evidence_event_ids"] = claim["evidence_event_ids"] + [event_id]
    claim["next_missing"] = (
        "fresh direct A1 is capable, but the frozen reported A2 is present-tense incapable; "
        "only a separately frozen construction with an explicit answer-clause time anchor may "
        "attempt native capability before causal work")
    record["claims"].append(claim)
    event = {
        "event_id": event_id, "test_type": "capability", "stage": "complete",
        "verdict": "null", "failure_kind": "scientific_null", "site_id": None,
        "result_artifact_id": "newlex_a1_a2_capability_result_v1",
        "prereg_artifact_id": "newlex_a1_a2_capability_prereg_v1",
        "metrics": [
            metric("minimum_FIT_A1_cell_accuracy", 1.0, ">=0.875"),
            metric("minimum_FIT_A2_cell_accuracy", 0.0, ">=0.875"),
            metric("minimum_FIT_P_cell_accuracy", 0.75, ">=0.875"),
            metric("minimum_FIT_C_cell_accuracy", 0.5, ">=0.875"),
            metric("holdout_opened", 0, "must equal 1 only after every FIT cell passes"),
            metric("model_forwards", 1, "must equal 1 on FIT failure"),
            metric("causal_interventions", 0, "must equal 0"),
            metric("license_emitted", 0, "must equal 1 before causal work"),
        ],
        "supersedes_event_id": None,
        "notes": {"scientific_status": "valid frozen native-capability null",
                  "localization": "A1 passed 4/4 in every cell; A2 answered was for all present endpoints",
                  "holdout": "sealed by the preregistered FIT gate",
                  "licensed_action": "new independently frozen answer-clause time-anchor construction only"},
        "claim_id": claim["claim_id"],
        "family_ids": ["a1_direct_narration", "a2_relative_clause",
                       "p_surface_rewrite", "c_same_answer_rewrite"],
        "evaluation_role": "frozen FIT screen",
        "input_artifact_ids": ["newlex_a1_a2_capability_authority_v1"],
        "split_plan_id": "narrative_tense_fit_v1", "seed": None,
        "checkpoint_sha256": CHECKPOINT, "replicates_event_id": None, "sections": [],
    }
    event["design_key"] = registry.design_key(record, event)
    event["execution_key"] = registry.execution_key(record, event)
    record["evidence_events"].append(event)
    registry.validate_v2(record)
    return record


def main():
    value = build()
    with registry._lock("registry"):
        if json.loads(PATH.read_text())["claims"][-1]["claim_id"] != "narrative_tense_at_final_position.v6":
            raise ValueError("narrative authority moved during publication")
        registry._atomic_json(PATH, value)
    registry.rebuild_registry_v2()
    print(json.dumps({"written": str(PATH.relative_to(REPO)), "gpu_used": False}))


if __name__ == "__main__":
    main()
