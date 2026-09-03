#!/usr/bin/env python3
"""Register R564 before its first model outcome."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import append_artifacts, append_evidence_event, circuit_path, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

TAG = "task.increment.state"
EVENT = "increment_native_capability.r564.preregistered.v1"
ARTIFACTS = {
    "r564_prereg": ("basis_aligned/polynomial_causal/INCREMENT_NATIVE_CAPABILITY_RUNG564_PREREGISTRATION.md", "preregistration"),
    "r564_script": ("basis_aligned/bilinear_quotient/ops/increment_native_capability_rung564.py", "implementation"),
}


def main() -> None:
    artifacts = {
        key: {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}
        for key, (path, kind) in ARTIFACTS.items()
    }
    append_artifacts(TAG, artifacts)
    record = json.loads(circuit_path(TAG).read_text())
    if not any(event["event_id"] == EVENT for event in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": EVENT,
            "test_type": "capability",
            "stage": "preregistered",
            "verdict": "inconclusive",
            "failure_kind": None,
            "family_ids": [
                "digit_coherent_shift", "word_coherent_shift", "cross_format_coherent_shift",
                "incoherent_middle_number_edit", "operation_preserved_surface_edit",
                "repeated_number_numeric_control", "step_two_numeric_control",
            ],
            "site_id": None,
            "evaluation_role": "FIT_then_conditional_SELECT",
            "metrics": [
                {"name": "numeric_candidate_accuracy_and_margin", "estimate": None,
                 "ci95": None, "bar": ">=0.75 correct and bootstrap lower mean margin >0 in every endpoint cell"},
                {"name": "middle_number_necessity", "estimate": None,
                 "ci95": None, "bar": ">=0.65 positive drops and bootstrap lower mean drop >0"},
                {"name": "split_opening", "estimate": None,
                 "ci95": None, "bar": "SELECT only after all FIT cells pass; FINAL_TEST/OOD closed"},
            ],
            "result_artifact_id": None,
            "prereg_artifact_id": "r564_prereg",
            "input_artifact_ids": ["r563_rows", "r563_receipt", "r563_correction", "r564_script"],
            "split_plan_id": "increment_counterfactual_split_r563_v1",
            "seed": 564,
            "checkpoint_sha256": None,
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["polynomial_causal/INCREMENT_NATIVE_CAPABILITY_RUNG564_PREREGISTRATION.md"],
            "claim_id": "increment_state.v3",
        })
    final = json.loads(circuit_path(TAG).read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({"tag": TAG, "event_id": EVENT, "stage": "preregistered"}, indent=2))


if __name__ == "__main__":
    main()
