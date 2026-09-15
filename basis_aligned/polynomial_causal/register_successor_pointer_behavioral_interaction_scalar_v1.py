#!/usr/bin/env python3
"""Register the prefixed parent boundary from the successor scalar experiment."""
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BQ = REPO / "basis_aligned/bilinear_quotient"
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import _atomic_json, _lock, circuit_path, design_key, execution_key, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

TAG = "task.successor_pointer"
OLD = "successor_pointer_state.v7"
NEW = "successor_pointer_state.v8"
EVENT = "successor_pointer_prefixed_parent_boundary.v1.complete.failed"
RESULT = HERE / "SUCCESSOR_POINTER_BEHAVIORAL_INTERACTION_SCALAR_V1_RESULT.json"
ARTIFACTS = {
    "successor_prefixed_length6_builder_v1": ("basis_aligned/polynomial_causal/build_successor_pointer_prefixed_length6_v1_rows.py", "builder"),
    "successor_prefixed_length6_test_v1": ("basis_aligned/polynomial_causal/test_build_successor_pointer_prefixed_length6_v1_rows.py", "test"),
    "successor_prefixed_length6_rows_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_PREFIXED_LENGTH6_V1_ROWS.json", "rows"),
    "successor_behavioral_interaction_scalar_prereg_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_BEHAVIORAL_INTERACTION_SCALAR_V1_PREREGISTRATION.md", "preregistration"),
    "successor_behavioral_interaction_scalar_binding_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_BEHAVIORAL_INTERACTION_SCALAR_V1_BINDING.json", "binding"),
    "successor_behavioral_interaction_scalar_runner_v1": ("basis_aligned/bilinear_quotient/ops/run_successor_pointer_behavioral_interaction_scalar_v1.py", "runner"),
    "successor_behavioral_interaction_scalar_invalid_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_BEHAVIORAL_INTERACTION_SCALAR_V1_RESULT.json", "invalid_result"),
}


def artifact(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main():
    result = json.loads(RESULT.read_text())
    assert result["terminal"] == "invalid_or_dead_parent"
    assert result["predictions"] == {
        "pred_a_exact_instrument_and_live_parent": False,
        "pred_b_suppressive_scalar_selected": False,
        "pred_c_prefixed_scalar_transfer": False,
        "pred_d_scalar_null": False,
    }
    assert result["price"]["observed_forwards"] == 55 and result["price"]["observed_sequences"] == 440
    assert all(value["scalar"]["passes"] for value in result["holdout"].values())

    path = circuit_path(TAG)
    current = json.loads(path.read_text())
    if any(event["event_id"] == EVENT for event in current["evidence_events"]):
        validate_v2(current); rebuild_registry_v2(); print("already registered"); return
    with _lock("registry"):
        record = json.loads(path.read_text())
        for artifact_id, spec in ARTIFACTS.items():
            record["artifacts"][artifact_id] = artifact(*spec)
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD)
        claim = deepcopy(previous)
        claim.update({
            "claim_id": NEW,
            "revision": 8,
            "supersedes": OLD,
            "evidence_event_ids": [*previous["evidence_event_ids"], EVENT],
            "next_missing": "The one-scalar interaction experiment is invalid for its preregistered transfer claim because the exact joint mediator does not recover at least half of the full native month target on any of four prefixed surfaces (projection .411-.472), and two digit early controls exceed .50. The fitted beta=.61575 nevertheless predicts all 32 opened exact joint effects with cosine .99738, relative L2 .07352, and sign 1.0; this is diagnostic only. Do not reinterpret or relax V1. Reopen only on newly frozen surfaces with a parent gate defined on the exact joint causal effect needed by the compression estimand, or move circuits.",
        })
        claim["split_plan_ids"] = claim.get("split_plan_ids", [])
        record["claims"].append(claim)
        minimum_month_projection = min(value["parent_families"]["month"]["projection"] for value in result["holdout"].values())
        maximum_digit_early = max(value["parent_families"]["digit"]["early_backward_control_fraction"] for value in result["holdout"].values())
        event = {
            "event_id": EVENT,
            "claim_id": NEW,
            "test_type": "ood",
            "stage": "complete",
            "verdict": "failed",
            "failure_kind": "scientific_null",
            "family_ids": ["prefix_change_final_pointer_preserved"],
            "site_id": "prefix_conditioned_baseline_branch",
            "split_plan_id": None,
            "evaluation_role": "prospective_prefixed_parent_gate",
            "metrics": [
                {"name": "minimum_prefixed_month_parent_projection", "estimate": minimum_month_projection, "ci95": None, "bar": ">=0.50 in each prefix"},
                {"name": "maximum_prefixed_digit_early_control_fraction", "estimate": maximum_digit_early, "ci95": None, "bar": "<=0.50 in each prefix"},
                {"name": "diagnostic_scalar_holdout_relative_l2", "estimate": result["holdout_overall"]["relative_l2_error"], "ci95": None, "bar": "not opened because parent failed"},
                {"name": "diagnostic_scalar_holdout_cosine", "estimate": result["holdout_overall"]["cosine"], "ci95": None, "bar": "not opened because parent failed"},
            ],
            "prereg_artifact_id": "successor_behavioral_interaction_scalar_prereg_v1",
            "result_artifact_id": "successor_behavioral_interaction_scalar_invalid_v1",
            "input_artifact_ids": list(ARTIFACTS),
            "seed": None,
            "checkpoint_sha256": result["checkpoint_sha256"],
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["basis_aligned/polynomial_causal/SUCCESSOR_POINTER_BEHAVIORAL_INTERACTION_SCALAR_V1_PREREGISTRATION.md"],
            "notes": "No scalar transfer verdict is registered. V1 exactness, capability, and price passed, but the immutable parent projection/control gate failed before scalar interpretation.",
        }
        event["design_key"] = design_key(record, event)
        event["execution_key"] = execution_key(record, event)
        record["evidence_events"].append(event)
        validate_v2(record); _atomic_json(path, record)
    rebuild_registry_v2(); validate_v2(json.loads(path.read_text()))
    print(json.dumps({"status": "registered", "claim_id": NEW, "event_id": EVENT}, indent=2))


if __name__ == "__main__":
    main()
