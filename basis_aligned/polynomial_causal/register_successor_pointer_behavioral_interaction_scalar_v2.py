#!/usr/bin/env python3
"""Register the second prefixed interaction-parent boundary."""
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
OLD = "successor_pointer_state.v8"
NEW = "successor_pointer_state.v9"
EVENT = "successor_pointer_second_prefixed_interaction_boundary.v2.complete.failed"
RESULT = HERE / "SUCCESSOR_POINTER_BEHAVIORAL_INTERACTION_SCALAR_V2_RESULT.json"
ARTIFACTS = {
    "successor_prefixed_length6_builder_v2": ("basis_aligned/polynomial_causal/build_successor_pointer_prefixed_length6_v2_rows.py", "builder"),
    "successor_prefixed_length6_test_v2": ("basis_aligned/polynomial_causal/test_build_successor_pointer_prefixed_length6_v2_rows.py", "test"),
    "successor_prefixed_length6_rows_v2": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_PREFIXED_LENGTH6_V2_ROWS.json", "rows"),
    "successor_behavioral_interaction_scalar_prereg_v2": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_BEHAVIORAL_INTERACTION_SCALAR_V2_PREREGISTRATION.md", "preregistration"),
    "successor_behavioral_interaction_scalar_binding_v2": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_BEHAVIORAL_INTERACTION_SCALAR_V2_BINDING.json", "binding"),
    "successor_behavioral_interaction_scalar_runner_v2": ("basis_aligned/bilinear_quotient/ops/run_successor_pointer_behavioral_interaction_scalar_v2.py", "runner"),
    "successor_behavioral_interaction_scalar_invalid_v2": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_BEHAVIORAL_INTERACTION_SCALAR_V2_RESULT.json", "invalid_result"),
}


def artifact(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main():
    result = json.loads(RESULT.read_text())
    assert result["terminal"] == "invalid_or_dead_interaction"
    assert result["predictions"] == {
        "pred_a_exact_instrument_and_interaction_live": False,
        "pred_b_frozen_suppressive_scalar": False,
        "pred_c_second_prefixed_scalar_transfer": False,
        "pred_d_scalar_null": False,
    }
    assert result["price"]["observed_forwards"] == 44 and result["price"]["observed_sequences"] == 352
    assert all(panel["scalar"]["passes"] for panel in result["holdout"].values())
    failures = [
        (prefix, family, report)
        for prefix, panel in result["holdout"].items()
        for family, report in panel["parent_families"].items()
        if not report["passes"]
    ]
    assert len(failures) == 1 and failures[0][:2] == ("next", "digit")

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
            "revision": 9,
            "supersedes": OLD,
            "evidence_event_ids": [*previous["evidence_event_ids"], EVENT],
            "next_missing": "A second prospective prefix authority is invalid for confirming the frozen beta=.61575 interaction adapter because the exact joint early-control ratio is .76953 in the Next-digit cell against the frozen .75 bar. All other parent cells and all eight scalar cells pass; diagnostic scalar fidelity is cosine .99723, relative L2 .07541, norm ratio 1.00967, sign 1.0. Two consecutive prefix authorities therefore cannot support a registered transfer verdict despite strong conditional predictions. Do not tune controls, add a third prefix panel, refit beta, or quantize. Move to another unresolved circuit unless an independently derived component generator changes the estimand.",
        })
        record["claims"].append(claim)
        failed = result["holdout"]["next"]["parent_families"]["digit"]
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
            "evaluation_role": "second_prospective_prefixed_interaction_parent_gate",
            "metrics": [
                {"name": "next_digit_exact_early_control_fraction", "estimate": failed["early_backward_control_fraction"], "ci95": None, "bar": "<=0.75"},
                {"name": "next_digit_interaction_to_joint_rms", "estimate": failed["interaction_to_joint_rms"], "ci95": None, "bar": ">=0.20"},
                {"name": "diagnostic_scalar_holdout_relative_l2", "estimate": result["holdout_overall"]["relative_l2_error"], "ci95": None, "bar": "not opened because parent failed"},
                {"name": "diagnostic_scalar_holdout_cosine", "estimate": result["holdout_overall"]["cosine"], "ci95": None, "bar": "not opened because parent failed"},
            ],
            "prereg_artifact_id": "successor_behavioral_interaction_scalar_prereg_v2",
            "result_artifact_id": "successor_behavioral_interaction_scalar_invalid_v2",
            "input_artifact_ids": list(ARTIFACTS),
            "seed": None,
            "checkpoint_sha256": result["checkpoint_sha256"],
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["basis_aligned/polynomial_causal/SUCCESSOR_POINTER_BEHAVIORAL_INTERACTION_SCALAR_V2_PREREGISTRATION.md"],
            "notes": "No scalar transfer verdict is registered. Seven of eight interaction-parent cells and every scalar-fidelity cell passed; the one frozen exact-control bar failed.",
        }
        event["design_key"] = design_key(record, event)
        event["execution_key"] = execution_key(record, event)
        record["evidence_events"].append(event)
        validate_v2(record); _atomic_json(path, record)
    rebuild_registry_v2(); validate_v2(json.loads(path.read_text()))
    print(json.dumps({"status": "registered", "claim_id": NEW, "event_id": EVENT}, indent=2))


if __name__ == "__main__":
    main()
