#!/usr/bin/env python3
"""Register the successor pointer-depth cross-type interaction."""
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BQ = REPO / "basis_aligned/bilinear_quotient"
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    _atomic_json, _lock, circuit_path, design_key, execution_key,
    file_sha256, rebuild_registry_v2, validate_v2,
)

TAG = "task.successor_pointer"
OLD_CLAIM = "successor_pointer_state.v5"
NEW_CLAIM = "successor_pointer_state.v6"
EVENT_ID = "successor_pointer_cross_type_interaction.v1.complete.held"
ARTIFACTS = {
    "successor_cross_type_prereg_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_CROSS_TYPE_INTERACTION_V1_PREREGISTRATION.md", "preregistration"),
    "successor_cross_type_binding_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_CROSS_TYPE_INTERACTION_V1_BINDING.json", "binding"),
    "successor_cross_type_runner_v1": ("basis_aligned/bilinear_quotient/ops/run_successor_pointer_cross_type_interaction_v1.py", "runner"),
    "successor_cross_type_result_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_CROSS_TYPE_INTERACTION_V1_RESULT.json", "result"),
}


def artifact(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def bind(record, event):
    event["design_key"] = design_key(record, event)
    event["execution_key"] = execution_key(record, event)
    return event


def main():
    result = json.loads((HERE / "SUCCESSOR_POINTER_CROSS_TYPE_INTERACTION_V1_RESULT.json").read_text())
    assert result["terminal"] == "cross_type_interaction"
    assert result["predictions"] == {
        "pred_a_instrument_and_live_joint": True,
        "pred_b_additive_union": False,
        "pred_c_cross_type_interaction": True,
        "pred_d_mixed_or_unresolved": False,
    }
    assert result["price"]["observed_forwards"] == 11 and result["price"]["observed_sequences"] == 88
    assert result["instrument"] == {"full_ceiling_max_logit_error": 0.0, "self_max_logit_error": 0.0}
    assert all(report["joint_confirmation_passes"] and report["interaction_family_passes"] for report in result["family_reports"].values())

    path = circuit_path(TAG)
    existing = json.loads(path.read_text())
    if any(event["event_id"] == EVENT_ID for event in existing["evidence_events"]):
        validate_v2(existing); rebuild_registry_v2()
        print(json.dumps({"status": "already_registered", "claim_id": NEW_CLAIM}, indent=2)); return

    with _lock("registry"):
        record = json.loads(path.read_text())
        if any(event["event_id"] == EVENT_ID for event in record["evidence_events"]):
            raise RuntimeError("event appeared while acquiring registry lock")
        for artifact_id, spec in ARTIFACTS.items():
            value = artifact(*spec)
            if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
                raise ValueError(f"artifact collision: {artifact_id}")
            record["artifacts"][artifact_id] = value

        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM, "revision": 6, "supersedes": OLD_CLAIM,
            "evidence_event_ids": [*previous["evidence_event_ids"], EVENT_ID],
            "next_missing": (
                "The confirmed attention/MLP8-12 pointer mediator is not an additive union. Its negative "
                "cross-type inclusion-exclusion term is 63.5-80.3% of joint final-residual RMS and cancels "
                "24.5-54.4% of the target before softcap. Next test whether this exact 1152D calibration "
                "interaction compresses to a frozen low-rank basis with prospective sequence-length transfer; "
                "do not repeat component/group screens or use quantization."
            ),
        })
        for site in claim["candidate_sites"]:
            if site["site_id"] == "prefix_conditioned_baseline_branch":
                site["ceiling_event_ids"] = [*site["ceiling_event_ids"], EVENT_ID]
                site["tensor_path"] = "attention/MLP8-12 joint response plus a negative cross-type final-residual calibration interaction"
        record["claims"].append(claim)

        digit = result["family_reports"]["digit"]
        month = result["family_reports"]["month"]
        event = {
            "event_id": EVENT_ID, "claim_id": NEW_CLAIM, "test_type": "composition",
            "stage": "complete", "verdict": "held", "failure_kind": None,
            "family_ids": ["prefix_change_final_pointer_preserved"],
            "site_id": "prefix_conditioned_baseline_branch", "split_plan_id": None,
            "evaluation_role": "mechanistic_interaction_decomposition",
            "metrics": [
                {"name": "digit_final_residual_interaction_fraction", "estimate": digit["final_residual_interaction_fraction"], "ci95": None, "bar": ">=0.20"},
                {"name": "month_final_residual_interaction_fraction", "estimate": month["final_residual_interaction_fraction"], "ci95": None, "bar": ">=0.20"},
                {"name": "digit_pre_softcap_interaction_target_projection", "estimate": digit["pre_softcap_interaction_vs_target"]["projection"], "ci95": None, "bar": "absolute >=0.10"},
                {"name": "month_pre_softcap_interaction_target_projection", "estimate": month["pre_softcap_interaction_vs_target"]["projection"], "ci95": None, "bar": "absolute >=0.10"},
                {"name": "maximum_interaction_control_fraction", "estimate": max(result["family_reports"][f][k] for f in result["family_reports"] for k in ("interaction_late_forward_control_fraction", "interaction_early_backward_control_fraction")), "ci95": None, "bar": "<=0.50"},
            ],
            "prereg_artifact_id": "successor_cross_type_prereg_v1",
            "result_artifact_id": "successor_cross_type_result_v1",
            "input_artifact_ids": list(ARTIFACTS), "seed": None, "checkpoint_sha256": None,
            "supersedes_event_id": None, "replicates_event_id": None,
            "sections": ["basis_aligned/polynomial_causal/SUCCESSOR_POINTER_CROSS_TYPE_INTERACTION_V1_PREREGISTRATION.md"],
            "notes": (
                "Inclusion-exclusion was measured at final residual, RMS-normalized pre-softcap score, and "
                "post-softcap margins. A and M over-add; a large selective negative interaction calibrates AM."
            ),
        }
        record["evidence_events"].append(bind(record, event))
        validate_v2(record); _atomic_json(path, record)
    rebuild_registry_v2()
    final = json.loads(path.read_text()); validate_v2(final)
    print(json.dumps({"status": "registered", "claim_id": NEW_CLAIM, "event_id": EVENT_ID}, indent=2))


if __name__ == "__main__":
    main()
