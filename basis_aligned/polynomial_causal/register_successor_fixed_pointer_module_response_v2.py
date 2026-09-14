#!/usr/bin/env python3
"""Register the fixed-pointer complete-module response null."""
from __future__ import annotations

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
OLD_CLAIM = "successor_pointer_state.v3"
NEW_CLAIM = "successor_pointer_state.v4"
EVENT_ID = "successor_fixed_pointer_module_response.v2.complete.null"
ARTIFACTS = {
    "fixed_pointer_module_prereg_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V1_PREREGISTRATION.md", "preregistration"),
    "fixed_pointer_module_prior_art_v1": ("basis_aligned/bilinear_quotient/circuits/prior_art/successor_fixed_pointer_module_response_v1.json", "prior_art"),
    "fixed_pointer_module_binding_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V1_BINDING.json", "binding"),
    "fixed_pointer_module_runner_v1": ("basis_aligned/bilinear_quotient/ops/run_successor_fixed_pointer_module_response_v1.py", "runner"),
    "fixed_pointer_module_invalid_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V1_RESULT.json", "invalid_result"),
    "fixed_pointer_module_correction_v2": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V2_INSTRUMENT_CORRECTION.md", "preregistration_amendment"),
    "fixed_pointer_module_binding_v2": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V2_BINDING.json", "binding"),
    "fixed_pointer_module_runner_v2": ("basis_aligned/bilinear_quotient/ops/run_successor_fixed_pointer_module_response_v2.py", "runner"),
    "fixed_pointer_module_result_v2": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V2_RESULT.json", "result"),
}


def artifact(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def bind(record, event):
    event["design_key"] = design_key(record, event)
    event["execution_key"] = execution_key(record, event)
    return event


def main():
    result = json.loads((HERE / "SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V2_RESULT.json").read_text())
    assert result["terminal"] == "distributed_or_unresolved"
    assert result["predictions"] == {
        "pred_a_exact_instrument": True,
        "pred_b_live_target_and_capability": True,
        "pred_c_singleton_complete_module_response": False,
        "pred_d_distributed_or_unresolved_null": True,
    }
    assert result["price"]["observed_forwards"] == 77
    assert result["price"]["observed_sequences"] == 770
    assert result["instrument"] == {
        "full_ceiling_max_logit_error": 0.0,
        "self_max_logit_error": 0.0,
    }
    assert result["eligible_modules"] == []

    path = circuit_path(TAG)
    existing = json.loads(path.read_text())
    if any(event["event_id"] == EVENT_ID for event in existing["evidence_events"]):
        validate_v2(existing)
        rebuild_registry_v2()
        print(json.dumps({"status": "already_registered", "claim_id": NEW_CLAIM}, indent=2))
        return

    with _lock("registry"):
        record = json.loads(path.read_text())
        if any(event["event_id"] == EVENT_ID for event in record["evidence_events"]):
            raise RuntimeError("event appeared while acquiring the registry lock")
        for artifact_id, spec in ARTIFACTS.items():
            value = artifact(*spec)
            if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
                raise ValueError(f"artifact collision: {artifact_id}")
            record["artifacts"][artifact_id] = value

        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 4,
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": [*previous["evidence_event_ids"], EVENT_ID],
            "next_missing": (
                "The fixed-pointer prefix baseline is distributed at complete-module resolution: no singleton "
                "attention or MLP final-query response passed both month and digit, although attention8 was the "
                "strongest response. Do not threshold-rescue or immediately split attention8. Reopen successor "
                "only for an independently specified distributed or typed operation with fresh confirmation; "
                "otherwise move to a distinct unresolved circuit authority."
            ),
        })
        for site in claim["candidate_sites"]:
            if site["site_id"] == "prefix_conditioned_baseline_branch":
                site["ceiling_event_ids"] = [*site["ceiling_event_ids"], EVENT_ID]
                site["tensor_path"] = "distributed native final-query responses under fixed final pointer"
        record["claims"].append(claim)

        top = result["candidate_reports"]["attn8"]["families"]
        event = {
            "event_id": EVENT_ID,
            "claim_id": NEW_CLAIM,
            "test_type": "full_swap_ceiling",
            "stage": "complete",
            "verdict": "null",
            "failure_kind": "scientific_null",
            "family_ids": ["prefix_change_final_pointer_preserved"],
            "site_id": "prefix_conditioned_baseline_branch",
            "split_plan_id": None,
            "evaluation_role": "prospective_fresh_authored_complete_module_screen",
            "metrics": [
                {"name": "self_replay_max_logit_error", "estimate": 0.0, "ci95": None, "bar": "<=1e-5"},
                {"name": "full_ceiling_max_logit_error", "estimate": 0.0, "ci95": None, "bar": "<=1e-4"},
                {"name": "digit_late_backward_target_mean", "estimate": result["family_mean_late_backward_target"]["digit"], "ci95": None, "bar": ">=1.0"},
                {"name": "month_late_backward_target_mean", "estimate": result["family_mean_late_backward_target"]["month"], "ci95": None, "bar": ">=1.0"},
                {"name": "attention8_digit_projection", "estimate": top["digit"]["projection"], "ci95": None, "bar": ">=0.50 each family"},
                {"name": "attention8_month_projection", "estimate": top["month"]["projection"], "ci95": None, "bar": ">=0.50 each family"},
                {"name": "eligible_complete_modules", "estimate": 0.0, "ci95": None, "bar": ">=1 for singleton response"},
            ],
            "prereg_artifact_id": "fixed_pointer_module_prereg_v1",
            "result_artifact_id": "fixed_pointer_module_result_v2",
            "input_artifact_ids": list(ARTIFACTS),
            "seed": None,
            "checkpoint_sha256": None,
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V1_PREREGISTRATION.md"],
            "notes": (
                "V1 was invalid solely because its hook could not count calls made through the custom forward. "
                "V2 retained the design and counted at the call boundary. Exact instruments and capability pass; "
                "no complete module passes both families. Attention8 is a frozen near carrier, not a promoted site."
            ),
        }
        record["evidence_events"].append(bind(record, event))
        validate_v2(record)
        _atomic_json(path, record)
    rebuild_registry_v2()
    final = json.loads(path.read_text())
    validate_v2(final)
    print(json.dumps({"status": "registered", "claim_id": NEW_CLAIM, "event_id": EVENT_ID}, indent=2))


if __name__ == "__main__":
    main()
