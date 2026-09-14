#!/usr/bin/env python3
"""Register the confirmed successor fixed-pointer typed response group."""
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
OLD_CLAIM = "successor_pointer_state.v4"
NEW_CLAIM = "successor_pointer_state.v5"
EVENT_ID = "successor_fixed_pointer_typed_group_length6.v1.complete.held"
ARTIFACTS = {
    "successor_typed_group_prereg_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_TYPED_GROUP_RESPONSE_V1_PREREGISTRATION.md", "preregistration"),
    "successor_typed_group_binding_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_TYPED_GROUP_RESPONSE_V1_BINDING.json", "binding"),
    "successor_typed_group_runner_v1": ("basis_aligned/bilinear_quotient/ops/run_successor_fixed_pointer_typed_group_response_v1.py", "runner"),
    "successor_typed_group_result_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_TYPED_GROUP_RESPONSE_V1_RESULT.json", "result"),
    "successor_length6_builder_v1": ("basis_aligned/polynomial_causal/build_successor_fixed_pointer_length6_confirmation_v1_rows.py", "builder"),
    "successor_length6_rows_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_ROWS.json", "rows"),
    "successor_length6_prereg_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_PREREGISTRATION.md", "preregistration"),
    "successor_length6_binding_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_BINDING.json", "binding"),
    "successor_length6_runner_v1": ("basis_aligned/bilinear_quotient/ops/run_successor_fixed_pointer_length6_confirmation_v1.py", "runner"),
    "successor_length6_result_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_RESULT.json", "result"),
}


def artifact(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def bind(record, event):
    event["design_key"] = design_key(record, event)
    event["execution_key"] = execution_key(record, event)
    return event


def main():
    screen = json.loads((HERE / "SUCCESSOR_FIXED_POINTER_TYPED_GROUP_RESPONSE_V1_RESULT.json").read_text())
    confirm = json.loads((HERE / "SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_RESULT.json").read_text())
    assert screen["terminal"] == "typed_group_screen"
    assert screen["eligible_groups"] == ["pointer_all", "all_attention"]
    assert screen["selected_group"] == "pointer_all"
    assert screen["price"]["observed_forwards"] == 23 and screen["price"]["observed_sequences"] == 230
    assert confirm["terminal"] == "pointer_group_confirmed"
    assert confirm["predictions"] == {
        "pred_a_exact_instrument": True,
        "pred_b_live_target_and_capability": True,
        "pred_c_pointer_group_confirmed": True,
        "pred_d_pointer_group_null": False,
    }
    assert confirm["price"]["observed_forwards"] == 7 and confirm["price"]["observed_sequences"] == 56
    assert confirm["instrument"] == {"full_ceiling_max_logit_error": 0.0, "self_max_logit_error": 0.0}
    reports = confirm["candidate_reports"]["pointer_all"]["families"]
    assert all(report["passes"] for report in reports.values())

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
            "claim_id": NEW_CLAIM, "revision": 5, "supersedes": OLD_CLAIM,
            "evidence_event_ids": [*previous["evidence_event_ids"], EVENT_ID],
            "next_missing": (
                "The fixed-pointer prefix baseline has a prospectively confirmed native-oracle typed mediator: "
                "joint attention/MLP8-12 response restoration passes month and digit at both length five and fresh "
                "length six, while attention-only and MLP-only groups fail the screen. Next decompose the frozen "
                "cross-type interaction or derive a non-oracle weight/executable interface; do not repeat group, "
                "singleton, gain, threshold, or quantization screens."
            ),
        })
        for site in claim["candidate_sites"]:
            if site["site_id"] == "prefix_conditioned_baseline_branch":
                site["ceiling_event_ids"] = [*site["ceiling_event_ids"], EVENT_ID]
                site["tensor_path"] = "joint native final-query attention/MLP8-12 response group under fixed final pointer"
                site["intervention"] = "restore the coherent attention/MLP8-12 final-query response group while preserving the recipient pointer and prompt"
        record["claims"].append(claim)

        event = {
            "event_id": EVENT_ID, "claim_id": NEW_CLAIM, "test_type": "full_swap_ceiling",
            "stage": "complete", "verdict": "held", "failure_kind": None,
            "family_ids": ["prefix_change_final_pointer_preserved"],
            "site_id": "prefix_conditioned_baseline_branch", "split_plan_id": None,
            "evaluation_role": "prospective_length_generalization_confirmation",
            "metrics": [
                {"name": "screen_pointer_all_digit_projection", "estimate": screen["candidate_reports"]["pointer_all"]["families"]["digit"]["projection"], "ci95": None, "bar": ">=0.50"},
                {"name": "screen_pointer_all_month_projection", "estimate": screen["candidate_reports"]["pointer_all"]["families"]["month"]["projection"], "ci95": None, "bar": ">=0.50"},
                {"name": "length6_digit_projection", "estimate": reports["digit"]["projection"], "ci95": None, "bar": ">=0.50"},
                {"name": "length6_month_projection", "estimate": reports["month"]["projection"], "ci95": None, "bar": ">=0.50"},
                {"name": "length6_digit_cosine", "estimate": reports["digit"]["cosine"], "ci95": None, "bar": ">=0.70"},
                {"name": "length6_month_cosine", "estimate": reports["month"]["cosine"], "ci95": None, "bar": ">=0.70"},
                {"name": "maximum_length6_control_fraction", "estimate": max(reports[f][k] for f in reports for k in ("late_forward_control_fraction", "early_backward_control_fraction")), "ci95": None, "bar": "<=0.50"},
            ],
            "prereg_artifact_id": "successor_length6_prereg_v1",
            "result_artifact_id": "successor_length6_result_v1",
            "input_artifact_ids": list(ARTIFACTS), "seed": None, "checkpoint_sha256": None,
            "supersedes_event_id": None, "replicates_event_id": None,
            "sections": ["basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_PREREGISTRATION.md"],
            "notes": (
                "The architecture-defined screen selected the smallest eligible group before length-six outcomes. "
                "The outcome-blind builder then froze every valid length-six window. The preselected group passed "
                "both families with no all-attention fallback. This is native-oracle mediation, not extraction."
            ),
        }
        record["evidence_events"].append(bind(record, event))
        validate_v2(record); _atomic_json(path, record)
    rebuild_registry_v2()
    final = json.loads(path.read_text()); validate_v2(final)
    print(json.dumps({"status": "registered", "claim_id": NEW_CLAIM, "event_id": EVENT_ID}, indent=2))


if __name__ == "__main__":
    main()
