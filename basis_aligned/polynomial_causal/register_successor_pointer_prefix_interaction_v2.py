#!/usr/bin/env python3
"""Register the prospective pointer x prefix result and its corrected interpretation."""
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
OLD_CLAIM = "successor_pointer_state.v2"
NEW_CLAIM = "successor_pointer_state.v3"
EVENT_ID = "successor_pointer_prefix_interaction.v2.complete.mixed"
ARTIFACTS = {
    "pointer_prefix_rows_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_PREFIX_INTERACTION_V1_ROWS.json", "dataset"),
    "pointer_prefix_prereg_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_PREFIX_INTERACTION_V1_PREREGISTRATION.md", "preregistration"),
    "pointer_prefix_binding_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_PREFIX_INTERACTION_V1_BINDING.json", "binding"),
    "pointer_prefix_runner_v1": ("basis_aligned/bilinear_quotient/ops/run_successor_pointer_prefix_interaction_v1.py", "runner"),
    "pointer_prefix_invalid_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_PREFIX_INTERACTION_V1_RESULT.json", "invalid_result"),
    "pointer_prefix_correction_v2": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_PREFIX_INTERACTION_V2_INSTRUMENT_CORRECTION.md", "preregistration_amendment"),
    "pointer_prefix_binding_v2": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_PREFIX_INTERACTION_V2_BINDING.json", "binding"),
    "pointer_prefix_runner_v2": ("basis_aligned/bilinear_quotient/ops/run_successor_pointer_prefix_interaction_v2.py", "runner"),
    "pointer_prefix_result_v2": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_PREFIX_INTERACTION_V2_RESULT.json", "result"),
    "pointer_prefix_analysis_source_v2": ("basis_aligned/polynomial_causal/successor_pointer_prefix_interaction_v2_analysis.py", "audit_implementation"),
    "pointer_prefix_analysis_v2": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_PREFIX_INTERACTION_V2_ANALYSIS.json", "audit"),
}


def artifact(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def bind(record, event):
    event["design_key"] = design_key(record, event)
    event["execution_key"] = execution_key(record, event)
    return event


def main():
    result = json.loads((HERE / "SUCCESSOR_POINTER_PREFIX_INTERACTION_V2_RESULT.json").read_text())
    analysis = json.loads((HERE / "SUCCESSOR_POINTER_PREFIX_INTERACTION_V2_ANALYSIS.json").read_text())
    assert result["terminal"] == "mixed_interaction"
    assert result["predictions"] == {
        "pred_a_instrument_live": True,
        "pred_b_answer_preserving_prefix_capability": True,
        "pred_c_coherent_run_gate": False,
        "pred_d_independent_pointer": False,
    }
    assert result["exactness"]["maximum_self_logit_absolute_error"] == 0.0
    assert analysis["maximum_identity_absolute_error"] <= 1e-6
    assert analysis["reports"]["digit"]["backward"]["late_swap_incoherent"][
        "donor_answer_win_flip_count"] == 4
    assert analysis["reports"]["month"]["backward"]["late_swap_incoherent"][
        "donor_answer_win_flip_count"] == 2

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
            "revision": 3,
            "supersedes": OLD_CLAIM,
            "status": "site_live",
            "evidence_event_ids": [*previous["evidence_event_ids"], EVENT_ID],
            "next_missing": (
                "Exact pointer effects are large but a late prefix-coherence break does not selectively amplify "
                "them; donor-win flips are caused mainly by a separate changed baseline decision state. Next run "
                "a fixed-pointer prefix-only complete-module response screen on the frozen coherent/late/early "
                "rows, scoring backward-alternative suppression with forward and early controls. Do not repeat "
                "legacy final-element component patching, which changes pointer and coherence together."
            ),
        })
        claim["causal_variable"] = {
            **claim["causal_variable"],
            "read": "the final element's exact layer-0 value-cache pointer plus a separately prefix-conditioned baseline state",
            "operation": (
                "apply an identity-keyed successor map whose effect competes with, but is not simply multiplied "
                "by, the prefix-conditioned continuation baseline"
            ),
        }
        for family in claim["counterfactual_families"]:
            if family["family_id"] == "prefix_change_final_pointer_preserved":
                family["builder_artifact_id"] = "pointer_prefix_rows_v1"
                family["status"] = "validated"
        claim["candidate_sites"].append({
            "site_id": "prefix_conditioned_baseline_branch",
            "tensor_path": "unknown complete native module response at the final query under fixed final pointer",
            "shape": ["module", "batch", 1152],
            "intervention": (
                "transplant coherent versus late-swap complete module responses while final pointer, token "
                "multiset, position and answer stay fixed"
            ),
            "ceiling_event_ids": [EVENT_ID],
        })
        record["claims"].append(claim)

        digit = result["interactions"]["digit"]["backward"]
        month = result["interactions"]["month"]["backward"]
        event = {
            "event_id": EVENT_ID,
            "claim_id": NEW_CLAIM,
            "test_type": "invariance",
            "stage": "complete",
            "verdict": "inconclusive",
            "failure_kind": "scientific_null",
            "family_ids": ["prefix_change_final_pointer_preserved", "internal_pointer_imposition"],
            "site_id": "layer0_value_cache_pointer",
            "split_plan_id": None,
            "evaluation_role": "prospective_fresh_authored_factorial_screen",
            "metrics": [
                {"name": "exact_self_replay_max_logit_error", "estimate": 0.0, "ci95": None, "bar": "<=1e-5"},
                {"name": "minimum_native_and_self_capability", "estimate": 1.0, "ci95": None, "bar": ">=0.75 every family-condition cell"},
                {"name": "digit_backward_late_pointer_interaction_scale", "estimate": digit["paired_differences"]["late_swap_incoherent"]["median_difference_in_native_margin_units"], "ci95": None, "bar": ">=0.20 for coherent-run gate"},
                {"name": "month_backward_late_pointer_interaction_scale", "estimate": month["paired_differences"]["late_swap_incoherent"]["median_difference_in_native_margin_units"], "ci95": None, "bar": ">=0.20 for coherent-run gate"},
                {"name": "digit_backward_flip_baseline_absolute_share", "estimate": analysis["reports"]["digit"]["backward"]["late_swap_incoherent"]["baseline_share_of_absolute_change_components"], "ci95": None, "bar": "descriptive decomposition"},
                {"name": "month_backward_flip_baseline_absolute_share", "estimate": analysis["reports"]["month"]["backward"]["late_swap_incoherent"]["baseline_share_of_absolute_change_components"], "ci95": None, "bar": "descriptive decomposition"},
            ],
            "prereg_artifact_id": "pointer_prefix_prereg_v1",
            "result_artifact_id": "pointer_prefix_result_v2",
            "input_artifact_ids": list(ARTIFACTS),
            "seed": None,
            "checkpoint_sha256": None,
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["basis_aligned/polynomial_causal/SUCCESSOR_POINTER_PREFIX_INTERACTION_V1_PREREGISTRATION.md"],
            "notes": (
                "V1 was invalid only for a batch-shape self-replay error. V2 preserved rows and bars, replayed "
                "self exactly, and rejected both a selective multiplicative coherence gate and strict pointer "
                "invariance. All six backward donor-win flips under the late swap are baseline dominated; this "
                "opens a separate fixed-pointer baseline branch rather than a threshold or site rescue."
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
