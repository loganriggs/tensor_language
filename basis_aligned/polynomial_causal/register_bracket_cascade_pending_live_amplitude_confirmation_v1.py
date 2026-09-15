#!/usr/bin/env python3
"""Register the seventh-construction null for the frozen bracket suffix law."""
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BQ = REPO / "basis_aligned/bilinear_quotient"
sys.path.insert(0, str(BQ))

from circuit_registry_v2 import (  # noqa: E402
    _atomic_json,
    _lock,
    circuit_path,
    design_key,
    execution_key,
    file_sha256,
    rebuild_registry_v2,
    validate_v2,
)

TAG = "task.bracket_pending_opener"
OLD = "pending_opener_state.v36"
NEW = "pending_opener_state.v37"
EVENT = "pending_opener.cascade_live_direct_amplitude.null.v1"
RESULT = HERE / "BRACKET_CASCADE_PENDING_LIVE_AMPLITUDE_CONFIRMATION_V1_RESULT.json"
ARTIFACTS = {
    "cascade_pending_rows_builder_v1": ("basis_aligned/polynomial_causal/build_bracket_cascade_pending_ood_v1_rows.py", "builder"),
    "cascade_pending_rows_test_v1": ("basis_aligned/polynomial_causal/test_build_bracket_cascade_pending_ood_v1_rows.py", "test"),
    "cascade_pending_rows_v1": ("basis_aligned/polynomial_causal/BRACKET_CASCADE_PENDING_OOD_V1_ROWS.json", "rows"),
    "cascade_live_amplitude_v1_prereg": ("basis_aligned/polynomial_causal/BRACKET_CASCADE_PENDING_LIVE_AMPLITUDE_CONFIRMATION_V1_PREREGISTRATION.md", "preregistration"),
    "cascade_live_amplitude_v1_binding": ("basis_aligned/polynomial_causal/BRACKET_CASCADE_PENDING_LIVE_AMPLITUDE_CONFIRMATION_V1_BINDING.json", "binding"),
    "cascade_live_amplitude_v1_runner": ("basis_aligned/bilinear_quotient/ops/run_bracket_cascade_pending_live_amplitude_confirmation_v1.py", "runner"),
    "cascade_live_amplitude_v1_result": ("basis_aligned/polynomial_causal/BRACKET_CASCADE_PENDING_LIVE_AMPLITUDE_CONFIRMATION_V1_RESULT.json", "result"),
}


def artifact(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main():
    result = json.loads(RESULT.read_text())
    expected = {
        "pred_a_exact_instrument_and_capability": True,
        "pred_b_exact_joint_parent_live": True,
        "pred_c_donor_free_source_transfers": False,
        "pred_d_live_amplitude_predicts_exact_effect": False,
        "pred_e_live_amplitude_predicts_program_effect": True,
        "pred_f_control_selectivity": True,
    }
    assert result["terminal"] == "source_or_control_null"
    assert result["predictions"] == expected
    assert result["live_amplitude_program"]["coefficients"] == [0.0005561862682518956]
    assert result["price"]["observed_forwards"] == 4
    assert result["price"]["observed_sequences"] == 576
    assert result["price"]["fits"] == 0

    path = circuit_path(TAG)
    current = json.loads(path.read_text())
    if any(event["event_id"] == EVENT for event in current["evidence_events"]):
        validate_v2(current)
        rebuild_registry_v2()
        print("already registered")
        return

    with _lock("registry"):
        record = json.loads(path.read_text())
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD)
        for artifact_id, spec in ARTIFACTS.items():
            record["artifacts"][artifact_id] = artifact(*spec)

        split = "pending_opener_seventh_cascade_v1"
        record["split_plans"].append({
            "split_plan_id": split,
            "unit": "fresh seventh-construction lexical group and exact token sequence",
            "partition_artifact_id": "cascade_pending_rows_v1",
            "builder_artifact_id": "cascade_pending_rows_builder_v1",
            "seed": None,
            "groups": {"SEVENTH_CONSTRUCTION_TEST": 72},
            "leakage_group_keys": ["exact token sequence", "lexical group", "construction template", "row id"],
            "sealed_before_outcomes": True,
            "sealed_at": "2026-09-15T00:44:50Z",
        })

        claim = deepcopy(previous)
        claim.update({
            "claim_id": NEW,
            "revision": 37,
            "supersedes": OLD,
            "evidence_event_ids": [*previous["evidence_event_ids"], EVENT],
            "next_missing": "The frozen one-parameter global direct-readout amplitude law is rejected on a prospective seventh construction under its pairwise bars. Aggregate amplitude-versus-exact behavior remains strong (cosine .9553, relative L2 .3350, sign 1.0), and amplitude predicts the donor-free program closely (cosine .9935, relative L2 .1222), but pair 1->60 has .5205 relative error and norm ratio 1.5150. The frozen donor-free source is also a strict near-null because pair 1->8 has .50059 relative error against a .50 bar. Do not refit or relax bars on the seventh rows; retain these components as approximate explanatory factors and move circuit-completion effort to another unresolved circuit or a separately preregistered new mechanism.",
        })
        claim["counterfactual_families"].append({
            "family_id": "cascade_pending_stack_top",
            "role": "interchange",
            "changes": ["innermost pending delimiter type", "correct immediate closer"],
            "holds_fixed": ["outer and middle pending delimiters", "seventh surface and words", "aligned positions"],
            "builder_artifact_id": "cascade_pending_rows_builder_v1",
            "control_ids": ["exact donor-term ceiling", "donor-free source program", "frozen global live-amplitude law"],
            "split_plan_id": split,
            "status": "validated",
        })
        claim["split_plan_ids"] = [*claim.get("split_plan_ids", []), split]
        record["claims"].append(claim)

        overall = result["overall"]
        event = {
            "event_id": EVENT,
            "claim_id": NEW,
            "test_type": "cross_family_transfer",
            "stage": "complete",
            "verdict": "null",
            "failure_kind": "scientific_null",
            "family_ids": ["cascade_pending_stack_top"],
            "site_id": "pending_opener.suffix_direct_readout_live_amplitude",
            "split_plan_id": split,
            "evaluation_role": "prospective_seventh_construction_confirmation",
            "metrics": [
                {"name": "amplitude_vs_exact_relative_l2", "estimate": overall["amplitude_vs_exact"]["relative_l2_error"], "ci95": None, "bar": "<=0.40 overall and <=0.50 each pair"},
                {"name": "amplitude_vs_exact_cosine", "estimate": overall["amplitude_vs_exact"]["cosine"], "ci95": None, "bar": ">=0.90 overall and >=0.75 each pair"},
                {"name": "amplitude_vs_program_relative_l2", "estimate": overall["amplitude_vs_program"]["relative_l2_error"], "ci95": None, "bar": "<=0.40 overall and <=0.50 each pair"},
                {"name": "program_vs_exact_relative_l2", "estimate": overall["program_vs_exact"]["relative_l2_error"], "ci95": None, "bar": "<=0.40 overall and <=0.50 each pair"},
                {"name": "control_to_target_rms", "estimate": result["control_to_target_rms"], "ci95": None, "bar": "<=0.50"},
            ],
            "prereg_artifact_id": "cascade_live_amplitude_v1_prereg",
            "result_artifact_id": "cascade_live_amplitude_v1_result",
            "input_artifact_ids": list(ARTIFACTS),
            "seed": None,
            "checkpoint_sha256": result["checkpoint_sha256"],
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["basis_aligned/polynomial_causal/BRACKET_CASCADE_PENDING_LIVE_AMPLITUDE_CONFIRMATION_V1_PREREGISTRATION.md"],
            "notes": "Frozen global coefficient and rank-two source were evaluated without fits, coefficient/rank changes, construction features, donor state in the program arm, updates, or quantization. Aggregate metrics held, but immutable pairwise bars failed.",
        }
        event["design_key"] = design_key(record, event)
        event["execution_key"] = execution_key(record, event)
        record["evidence_events"].append(event)
        validate_v2(record)
        _atomic_json(path, record)

    rebuild_registry_v2()
    final = json.loads(path.read_text())
    validate_v2(final)
    print(json.dumps({"status": "registered", "claim_id": NEW, "event_id": EVENT}, indent=2))


if __name__ == "__main__":
    main()
