#!/usr/bin/env python3
"""Register nested-pending capability and the split vector/scalar transfer result."""
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


TAG = "task.bracket_pending_opener"
OLD_CLAIM = "pending_opener_state.v30"
NEW_CLAIM = "pending_opener_state.v31"
EVENTS = (
    "pending_opener.nested_pending_ood_capability.held.v1",
    "pending_opener.nested_pending_vector_program.null.v1",
    "pending_opener.nested_pending_six_scalar_law.held.v1",
)
ARTIFACTS = {
    "nested_pending_rows_builder_v1": ("basis_aligned/polynomial_causal/build_bracket_nested_pending_ood_v1_rows.py", "builder"),
    "nested_pending_rows_v1": ("basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_OOD_V1_ROWS.json", "dataset"),
    "nested_pending_capability_prereg_v1": ("basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_OOD_CAPABILITY_V1_PREREGISTRATION.md", "preregistration"),
    "nested_pending_capability_binding_v1": ("basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_OOD_CAPABILITY_V1_BINDING.json", "binding"),
    "nested_pending_capability_runner_v1": ("basis_aligned/bilinear_quotient/ops/run_bracket_nested_pending_ood_capability_v1.py", "runner"),
    "nested_pending_capability_result_v1": ("basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_OOD_CAPABILITY_V1_RESULT.json", "result"),
    "nested_pending_transfer_prereg_v1": ("basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_PROGRAM_TRANSFER_V1_PREREGISTRATION.md", "preregistration"),
    "nested_pending_transfer_binding_v1": ("basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_PROGRAM_TRANSFER_V1_BINDING.json", "binding"),
    "nested_pending_transfer_runner_v1": ("basis_aligned/bilinear_quotient/ops/run_bracket_nested_pending_program_transfer_v1.py", "runner"),
    "nested_pending_transfer_result_v1": ("basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_PROGRAM_TRANSFER_V1_RESULT.json", "result"),
}


def artifact(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def bind(record, item):
    item["design_key"] = design_key(record, item)
    item["execution_key"] = execution_key(record, item)
    return item


def make_event(event_id, test_type, verdict, metrics, result_id, site_id, notes):
    return {
        "event_id": event_id, "claim_id": NEW_CLAIM, "test_type": test_type,
        "stage": "complete", "verdict": verdict,
        "failure_kind": "scientific_null" if verdict == "null" else None,
        "family_ids": ["nested_two_pending_stack_top", "outer_pending_type_change_inner_fixed"],
        "site_id": site_id, "split_plan_id": "pending_opener_three_value_fresh_split_r545_v1",
        "evaluation_role": "prospective_third_nested_pending_construction",
        "metrics": metrics,
        "prereg_artifact_id": "nested_pending_transfer_prereg_v1" if result_id == "nested_pending_transfer_result_v1" else "nested_pending_capability_prereg_v1",
        "result_artifact_id": result_id, "input_artifact_ids": list(ARTIFACTS),
        "seed": None, "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "supersedes_event_id": None, "replicates_event_id": None,
        "sections": [ARTIFACTS[result_id][0]], "notes": notes,
    }


def main():
    capability = json.loads((HERE / "BRACKET_NESTED_PENDING_OOD_CAPABILITY_V1_RESULT.json").read_text())
    result = json.loads((HERE / "BRACKET_NESTED_PENDING_PROGRAM_TRANSFER_V1_RESULT.json").read_text())
    assert capability["terminal"] == "capability_pass" and all(capability["predictions"].values())
    assert result["terminal"] == "null"
    assert result["predictions"] == {
        "pred_a_exact_instrument": True,
        "pred_b_immutable_vector_program_transfers": False,
        "pred_c_each_ordered_pair_recurs": False,
        "pred_d_zero_dispatch_controls": True,
        "pred_e_six_scalar_effect_law_transfers": True,
    }
    path = circuit_path(TAG)
    existing = json.loads(path.read_text())
    if all(any(item["event_id"] == key for item in existing["evidence_events"]) for key in EVENTS):
        validate_v2(existing); rebuild_registry_v2()
        print(json.dumps({"status":"already_registered","claim_id":NEW_CLAIM},indent=2)); return
    with _lock("registry"):
        record = json.loads(path.read_text())
        if any(any(item["event_id"] == key for item in record["evidence_events"]) for key in EVENTS):
            raise RuntimeError("partial registration exists")
        for artifact_id, spec in ARTIFACTS.items():
            value = artifact(*spec)
            if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
                raise ValueError(f"artifact collision: {artifact_id}")
            record["artifacts"][artifact_id] = value
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM, "revision": 31, "supersedes": OLD_CLAIM,
            "evidence_event_ids": [*previous["evidence_event_ids"], *EVENTS],
            "next_missing": (
                "The exact ordered-pair effect admits a prospective six-scalar behavioral law across a third "
                "nested-stack construction, but the six fixed L13H8 displacement vectors over-amplify there and "
                "are not a universal executable intervention. Do not refit or gain-rescue them. Selective "
                "pair-centered necessity remains null; move to another circuit unless a new causal contrast is "
                "derived independently from distance-dependent CE."
            ),
        })
        claim["counterfactual_families"].extend([
            {
                "family_id": "nested_two_pending_stack_top", "role": "interchange",
                "changes": ["inner stack-top pending delimiter type", "correct immediate closer"],
                "holds_fixed": ["outer third-type pending delimiter", "words", "length", "positions"],
                "builder_artifact_id": "nested_pending_rows_builder_v1",
                "control_ids": ["outer pending type change", "zero dispatch", "native factor replay"],
                "split_plan_id": "pending_opener_three_value_fresh_split_r545_v1", "status": "validated",
            },
            {
                "family_id": "outer_pending_type_change_inner_fixed", "role": "invariance",
                "changes": ["outer pending delimiter type"],
                "holds_fixed": ["inner stack-top delimiter", "correct immediate closer", "words", "length"],
                "builder_artifact_id": "nested_pending_rows_builder_v1",
                "control_ids": ["zero displacement dispatch", "native closer capability"],
                "split_plan_id": "pending_opener_three_value_fresh_split_r545_v1", "status": "validated",
            },
        ])
        claim["candidate_sites"].append({
            "site_id": "pending_opener.ordered_pair_six_scalar_endpoint_law",
            "tensor_path": "six ordered closer-pair scalar predictions of exact donor-term endpoint effect",
            "shape": [6],
            "intervention": "prediction only; no internal vector or state is installed",
            "ceiling_event_ids": [EVENTS[2]],
        })
        record["claims"].append(claim)
        min_accuracy = min(value["accuracy"] for value in capability["cell_reports"].values())
        vector, scalar = result["overall_vector_program"], result["scalar_overall"]
        events = [
            make_event(EVENTS[0], "capability", "held", [
                {"name":"minimum_nested_cell_accuracy","estimate":min_accuracy,"ci95":None,"bar":">=0.75 each cell"},
                {"name":"nested_endpoint_count","estimate":144.0,"ci95":None,"bar":"=144"},
            ], "nested_pending_capability_result_v1", None, "All target and answer-preserving control cells pass without filtering."),
            make_event(EVENTS[1], "ood", "null", [
                {"name":"vector_program_effect_cosine","estimate":vector["cosine"],"ci95":None,"bar":">=0.80"},
                {"name":"vector_program_relative_l2","estimate":vector["relative_l2_error"],"ci95":None,"bar":"<=0.60"},
                {"name":"vector_program_norm_ratio","estimate":vector["predicted_to_actual_norm_ratio"],"ci95":None,"bar":"0.50--1.50"},
                {"name":"vector_program_sign_agreement","estimate":vector["sign_agreement"],"ci95":None,"bar":">=0.90"},
            ], "nested_pending_transfer_result_v1", "attention13.head8.ordered_pair_displacement_program.final_query", "Direction transfers, magnitude does not; three ordered-pair cells fail relative-error bars."),
            make_event(EVENTS[2], "cross_family_transfer", "held", [
                {"name":"six_scalar_effect_cosine","estimate":scalar["cosine"],"ci95":None,"bar":">=0.80"},
                {"name":"six_scalar_effect_relative_l2","estimate":scalar["relative_l2_error"],"ci95":None,"bar":"<=0.60"},
                {"name":"six_scalar_effect_norm_ratio","estimate":scalar["predicted_to_actual_norm_ratio"],"ci95":None,"bar":"0.50--1.50"},
                {"name":"six_scalar_effect_sign_agreement","estimate":scalar["sign_agreement"],"ci95":None,"bar":">=0.90"},
            ], "nested_pending_transfer_result_v1", "pending_opener.ordered_pair_six_scalar_endpoint_law", "Prospective endpoint prediction only; native generator and suffix remain external."),
        ]
        for item in events: record["evidence_events"].append(bind(record,item))
        validate_v2(record); _atomic_json(path,record)
    rebuild_registry_v2(); validate_v2(json.loads(path.read_text()))
    print(json.dumps({"status":"registered","claim_id":NEW_CLAIM,"events":EVENTS},indent=2))


if __name__ == "__main__": main()
