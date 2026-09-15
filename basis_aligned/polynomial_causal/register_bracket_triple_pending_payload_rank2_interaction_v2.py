#!/usr/bin/env python3
"""Register the transferred rank-two bracket payload interaction program."""
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
OLD = "pending_opener_state.v32"
NEW = "pending_opener_state.v33"
EVENT = "pending_opener.triple_payload_rank2_live_score.held.v1"
RESULT = HERE / "BRACKET_TRIPLE_PENDING_PAYLOAD_RANK2_INTERACTION_V2_RESULT.json"
ARTIFACTS = {
    "triple_pending_rows_builder_v1": (
        "basis_aligned/polynomial_causal/build_bracket_triple_pending_ood_v1_rows.py", "builder"),
    "triple_pending_rows_test_v1": (
        "basis_aligned/polynomial_causal/test_build_bracket_triple_pending_ood_v1_rows.py", "test"),
    "triple_pending_rows_v1": (
        "basis_aligned/polynomial_causal/BRACKET_TRIPLE_PENDING_OOD_V1_ROWS.json", "rows"),
    "triple_payload_rank2_v1_prereg": (
        "basis_aligned/polynomial_causal/BRACKET_TRIPLE_PENDING_PAYLOAD_RANK2_INTERACTION_V1_PREREGISTRATION.md", "preregistration"),
    "triple_payload_rank2_v1_binding": (
        "basis_aligned/polynomial_causal/BRACKET_TRIPLE_PENDING_PAYLOAD_RANK2_INTERACTION_V1_BINDING.json", "binding"),
    "triple_payload_rank2_v1_runner": (
        "basis_aligned/bilinear_quotient/ops/run_bracket_triple_pending_payload_rank2_interaction_v1.py", "runner"),
    "triple_payload_rank2_v2_prereg": (
        "basis_aligned/polynomial_causal/BRACKET_TRIPLE_PENDING_PAYLOAD_RANK2_INTERACTION_V2_PREREGISTRATION.md", "preregistration"),
    "triple_payload_rank2_v2_binding": (
        "basis_aligned/polynomial_causal/BRACKET_TRIPLE_PENDING_PAYLOAD_RANK2_INTERACTION_V2_BINDING.json", "binding"),
    "triple_payload_rank2_v2_runner": (
        "basis_aligned/bilinear_quotient/ops/run_bracket_triple_pending_payload_rank2_interaction_v2.py", "runner"),
    "triple_payload_rank2_v2_result": (
        "basis_aligned/polynomial_causal/BRACKET_TRIPLE_PENDING_PAYLOAD_RANK2_INTERACTION_V2_RESULT.json", "result"),
}


def artifact(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main():
    result = json.loads(RESULT.read_text())
    assert result["terminal"] == "payload_rank2_interaction_transfer"
    assert all(result["predictions"].values())
    assert result["price"]["observed_forwards"] == 5
    assert result["price"]["observed_sequences"] == 1440
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
        split_plan_id = "pending_opener_fourth_triple_pending_v1"
        if not any(item["split_plan_id"] == split_plan_id for item in record["split_plans"]):
            record["split_plans"].append({
                "split_plan_id": split_plan_id,
                "unit": "fresh lexical group and exact token sequence across third-authority training and fourth-construction test",
                "partition_artifact_id": "triple_pending_rows_v1",
                "builder_artifact_id": "triple_pending_rows_builder_v1",
                "seed": None,
                "groups": {"THIRD_AUTHORITY_TRAIN": 72, "FOURTH_CONSTRUCTION_TEST": 72},
                "leakage_group_keys": ["exact token sequence", "lexical group", "construction template", "row id"],
                "sealed_before_outcomes": True,
                "sealed_at": "2026-09-15T00:13:27Z",
            })
        claim = deepcopy(previous)
        claim.update({
            "claim_id": NEW, "revision": 33, "supersedes": OLD,
            "evidence_event_ids": [*previous["evidence_event_ids"], EVENT],
            "next_missing": (
                "A centered rank-two delimiter-type opener-payload table derived without outcomes transfers "
                "from the third to a fresh fourth triple-pending construction when multiplied by the live "
                "donor score: payload-difference relative error .0375 and downstream effect relative error "
                ".0100, versus .9967 for score-only, with controls .1197 of target RMS. Next derive frozen "
                "delimiter-type key prototypes for the two bilinear score factors and combine them with this "
                "payload table on another construction, retaining the live query but removing donor-state "
                "access; do not increase payload rank, fit gains, reuse full-term vectors, or quantize."
            ),
        })
        claim["counterfactual_families"].extend([
            {
                "family_id": "triple_pending_stack_top", "role": "interchange",
                "changes": ["inner stack-top pending delimiter type", "correct immediate closer"],
                "holds_fixed": ["outer brace opener", "middle pending delimiter", "fresh words", "length and positions"],
                "builder_artifact_id": "triple_pending_rows_builder_v1",
                "control_ids": ["middle pending type change", "exact joint ceiling", "score-only baseline"],
                "split_plan_id": split_plan_id, "status": "validated",
            },
            {
                "family_id": "middle_pending_type_change_inner_fixed", "role": "invariance",
                "changes": ["middle pending delimiter type"],
                "holds_fixed": ["outer brace opener", "inner stack-top delimiter", "correct immediate closer", "fresh words and length"],
                "builder_artifact_id": "triple_pending_rows_builder_v1",
                "control_ids": ["rank-two zero type difference", "native closer capability", "exact joint ceiling"],
                "split_plan_id": split_plan_id, "status": "validated",
            },
        ])
        claim["split_plan_ids"] = [*claim.get("split_plan_ids", []), split_plan_id]
        site = {
            "site_id": "attention13.head8.opener_type_payload_rank2_live_score",
            "tensor_path": "p_d scalar times (u_r plus centered rank-two delimiter-type payload difference)",
            "shape": ["batch", 1, 1152],
            "intervention": "install third-authority rank-two type payload difference at opener source and multiply by live donor score before native suffix",
            "ceiling_event_ids": [EVENT],
        }
        claim["candidate_sites"].append(site)
        record["claims"].append(claim)
        metrics = result["compressed_vs_exact_effect"]
        source = result["source_payload_difference"]
        event = {
            "event_id": EVENT, "claim_id": NEW, "test_type": "cross_family_transfer",
            "stage": "complete", "verdict": "held", "failure_kind": None,
            "family_ids": ["triple_pending_stack_top", "middle_pending_type_change_inner_fixed"],
            "site_id": site["site_id"], "split_plan_id": "pending_opener_fourth_triple_pending_v1",
            "evaluation_role": "fresh_construction_prospective_confirmation",
            "metrics": [
                {"name": "payload_difference_cosine", "estimate": source["cosine"], "ci95": None, "bar": ">=0.90"},
                {"name": "payload_difference_relative_l2", "estimate": source["relative_l2_error"], "ci95": None, "bar": "<=0.50"},
                {"name": "compressed_effect_cosine", "estimate": metrics["cosine"], "ci95": None, "bar": ">=0.95"},
                {"name": "compressed_effect_relative_l2", "estimate": metrics["relative_l2_error"], "ci95": None, "bar": "<=0.30"},
                {"name": "relative_l2_improvement_over_score_only", "estimate": result["relative_l2_improvement_over_score_only"], "ci95": None, "bar": ">=0.20"},
                {"name": "control_to_target_rms", "estimate": result["control_to_target_rms"], "ci95": None, "bar": "<=0.50"},
            ],
            "prereg_artifact_id": "triple_payload_rank2_v2_prereg",
            "result_artifact_id": "triple_payload_rank2_v2_result",
            "input_artifact_ids": list(ARTIFACTS), "seed": None,
            "checkpoint_sha256": result["checkpoint_sha256"],
            "supersedes_event_id": None, "replicates_event_id": None,
            "sections": ["basis_aligned/polynomial_causal/BRACKET_TRIPLE_PENDING_PAYLOAD_RANK2_INTERACTION_V2_PREREGISTRATION.md"],
            "notes": (
                "V1 was invalid due only to retaining full-vocabulary logits on GPU. V2 moved logits to CPU; "
                "the frozen program and scientific contract were unchanged. One outcome-blind SVD, no gradient, "
                "gain fit, update, full-term vector reuse, or quantization."
            ),
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
