#!/usr/bin/env python3
"""Register R576 in numbered-list and numeric-sequence canonical records, CPU only."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


BQ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    _atomic_json, _lock, circuit_path, design_key, execution_key,
    rebuild_registry_v2, validate_v2,
)


LIST_TAG = "task.numbered_list.index_successor"
SEQUENCE_TAG = "task.numeric_sequence.continuation"
LIST_CLAIM = "numbered_list_index_successor.v8"
SEQUENCE_CLAIM = "numeric_sequence_continuation.v3"
COMPILE_ID = "numbered_list_cached_value_weights.r576.preregistered.v1"
REMOVAL_ID = "numbered_list_cached_value_removal.r576.preregistered.v1"
REUSE_ID = "numeric_sequence_cached_value_reuse.r576.preregistered.v1"


ARTIFACTS = {
    "r575_numeric_positions": {
        "path": "basis_aligned/bilinear_quotient/numeric_factor_removal_positions_rung575.json",
        "sha256": "3663ebc48e5dca1ff336cb0627fc43c6db8d7d6e1666b81d7631ab150168dd4b",
        "kind": "semantic_audit", "status": "frozen"},
    "r575_numeric_positions_script": {
        "path": "basis_aligned/bilinear_quotient/ops/numeric_factor_removal_positions_rung575.py",
        "sha256": "e9a5dbe3656a4be2694744b15da53c0cb805a604632dba5c35c33332452aadf5",
        "kind": "implementation", "status": "frozen"},
    "r576_prereg": {
        "path": "basis_aligned/polynomial_causal/NUMBERED_LIST_CACHED_VALUE_WEIGHT_REMOVAL_RUNG576_PREREGISTRATION.md",
        "sha256": "a776ebc1df29a6f3193d3315e190ec9494c95905596e450461c002378f8f59b6",
        "kind": "preregistration", "status": "frozen"},
    "r576_script": {
        "path": "basis_aligned/bilinear_quotient/ops/numbered_list_cached_value_weight_removal_rung576.py",
        "sha256": "91db3a2a9210aef915ce2e4f0a62253274e0b5470cbfaa05a95d50a3c0cf985a",
        "kind": "implementation", "status": "frozen"},
    "r576_test": {
        "path": "basis_aligned/bilinear_quotient/ops/test_numbered_list_cached_value_weight_removal_rung576.py",
        "sha256": "a341573a82e223e38643c26fddbae0e34ac47117741607eb1b64c03362bf6719",
        "kind": "test", "status": "frozen"},
    "r573_list_factor_result": {
        "path": "basis_aligned/bilinear_quotient/numbered_list_factor_localization_rung573_v2_results.json",
        "sha256": "052930b8b9086e8b7606e3d05929f521f468c04427be8d1182720f1772ee43ec",
        "kind": "result", "status": "frozen"},
    "r574_list_factor_audit": {
        "path": "basis_aligned/bilinear_quotient/numbered_list_factor_localization_rung574_audit.json",
        "sha256": "3d6580ee1a4f1bb77c07e4ee2b404bc23dc70f733db31425bc5da2a11a25a04e",
        "kind": "audit", "status": "frozen"},
}


def bind(record: dict, event: dict) -> dict:
    event["design_key"] = design_key(record, event)
    event["execution_key"] = execution_key(record, event)
    return event


def common_event(event_id: str, test_type: str, claim_id: str, families: list[str], site_id: str,
                 role: str, metrics: list[dict]) -> dict:
    return {"event_id": event_id, "test_type": test_type, "stage": "preregistered",
            "verdict": "inconclusive", "failure_kind": None, "family_ids": families,
            "site_id": site_id, "evaluation_role": role, "metrics": metrics,
            "result_artifact_id": None, "prereg_artifact_id": "r576_prereg",
            "input_artifact_ids": ["r575_numeric_positions", "r575_numeric_positions_script",
                                   "r573_list_factor_result", "r574_list_factor_audit",
                                   "r576_script", "r576_test"],
            "split_plan_id": "numbered_list_successor_split_r567_v1" if claim_id == LIST_CLAIM
                             else "numeric_sequence_continuation_split_r567_v1",
            "seed": 576, "checkpoint_sha256": None,
            "supersedes_event_id": None, "replicates_event_id": None,
            "sections": ["polynomial_causal/NUMBERED_LIST_CACHED_VALUE_WEIGHT_REMOVAL_RUNG576_PREREGISTRATION.md"],
            "claim_id": claim_id}


def add_artifacts(record: dict) -> None:
    for artifact_id, value in ARTIFACTS.items():
        if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
            raise ValueError(f"artifact collision: {artifact_id}")
        record["artifacts"][artifact_id] = value


def main() -> None:
    list_path, sequence_path = circuit_path(LIST_TAG), circuit_path(SEQUENCE_TAG)
    with _lock("registry"):
        list_record = json.loads(list_path.read_text())
        sequence_record = json.loads(sequence_path.read_text())
        if any(event["event_id"] == REMOVAL_ID for event in list_record["evidence_events"]):
            assert any(event["event_id"] == REUSE_ID for event in sequence_record["evidence_events"])
            validate_v2(list_record); validate_v2(sequence_record)
            print(json.dumps({"status": "already_registered"}, indent=2)); return
        add_artifacts(list_record); add_artifacts(sequence_record)

        prior_list = next(claim for claim in list_record["claims"]
                          if claim["claim_id"] == "numbered_list_index_successor.v7")
        list_claim = deepcopy(prior_list)
        list_claim.update({"claim_id": LIST_CLAIM, "revision": 8,
                           "supersedes": "numbered_list_index_successor.v7",
                           "next_missing": "run R576 exact weight compilation and active removal; adoption remains blocked unless list necessity and nonzero copy collateral controls both hold"})
        list_claim["evidence_event_ids"] = [*prior_list["evidence_event_ids"], COMPILE_ID, REMOVAL_ID]
        compiled_site = {"site_id": "final_label_l0_value_through_l8h3_h7",
                         "tensor_path": "sum_h p8[h,q,k] WO8[h] (lambda8 WV0[h] z0[k]) for h in {3,7}",
                         "shape": ["batch", "final query", "residual width 1152"],
                         "intervention": "replace or subtract the exact weight-computed final-source term",
                         "ceiling_event_ids": [COMPILE_ID, REMOVAL_ID]}
        if not any(site["site_id"] == compiled_site["site_id"] for site in list_claim["candidate_sites"]):
            list_claim["candidate_sites"].append(compiled_site)
        list_record["claims"].append(list_claim)
        list_families = [family["family_id"] for family in list_claim["counterfactual_families"]]
        compile_event = common_event(
            COMPILE_ID, "compiled_equivalence", LIST_CLAIM,
            ["list_two_line_state_shift", "list_three_line_state_shift"], compiled_site["site_id"],
            "FIT_then_conditional_SELECT_exact_activation_to_weight_equivalence",
            [{"name": "compiled_cached_value_and_logits_error", "estimate": None, "ci95": None,
              "bar": "cached bus, projected term, and activation-versus-weight patch relative squared errors <=1e-10"},
             {"name": "native_replay_error", "estimate": None, "ci95": None, "bar": "<=1e-12"}])
        removal_event = common_event(
            REMOVAL_ID, "removal", LIST_CLAIM, list_families, compiled_site["site_id"],
            "FIT_then_conditional_SELECT_active_necessity_and_copy_collateral",
            [{"name": "list_successor_removal_damage", "estimate": None, "ci95": None,
              "bar": ">=0.75 positive margin damage and positive bootstrap lower mean margin/CE damage in every non-copy list cell"},
             {"name": "active_repeated_list_copy_preservation", "estimate": None, "ci95": None,
              "bar": "term norm >=0.10 target scale, answer >=0.75, CE <=0.1 nat, margin and logit RMS <=0.25 target scales"},
             {"name": "split_and_price", "estimate": None, "ci95": None,
              "bar": "FIT then conditional SELECT; <=210 forwards; zero backwards; FINAL_TEST/OOD closed"}])
        list_record["evidence_events"].extend([
            bind(list_record, compile_event), bind(list_record, removal_event)])

        prior_sequence = next(claim for claim in sequence_record["claims"]
                              if claim["claim_id"] == "numeric_sequence_continuation.v2")
        sequence_claim = deepcopy(prior_sequence)
        sequence_claim.update({"claim_id": SEQUENCE_CLAIM, "revision": 3,
                               "supersedes": "numeric_sequence_continuation.v2",
                               "next_missing": "evaluate the fixed R576 list-derived cached-value path as a shared candidate while independently preparing a complete-state sequence site localization"})
        sequence_claim["evidence_event_ids"] = [*prior_sequence["evidence_event_ids"], REUSE_ID]
        reuse_site = deepcopy(compiled_site)
        reuse_site["ceiling_event_ids"] = [REUSE_ID]
        if not any(site["site_id"] == reuse_site["site_id"] for site in sequence_claim["candidate_sites"]):
            sequence_claim["candidate_sites"].append(reuse_site)
        sequence_record["claims"].append(sequence_claim)
        reuse_families = ["sequence_digit_state_shift", "sequence_word_state_shift",
                          "sequence_cross_format_shift", "sequence_digit_copy_control",
                          "sequence_word_copy_control"]
        reuse_event = common_event(
            REUSE_ID, "cross_family_transfer", SEQUENCE_CLAIM, reuse_families, reuse_site["site_id"],
            "fixed_list_derived_weight_factor_reuse_characterization",
            [{"name": "digit_word_cross_format_successor_removal_damage", "estimate": None, "ci95": None,
              "bar": "same necessity inequalities pass in every target family and endpoint on FIT and SELECT"},
             {"name": "active_digit_and_word_copy_preservation", "estimate": None, "ci95": None,
              "bar": "same nonzero intervention and collateral-preservation inequalities as R576"}])
        sequence_record["evidence_events"].append(bind(sequence_record, reuse_event))
        validate_v2(list_record); validate_v2(sequence_record)
        _atomic_json(list_path, list_record); _atomic_json(sequence_path, sequence_record)
    rebuild_registry_v2()
    print(json.dumps({"status": "registered", "list_claim": LIST_CLAIM,
                      "sequence_claim": SEQUENCE_CLAIM,
                      "events": [COMPILE_ID, REMOVAL_ID, REUSE_ID], "gpu_used": False}, indent=2))


if __name__ == "__main__":
    main()
