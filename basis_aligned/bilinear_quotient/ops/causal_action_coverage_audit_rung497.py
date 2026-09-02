#!/usr/bin/env python3
"""Rung497: CPU audit of finite causal-action coverage in the post-02:19 archive."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "CAUSAL_ACTION_COVERAGE_AUDIT_RUNG497_PREREGISTRATION.md"
PARENT = ROOT / "attention1_query_key_downstream_shapley_rung496_results.json"
OUT = ROOT / "causal_action_coverage_audit_rung497_results.json"
MATRIX = ROOT / "causal_action_coverage_audit_rung497_matrix.csv"
PREREG_SHA256 = "cd9a875840794d74637bd16f04d146eef02226f325ff9eb10aaffe68af59a300"
PARENT_SHA256 = "4e43f303c3f7bc7e2e70f6abc967c9fdc9b4b5d0e8bb4154ffec5b9b86dc177d"

FINITE_ACTIONS = ("remove", "restore", "substitute", "compose")
ALLOWED_ACTIONS = set(FINITE_ACTIONS) | {"equalize", "local_derivative"}
REQUIRED_OBSERVATIONS = (
    "per_example",
    "two_document_splits",
    "same_action_semantics",
    "physical_suffix",
    "dedicated_task",
    "heldout_circuits",
)


def record(
    evidence_id: str,
    family: str,
    result_file: str,
    actions: tuple[str, ...],
    instrument_path: str,
    *,
    candidates: int,
    sites: str,
    per_example: bool,
    document_splits: int,
    same_action_semantics: bool,
    physical_suffix: bool,
    dedicated_task: bool,
    circuit_scope: str,
    heldout_circuits: bool = False,
    sequential_actions: bool = False,
    known_positive: bool = False,
    known_negative: bool = False,
    corpus: str = "natural",
    bundle_required: bool = False,
    note: str = "",
) -> dict:
    return dict(
        evidence_id=evidence_id,
        family=family,
        result_file=result_file,
        actions=list(actions),
        instrument_path=instrument_path,
        candidates=candidates,
        sites=sites,
        per_example=per_example,
        document_splits=document_splits,
        two_document_splits=document_splits >= 2,
        same_action_semantics=same_action_semantics,
        physical_suffix=physical_suffix,
        dedicated_task=dedicated_task,
        circuit_scope=circuit_scope,
        heldout_circuits=heldout_circuits,
        sequential_actions=sequential_actions,
        known_positive=known_positive,
        known_negative=known_negative,
        corpus=corpus,
        bundle_required=bundle_required,
        note=note,
    )


EVIDENCE = (
    record(
        "eq_heads_subset", "equality_attention_terms",
        "equality_term_subset_factorial_stage1_results.json",
        ("remove", "restore", "compose"),
        "analysis.pred_a_instrument_and_endpoint_liveness",
        candidates=4, sites="L5H5,L7H3,L8H3,L8H4", per_example=False,
        document_splits=2, same_action_semantics=True, physical_suffix=True,
        dedicated_task=True, circuit_scope="equality_task_masks",
        corpus="natural", note="All 16 remove/extract subsets; JSON summaries only."),
    record(
        "eq_heads_transplant_natural", "equality_attention_terms",
        "equality_term_score_payload_rung459_results.json", ("substitute",),
        "pred_a_instrument", candidates=4, sites="L5H5->L8H4",
        per_example=False, document_splits=2, same_action_semantics=True,
        physical_suffix=True, dedicated_task=True,
        circuit_scope="equality_task_masks", known_positive=True,
        known_negative=True, corpus="natural",
        note="Known-positive score transplant and output/payload separation; aggregate receipt."),
    record(
        "eq_heads_transplant_code", "equality_attention_terms",
        "equality_score_code_ood_rung460_results.json", ("substitute",),
        "pred_a_instrument", candidates=4, sites="L5H5->L8H4",
        per_example=False, document_splits=2, same_action_semantics=True,
        physical_suffix=True, dedicated_task=True,
        circuit_scope="equality_task_masks", known_positive=True,
        known_negative=True, corpus="code",
        note="Frozen cross-corpus score transplant; aggregate receipt."),
    record(
        "eq_correction_interchange", "equality_correction_sites",
        "equality_score_correction_interchange_rung464_results.json",
        ("substitute",), "pred_a_instrument", candidates=5,
        sites="MLP8,MLP9,MLP12,A14,MLP17", per_example=False,
        document_splits=2, same_action_semantics=False, physical_suffix=True,
        dedicated_task=True, circuit_scope="equality_task_masks", corpus="code",
        note="Correction interchange is source-conditioned and not common across every site."),
    record(
        "eq_correction_factorial", "equality_correction_sites",
        "equality_correction_group_factorial_rung466_results.json",
        ("remove", "restore", "compose"), "pred_a_instrument", candidates=5,
        sites="MLP8,MLP9,MLP12,A14,MLP17", per_example=False,
        document_splits=2, same_action_semantics=True, physical_suffix=True,
        dedicated_task=True, circuit_scope="equality_task_masks", corpus="code",
        note="All 32 subsets, but only sufficient statistics were retained."),
    record(
        "eq_query_remove", "equality_correction_sites",
        "equality_query_position_intervention_rung472_results.json",
        ("remove", "compose"), "pred_a_instrument", candidates=3,
        sites="MLP8,MLP9,MLP12", per_example=True, document_splits=3,
        same_action_semantics=True, physical_suffix=True, dedicated_task=True,
        circuit_scope="equality_task_masks", corpus="natural+code",
        bundle_required=True,
        note="Per-document query/nonquery/full effects, not the 62-circuit battery."),
    record(
        "eq_query_factorial", "equality_correction_sites",
        "equality_query_subtractive_factorial_rung474_results.json",
        ("remove", "compose"), "pred_a_instrument", candidates=3,
        sites="MLP8,MLP9,MLP12", per_example=True, document_splits=3,
        same_action_semantics=True, physical_suffix=True, dedicated_task=True,
        circuit_scope="equality_task_masks", corpus="natural+code",
        bundle_required=True,
        note="Seven per-document subset effects under the subtractive intervention."),
    record(
        "mlp0_branch_circuits", "mlp0_branches_block1",
        "mlp0_branch_circuit_response_rung481_results.json", ("remove",),
        "pred_a_exact_lawful_instrument", candidates=4,
        sites="MLP0:T,C,I,S", per_example=False, document_splits=2,
        same_action_semantics=True, physical_suffix=True, dedicated_task=False,
        circuit_scope="32_discovery_circuits", corpus="natural",
        note="Finite branch-by-circuit summaries; conditional held-out circuits stayed closed."),
    record(
        "mlp0_block1_carriers", "mlp0_branches_block1",
        "mlp0_coupled_block1_bigram_response_rung486_results.json",
        ("remove", "compose"), "pred_a_exact_lawful_instrument", candidates=3,
        sites="MLP0-direct,attention1,MLP1", per_example=False,
        document_splits=2, same_action_semantics=True, physical_suffix=True,
        dedicated_task=False, circuit_scope="branch_effects", corpus="natural",
        note="Exact D/A/M factorial for T/C/I, retained as aggregate response profiles."),
    record(
        "mlp0_midpoint_interchange", "mlp0_branches_block1",
        "mlp1_finite_secant_factor_interchange_rung488_results.json",
        ("substitute",), "pred_a_exact_lawful_instrument", candidates=3,
        sites="MLP1 finite secant", per_example=False, document_splits=4,
        same_action_semantics=True, physical_suffix=True, dedicated_task=False,
        circuit_scope="branch_effects", corpus="natural",
        note="Held-out factor interchange, later bounded by the common-native-state control."),
    record(
        "mlp0_named_sources", "mlp0_branches_block1",
        "mlp1_live_state_source_decomposition_rung491_results.json",
        ("remove", "restore", "compose"), "pred_a_exact_lawful_instrument",
        candidates=4, sites="named sources entering MLP1", per_example=False,
        document_splits=4, same_action_semantics=True, physical_suffix=True,
        dedicated_task=False, circuit_scope="branch_effects", corpus="natural",
        note="Singleton and leave-one-source-out effects; no per-document bundle."),
    record(
        "mlp0_site_equalization", "mlp0_branches_block1",
        "mlp0_TI_site_graded_merge_intervention_rung493_results.json",
        ("equalize",), "pred_a_exact_lawful_live_merge_instrument", candidates=4,
        sites="attention1-direct,attention1-recomputed,MLP1", per_example=False,
        document_splits=2, same_action_semantics=True, physical_suffix=True,
        dedicated_task=False, circuit_scope="branch_effects", corpus="natural",
        note="All six branch pairs; equalization is not bidirectional substitution."),
    record(
        "mlp1_write_portability", "mlp1_write_adjustments",
        "mlp1_write_interface_portability_probe_results.json",
        ("restore", "substitute"), "pred_a_exact_lawful_live_transplant_instrument",
        candidates=2, sites="MLP1:T,I", per_example=True, document_splits=2,
        same_action_semantics=True, physical_suffix=True, dedicated_task=False,
        circuit_scope="branch_effects", corpus="natural", bundle_required=True,
        known_positive=True, known_negative=True,
        note="Own restoration is positive; cross-document donor substitution is a known negative."),
    record(
        "attention1_three_factor", "attention1_factor_writes",
        "attention1_downstream_use_quotient_rung495b_results.json",
        ("local_derivative",), "pred_a_exact_live_instrument", candidates=63,
        sites="attention1 raw write", per_example=False, document_splits=2,
        same_action_semantics=True, physical_suffix=False, dedicated_task=False,
        circuit_scope="32_discovery_circuits", corpus="natural", bundle_required=True,
        note="Exact pieces but only gradient contractions; no finite piece action."),
    record(
        "attention1_five_factor", "attention1_factor_writes",
        "attention1_query_key_downstream_shapley_rung496_results.json",
        ("local_derivative",), "pred_a_exact_live_instrument", candidates=45,
        sites="attention1 raw write", per_example=False, document_splits=2,
        same_action_semantics=True, physical_suffix=False, dedicated_task=False,
        circuit_scope="32_discovery_circuits", corpus="natural", bundle_required=True,
        note="Shapley/first/last downstream contractions; held-out circuits stayed closed."),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(data: dict, dotted: str):
    value = data
    for key in dotted.split("."):
        value = value[key]
    return value


def validate_evidence(records=EVIDENCE) -> list[dict]:
    seen = set()
    checked = []
    for item in records:
        assert item["evidence_id"] not in seen
        seen.add(item["evidence_id"])
        assert set(item["actions"]) <= ALLOWED_ACTIONS
        path = ROOT / item["result_file"]
        assert path.is_file(), path
        data = json.loads(path.read_text())
        assert data["status"] in ("complete", "completed")
        assert nested(data, item["instrument_path"]) is True
        bundle_hash_valid = None
        bundle_path = None
        if item["bundle_required"]:
            bundle = data.get("bundle")
            assert isinstance(bundle, dict) and bundle.get("path") and bundle.get("sha256")
            bundle_path = Path(bundle["path"])
            assert bundle_path.is_file(), bundle_path
            bundle_hash_valid = sha256(bundle_path) == bundle["sha256"]
            assert bundle_hash_valid
        checked.append({
            **item,
            "result_sha256": sha256(path),
            "bundle_path": None if bundle_path is None else str(bundle_path),
            "bundle_hash_valid": bundle_hash_valid,
            "instrument_valid": True,
        })
    return checked


def action_quality(item: dict) -> bool:
    return all(item[key] for key in REQUIRED_OBSERVATIONS)


def score_families(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for item in records:
        grouped[item["family"]].append(item)
    summaries = {}
    for family, rows in sorted(grouped.items()):
        actions = sorted({action for row in rows for action in row["actions"]})
        qualified = {
            action: any(action in row["actions"] and action_quality(row) for row in rows)
            for action in FINITE_ACTIONS
        }
        missing_actions = [action for action, present in qualified.items() if not present]
        family_ready = not missing_actions
        transition_ready = family_ready and any(row["sequential_actions"] for row in rows)
        summaries[family] = {
            "evidence_records": len(rows),
            "candidate_count_max": max(row["candidates"] for row in rows),
            "actions_observed_any_quality": actions,
            "qualified_finite_actions": qualified,
            "missing_qualified_actions": missing_actions,
            "archive_ready": family_ready,
            "transition_refinement_ready": transition_ready,
            "has_known_positive": any(row["known_positive"] for row in rows),
            "has_known_negative": any(row["known_negative"] for row in rows),
            "has_per_example_record": any(row["per_example"] for row in rows),
            "has_any_circuit_record": any("circuit" in row["circuit_scope"] for row in rows),
            "has_heldout_circuit_record": any(row["heldout_circuits"] for row in rows),
            "corpora": sorted({row["corpus"] for row in rows}),
        }
    return summaries


def write_matrix(records: list[dict]) -> None:
    fields = [
        "evidence_id", "family", "result_file", "result_sha256", "candidates",
        "sites", "actions", "per_example", "document_splits", "same_action_semantics",
        "physical_suffix", "dedicated_task", "circuit_scope", "heldout_circuits",
        "sequential_actions", "known_positive", "known_negative", "corpus",
        "bundle_path", "bundle_hash_valid", "instrument_valid", "note",
    ]
    with MATRIX.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            row = {key: item.get(key) for key in fields}
            row["actions"] = ";".join(item["actions"])
            writer.writerow(row)


def main() -> None:
    assert sha256(PREREG) == PREREG_SHA256
    assert sha256(PARENT) == PARENT_SHA256
    parent = json.loads(PARENT.read_text())
    assert parent["pred_a_exact_live_instrument"] is True
    assert parent["pred_b_shared_query_or_key_side"] is False
    assert parent["strong_null"] is True
    records = validate_evidence()
    summaries = score_families(records)
    pred_a = True
    pred_b = any(item["archive_ready"] for item in summaries.values())
    pred_c = any(item["transition_refinement_ready"] for item in summaries.values())
    next_step = (
        "build_action_indexed_quotient_from_archive" if pred_b and pred_c else
        "preregister_equality_matcher_action_quotient_calibration"
    )
    write_matrix(records)
    receipt = {
        "status": "complete",
        "rung": 497,
        "claim_level": "cpu_provenance_and_experiment_design_audit",
        "source_hashes": {
            str(PREREG): sha256(PREREG),
            str(PARENT): sha256(PARENT),
        },
        "frozen_action_alphabet": list(FINITE_ACTIONS),
        "required_observations": list(REQUIRED_OBSERVATIONS),
        "evidence_record_count": len(records),
        "family_count": len(summaries),
        "family_coverage": summaries,
        "pred_a_audit_integrity": pred_a,
        "pred_b_archive_action_complete": pred_b,
        "pred_c_transition_refinement_ready": pred_c,
        "archive_insufficient": not (pred_b and pred_c),
        "matrix": {"path": str(MATRIX), "sha256": sha256(MATRIX)},
        "selected_calibration_family": (
            "equality_attention_terms" if not (pred_b and pred_c) else None),
        "selection_reason": (
            "Only family with a held-out finite known-positive score transplant and a matched "
            "known-negative score/output split; recollect its full action table with dedicated "
            "equality masks plus downstream circuits before searching for new quotients."
            if not (pred_b and pred_c) else None),
        "next_step": next_step,
        "model_loaded": False,
        "gpu_used": False,
        "deployed_parameters_saved": 0,
        "deployed_parameters_added": 0,
    }
    OUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "rung": 497,
        "pred_a": pred_a,
        "pred_b": pred_b,
        "pred_c": pred_c,
        "archive_insufficient": receipt["archive_insufficient"],
        "selected_calibration_family": receipt["selected_calibration_family"],
        "next_step": next_step,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
