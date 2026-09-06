#!/usr/bin/env python3
"""Causal cross-task test of has/had L9H1/H4 on is/was rows."""

# BQGATE: EXPERIMENT pred_a_authority_capability_identity_and_route_agreement pred_b_full_l9_attention_route_is_live pred_c_q_has_h1_h4_cross_task_reader_reuse pred_d_selected_heads_dominate_complement pred_e_selected_head_selectivity_and_exact_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import aspectual_tense_dual_eval as evaluator
import circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2 as rows_builder
import circuit_das_subspace as das
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_l9h1_h4_cross_task_reader_reuse_v1.json"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_l9h1_h4_cross_task_reader_reuse_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.l9h1_h4_cross_task_reader_reuse_v1"
PATHS = {
    "q_has_l8_l9_factorial": ROOT / "circuits/followups/aspectual_anchor_layer8_9_module_factorial_v2_result.json",
    "q_has_source_bank": ROOT / "circuits/followups/aspectual_anchor_l9h1_h4_downstream_source_bank_v2_result.json",
    "q_is_pair_support_null": ROOT / "circuits/followups/aspectual_tense_resid10_pair_support_task_gate_v1_result.json",
    "q_is_scale_authority": ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json",
    "producer": ROOT / "ops/circuit_fast_screen_producer.py",
    "matched_v2_builder": ROOT / "ops/circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2.py",
}
EXPECTED_PRIOR_SHA256 = "4df8da67d47954fefeb98cf5d69d1bc6215ca9770de6d81e9e2f7243c09e6a60"
EXPECTED = {
    "q_has_l8_l9_factorial": "06918c951ab52c0b6082f440addba553bb994fec02e1a774772473917fd40050",
    "q_has_source_bank": "6d694f92d35970f4eb5eba25ca3d9aff15cdbd1949db158a8be18e827e0423a7",
    "q_is_pair_support_null": "d8069bba6941f9f0f053888b22a2a63593241ce1cd67171f4110e440205c5c1b",
    "q_is_scale_authority": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "producer": "528e9f50152d6dc2b28d084cd0828de58de042b893703d0f322b4a2f22c4a0a7",
    "matched_v2_builder": "1f4b29bda3e26af3ee0102316ab0af166e317d1646e8b0b51332061245e606d6",
}
EXPECTED_ROWS_SHA256 = "2efd47b9a89d0f092688a96d75bbc33e5b89991a8e5de28723c714319b9ccceb"
SELECTED = (1, 4)
COMPLEMENT = (0, 2, 3, 5, 6, 7, 8)
ALL_HEADS = tuple(range(9))


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_arm(rows, base_output, donor_output, patched_output, scale):
    records = []
    for row, base_pair, donor_pair, patched_pair in zip(rows, base_output.answer_foil, donor_output.answer_foil, patched_output.answer_foil):
        family = row["family"]
        base_margin = float(base_pair[0]) - float(base_pair[1])
        donor_margin = float(donor_pair[0]) - float(donor_pair[1])
        patched_margin = float(patched_pair[0]) - float(patched_pair[1])
        if not all(math.isfinite(value) for value in (base_margin, donor_margin, patched_margin)):
            raise ExperimentError("nonfinite arm margin")
        record = {"family": family, "row_id": str(row["row_id"]), "direction": evaluator.direction_for(row), "base_native_margin": base_margin, "donor_native_margin": donor_margin, "patched_native_margin": patched_margin}
        if family in ("A1", "A2"):
            base_target, donor_target, patched_target = -base_margin, donor_margin, -patched_margin
            denominator = donor_target - base_target
            if denominator == 0.0:
                raise ExperimentError("zero A recovery denominator")
            record["recovery"] = (patched_target - base_target) / denominator
        else:
            record["normalized_effect"] = abs(patched_margin - base_margin) / scale
        records.append(record)
    summary = {}
    for family in evaluator.FAMILIES:
        selected = [record for record in records if record["family"] == family]
        key = "recovery" if family in ("A1", "A2") else "normalized_effect"
        summary[family] = evaluator.metric_summary(selected, key)
    summary["pooled_A_mean_recovery"] = statistics.fmean((summary["A1"]["mean_recovery"], summary["A2"]["mean_recovery"]))
    return records, summary


def max_pair_error(first, second):
    if len(first.answer_foil) != len(second.answer_foil):
        raise ExperimentError("route comparison coverage changed")
    return max(abs(float(a) - float(b)) for left, right in zip(first.answer_foil, second.answer_foil) for a, b in zip(left, right))


def prediction_record(a, b, c, d, e):
    return {
        "pred_a_authority_capability_identity_and_route_agreement": a,
        "pred_b_full_l9_attention_route_is_live": b,
        "pred_c_q_has_h1_h4_cross_task_reader_reuse": c,
        "pred_d_selected_heads_dominate_complement": d,
        "pred_e_selected_head_selectivity_and_exact_price": e,
    }


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    prior = json.loads(PRIOR.read_text())
    loaded = {name: json.loads(path.read_text()) for name, path in PATHS.items() if name not in ("producer", "matched_v2_builder")}
    rows_by_bank = rows_builder.build_rows_by_bank()
    rows_sha = rows_builder.validate_rows_by_bank(rows_by_bank)
    expected_authorities = {
        "q_has_l8_l9_factorial_sha256": EXPECTED["q_has_l8_l9_factorial"], "q_has_source_bank_sha256": EXPECTED["q_has_source_bank"],
        "q_is_pair_support_null_sha256": EXPECTED["q_is_pair_support_null"], "q_is_scale_authority_sha256": EXPECTED["q_is_scale_authority"],
        "producer_sha256": EXPECTED["producer"], "matched_v2_builder_sha256": EXPECTED["matched_v2_builder"], "q_is_v8_rows_sha256": EXPECTED_ROWS_SHA256,
    }
    ok = (
        prior.get("candidate_id") == CANDIDATE_ID and prior.get("authorities") == expected_authorities
        and rows_sha["is_was"] == EXPECTED_ROWS_SHA256 and len(rows_by_bank["is_was"]) == 64
        and loaded["q_has_l8_l9_factorial"].get("terminal") == "screen"
        and loaded["q_has_l8_l9_factorial"]["score"]["localized_attn09_singleton_heads"] == ["attn:09:head:01", "attn:09:head:04"]
        and loaded["q_has_source_bank"].get("terminal") == "screen"
        and loaded["q_is_pair_support_null"].get("terminal") == "null"
        and loaded["q_is_scale_authority"].get("terminal") == "screen"
        and evaluator.verify_contract() and set(SELECTED) | set(COMPLEMENT) == set(ALL_HEADS) and not (set(SELECTED) & set(COMPLEMENT))
    )
    if not ok:
        raise ExperimentError("candidate, prior evidence, rows, heads, or shared contract changed")
    return rows_by_bank["is_was"], loaded


def main():
    rows, loaded = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "rows": 64, "patch_arms": 5, "model_forwards": 7, "example_evaluations": 448, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_output = backend.native(base_batch, capture=True)
    donor_output = backend.native(donor_batch, capture=True)
    capability = evaluator.capability_cells("is_was", rows, {"base": base_output, "donor": donor_output}, full_two_sided=True)
    capability_pass = all(cell["passed"] for cell in capability)
    if not capability_pass:
        result = {
            "schema": "tense_auxiliary_is_was_l9h1_h4_cross_task_reader_reuse_result_v1", "candidate_id": CANDIDATE_ID,
            "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256,
            "capability_cells": capability, "patch_outcomes_opened": False, "arm_records": {}, "predictions": prediction_record(False, False, False, False, False),
            "price": {"model_forwards": 2, "example_evaluations": 128, "rows": 64, "patch_arms_opened": 0, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
            "terminal": "invalid", "reason": "native_capability_failed_before_head_patch_outcomes", "serial_seconds": time.perf_counter() - started,
            "next_action": "retain capability failure without changing heads or rows",
        }
        atomic_create_json(OUT, result)
        print(json.dumps(result, sort_keys=True))
        return

    base_cache, donor_cache = base_output.captured, donor_output.captured
    arm_outputs = {
        "identity_selected": backend.patched_heads(base_batch, layer=9, heads=SELECTED, donor_cache=base_cache),
        "donor_selected": backend.patched_heads(base_batch, layer=9, heads=SELECTED, donor_cache=donor_cache),
        "donor_complement": backend.patched_heads(base_batch, layer=9, heads=COMPLEMENT, donor_cache=donor_cache),
        "donor_all_heads": backend.patched_heads(base_batch, layer=9, heads=ALL_HEADS, donor_cache=donor_cache),
        "donor_whole_attention": backend.patched(base_batch, site=kernel.SiteRef(evidence_kind="module", site_id="attn:09"), donor_cache=donor_cache),
    }
    scale = float(loaded["q_is_scale_authority"]["score"]["families"]["target_scale"])
    arm_records, summaries = {}, {}
    for arm, output in arm_outputs.items():
        arm_records[arm], summaries[arm] = summarize_arm(rows, base_output, donor_output, output, scale)
    identity_error = max_pair_error(base_output, arm_outputs["identity_selected"])
    route_error = max_pair_error(arm_outputs["donor_all_heads"], arm_outputs["donor_whole_attention"])
    full, selected, complement = summaries["donor_all_heads"], summaries["donor_selected"], summaries["donor_complement"]
    retained = {family: selected[family]["mean_recovery"] / full[family]["mean_recovery"] if full[family]["mean_recovery"] != 0.0 else math.nan for family in ("A1", "A2")}
    exact_coverage = all(len(records) == 64 for records in arm_records.values())
    pred_a = capability_pass and identity_error <= 1e-5 and route_error <= 1e-5 and exact_coverage
    pred_b = all(full[family]["mean_recovery"] >= 0.25 and full[family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_c = all(math.isfinite(retained[family]) and retained[family] >= 0.60 and selected[family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_d = selected["pooled_A_mean_recovery"] > complement["pooled_A_mean_recovery"]
    pred_e = selected["P"]["mean_absolute_normalized_effect"] <= 0.20 and selected["C"]["mean_absolute_normalized_effect"] <= 0.20
    predictions = prediction_record(pred_a, pred_b, pred_c, pred_d, pred_e)
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b else "invalid")
    reason = {"screen": "l9h1_h4_contextual_reader_reused_across_tasks", "null": "l9h1_h4_cross_task_reuse_dominance_or_selectivity_misses", "invalid": "authority_capability_identity_full_route_agreement_coverage_finiteness_or_price_invalid"}[terminal]
    result = {
        "schema": "tense_auxiliary_is_was_l9h1_h4_cross_task_reader_reuse_result_v1", "candidate_id": CANDIDATE_ID,
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "authority_sha256": EXPECTED, "rows_sha256": EXPECTED_ROWS_SHA256, "capability_cells": capability, "patch_outcomes_opened": True,
        "instrument": {"identity_selected_max_abs_logit_error": identity_error, "all_heads_vs_whole_attention_max_abs_logit_error": route_error},
        "summaries": summaries, "retained_fraction_of_full": retained, "arm_records": arm_records, "predictions": predictions,
        "price": {"model_forwards": 7, "example_evaluations": 448, "rows": 64, "patch_arms": 5, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
        "terminal": terminal, "reason": reason, "serial_seconds": time.perf_counter() - started,
        "next_action": "test shared contextual source-bank organization across both tasks" if terminal == "screen" else "localize the is/was reader path independently without a head sweep",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability_cells", "instrument", "summaries", "retained_fraction_of_full", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
