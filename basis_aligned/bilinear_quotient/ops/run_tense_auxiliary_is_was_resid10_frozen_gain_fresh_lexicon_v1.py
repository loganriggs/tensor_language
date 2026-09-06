#!/usr/bin/env python3
"""Prospective disjoint-lexicon test of the frozen q_is upstream controller."""

# BQGATE: EXPERIMENT pred_a_authority_population_capability_and_exact_heads pred_b_frozen_upstream_program pred_c_new_lexicon_A_prediction pred_d_new_lexicon_P_generalization pred_e_new_lexicon_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v5 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_rank1_donor_free_margin_reflection_v1 as head


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_resid10_frozen_gain_fresh_lexicon_v1.json"
UPSTREAM = ROOT / "circuits/followups/tense_auxiliary_is_was_resid10_margin_to_root_gain_v1_result.json"
Q_IS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
ROOT_ACTUATOR = ROOT / "circuits/followups/tense_auxiliary_is_was_bracketed_root_actuator_fresh_lexicon_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v5.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_resid10_frozen_gain_fresh_lexicon_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.resid10_frozen_gain_fresh_lexicon_v1"
TOKEN_IDS = {"is": 318, "was": 373}
COEFFICIENTS = {
    "past_to_present": {"intercept": -147.6980274976786, "slope": -3678.9199345310312},
    "present_to_past": {"intercept": -348.2583777946352, "slope": 3040.696664893123},
}
EXPECTED_PRIOR_SHA256 = "244285b402794b2c5aa0f9954ed39d606b3bcc06a727c10271b48ae3c7781097"
EXPECTED = {
    UPSTREAM: "0c52305cf9ec3bba5bdc2f9ceedf2ec4ab047ab68a5accca4a53b3a6071f60f6",
    Q_IS: "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    ROOT_ACTUATOR: "342817d6f102497bcb48272337deb138f7c638c864b1213013516ef1454511cb",
    BUILDER: "92edf9ee04e36c17fa15c85611c75170fd036e475a3ddd7023d7f7bec07ac585",
}
EXPECTED_ROWS_SHA256 = "6be1822997b9ecfa21e8ffb648e7777127713e76890e55be4c2f00b8d24f7b10"
MODEL_FORWARDS_EXACT = 2
EXAMPLE_EVALUATIONS_EXACT = 128


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direction_for(row):
    return row["direction_id"] if row["family"] in ("A1", "A2") else ("present_to_past" if row["group_number"] % 2 == 0 else "past_to_present")


def requested_ids(direction):
    return (TOKEN_IDS["was"], TOKEN_IDS["is"]) if direction == "present_to_past" else (TOKEN_IDS["is"], TOKEN_IDS["was"])


def current_ids(direction):
    target, current = requested_ids(direction)
    return current, target


def summarize(records, key):
    values = [record[key] for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError(f"missing/nonfinite {key}")
    return {"count": len(values), f"mean_{key}": statistics.fmean(values), f"mean_absolute_{key}": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    upstream, qi, root = json.loads(UPSTREAM.read_text()), json.loads(Q_IS.read_text()), json.loads(ROOT_ACTUATOR.read_text())
    rows = fresh.build_rows()
    ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and prior["frozen_program"]["coefficients"] == COEFFICIENTS
        and tuple(prior["prospective_boundary"]["new_agents"] for _ in [0]) == (16,)
        and upstream.get("terminal") == "screen"
        and upstream["fits"] == {direction: {**COEFFICIENTS[direction], "count": 8, "r2": upstream["fits"][direction]["r2"]} for direction in COEFFICIENTS}
        and root.get("terminal") == "screen"
        and qi.get("terminal") == "screen"
        and upstream["basis_sha256"] == qi["basis"]["sha256"] == root["basis_sha256"]
        and fresh.validate_rows(rows) == EXPECTED_ROWS_SHA256
        and tuple(prior["authorities"]["fresh_lexicon_v5_rows_sha256"] for _ in [0]) == (EXPECTED_ROWS_SHA256,)
    )
    if not ok:
        raise ExperimentError("candidate, frozen coefficients, evidence, basis, rows, or authority changed")
    return rows, qi


def main():
    rows, qi = validate_static()
    plan = {
        "schema": "tense_auxiliary_is_was_resid10_frozen_gain_fresh_lexicon_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only_capability_first",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS_SHA256,
        "feature_site": "resid:10", "write_site": "resid:18", "fixed_token_ids": TOKEN_IDS,
        "coefficients": COEFFICIENTS, "model_forwards_exact": MODEL_FORWARDS_EXACT,
        "example_evaluations_exact": EXAMPLE_EVALUATIONS_EXACT, "inherited_fit_parameters": 4,
        "root_evaluations": 0, "grid_evaluations": 0, "transformer_backwards": 0, "model_updates": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    q = torch.as_tensor(qi["basis"]["values_column_major"], device=backend.device, dtype=torch.float32)
    if q.shape != (1152,) or hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != qi["basis"]["sha256"]:
        raise ExperimentError("basis reconstruction failed")

    outputs = {}
    for side in ("base", "donor"):
        outputs[side] = backend.native(das._batch(backend, rows, side=side), capture=True)
    native = []
    for i, row in enumerate(rows):
        direction = direction_for(row)
        for side in ("base", "donor"):
            correct = float(outputs[side].answer_foil[i][0]) > float(outputs[side].answer_foil[i][1])
            native.append({"family": row["family"], "direction": direction, "side": side, "row_id": str(row["row_id"]), "correct": correct})
    cells = []
    for family in ("A1", "A2", "P", "C"):
        for direction in ("past_to_present", "present_to_past"):
            selected = [record for record in native if record["family"] == family and record["direction"] == direction]
            accuracy = sum(record["correct"] for record in selected) / len(selected)
            threshold = 0.75 if family == "C" else 0.85
            cells.append({"family": family, "direction": direction, "correct": sum(record["correct"] for record in selected), "total": len(selected), "accuracy": accuracy, "threshold": threshold, "passed": accuracy >= threshold})
    capability_ok = all(cell["passed"] for cell in cells)

    records = []
    local_error = 0.0
    final_error = 0.0
    causal_outcomes_opened = False
    if capability_ok:
        causal_outcomes_opened = True
        target_scale = float(qi["score"]["families"]["target_scale"])
        for i, row in enumerate(rows):
            family, direction = row["family"], direction_for(row)
            source_side = "donor" if family == "P" else "base"
            source10 = torch.as_tensor(outputs[source_side].captured[(row["row_id"], "resid:10")], device=backend.device).float()
            source18 = torch.as_tensor(outputs[source_side].captured[(row["row_id"], "resid:18")], device=backend.device).float()
            donor18 = torch.as_tensor(outputs["donor"].captured[(row["row_id"], "resid:18")], device=backend.device).float()
            current, other = current_ids(direction)
            feature = float(head.selected_margin(backend, source10[None, :], [current], [other])[0])
            full10 = das.head_logits(backend, source10[None, :])
            local_error = max(local_error, abs(feature - float(full10[0, current] - full10[0, other])))
            coefficient = COEFFICIENTS[direction]
            alpha = coefficient["intercept"] + coefficient["slope"] * feature
            patched = source18 + alpha * q
            target, foil = requested_ids(direction)
            base_margin = float(head.selected_margin(backend, source18[None, :], [target], [foil])[0])
            patched_margin = float(head.selected_margin(backend, patched[None, :], [target], [foil])[0])
            full18 = das.head_logits(backend, source18[None, :])
            final_error = max(final_error, abs(base_margin - float(full18[0, target] - full18[0, foil])))
            record = {
                "family": family, "row_id": str(row["row_id"]), "direction": direction,
                "resid10_confidence": feature, "alpha": alpha, "base_target_margin": base_margin,
                "patched_target_margin": patched_margin, "root_search_used": False,
                "donor_activation_used_to_select_alpha": False, "donor_margin_used_to_select_alpha": False,
                "row_id_or_outcome_used_to_select_alpha": False,
            }
            if family in ("A1", "A2"):
                donor_reference = float(head.selected_margin(backend, donor18[None, :], [target], [foil])[0])
                record["donor_reference_margin"] = donor_reference
                record["recovery"] = (patched_margin - base_margin) / (donor_reference - base_margin)
            elif family == "P":
                record["margin_reflection_fraction"] = (patched_margin - base_margin) / (-2.0 * base_margin)
            else:
                before_c = float(head.selected_margin(backend, source18[None, :], [row["base_answer_id"]], [row["base_foil_id"]])[0])
                after_c = float(head.selected_margin(backend, patched[None, :], [row["base_answer_id"]], [row["base_foil_id"]])[0])
                record["normalized_unrelated_effect"] = abs(after_c - before_c) / target_scale
            records.append(record)

    families = {family: [record for record in records if record["family"] == family] for family in ("A1", "A2", "P", "C")}
    summaries = None
    if records:
        summaries = {
            "A1": summarize(families["A1"], "recovery"), "A2": summarize(families["A2"], "recovery"),
            "P": summarize(families["P"], "margin_reflection_fraction"), "C": summarize(families["C"], "normalized_unrelated_effect"),
        }
    pred_a = fresh.authority_sha256() == EXPECTED_ROWS_SHA256 and capability_ok and local_error <= 1.0e-4 and final_error <= 1.0e-4
    pred_b = len(records) == 64 and all(
        math.isfinite(record["alpha"])
        and record["alpha"] == COEFFICIENTS[record["direction"]]["intercept"] + COEFFICIENTS[record["direction"]]["slope"] * record["resid10_confidence"]
        and not record["root_search_used"] and not record["donor_activation_used_to_select_alpha"]
        and not record["donor_margin_used_to_select_alpha"] and not record["row_id_or_outcome_used_to_select_alpha"]
        for record in records
    )
    pred_c = bool(summaries) and all(summaries[family]["mean_recovery"] >= 0.75 and summaries[family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_d = bool(summaries) and summaries["P"]["mean_margin_reflection_fraction"] >= 0.75 and summaries["P"]["direction_fraction"] >= 0.75
    pred_e = bool(summaries) and summaries["C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_f = len(records) == 64 and len({record["row_id"] for record in records}) == 64
    predictions = {
        "pred_a_authority_population_capability_and_exact_heads": pred_a,
        "pred_b_frozen_upstream_program": pred_b,
        "pred_c_new_lexicon_A_prediction": pred_c,
        "pred_d_new_lexicon_P_generalization": pred_d,
        "pred_e_new_lexicon_C_selectivity": pred_e,
        "pred_f_exact_coverage_and_price": pred_f,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_f else "invalid")
    reason = {"screen": "frozen_q_is_upstream_program_predicts_fresh_lexicon", "null": "frozen_program_misses_fresh_A_P_or_C", "invalid": "authority_capability_head_basis_program_or_coverage_invalid"}[terminal]
    result = {
        "schema": "tense_auxiliary_is_was_resid10_frozen_gain_fresh_lexicon_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only_capability_first",
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "rows_sha256": EXPECTED_ROWS_SHA256, "basis_sha256": qi["basis"]["sha256"],
        "feature_site": "resid:10", "write_site": "resid:18", "fixed_token_ids": TOKEN_IDS,
        "coefficients": COEFFICIENTS, "capability_cells": cells, "causal_outcomes_opened": causal_outcomes_opened,
        "head_controls": {"resid10_local": {"max_abs_difference": local_error}, "resid18_local": {"max_abs_difference": final_error}},
        "score": {"families": summaries, "model_forwards": 2, "example_evaluations": 128, "record_count": len(records), "inherited_fit_parameters": 4, "root_evaluations": 0, "grid_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
        "intervention_records": records, "predictions": predictions, "terminal": terminal, "reason": reason,
        "evidence_scope": "prospective_fifth_lexicon_within_registered_constructions",
        "next_action": "compile q_is read-compute-write program and test joint q_has/q_is composition" if terminal == "screen" else "retain root actuator and kill this frozen affine controller without refit",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "capability_cells": cells, "causal_outcomes_opened": causal_outcomes_opened, "families": summaries, "price": result["score"], "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
