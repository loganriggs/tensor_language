#!/usr/bin/env python3
"""Fit an upstream resid10 scalar read to predict the later aspectual carrier gain."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_capability_and_exact_heads pred_b_upstream_fit_and_actuator_well_formed pred_c_new_lexicon_A_prediction pred_d_new_lexicon_P_generalization pred_e_new_lexicon_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import run_aspectual_anchor_base_margin_affine_carrier_actuation_v1 as affine
import run_aspectual_anchor_base_margin_affine_fresh_lexicon_v2 as prospective
import run_aspectual_anchor_rank1_scalar_term_compression_split_v1 as scalar_parent
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_resid10_margin_to_carrier_gain_v1.json"
PROGRAM_V11 = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v11_result.json"
V10 = ROOT / "circuits/followups/aspectual_anchor_rank1_donor_free_margin_reflection_v1_result.json"
SCALAR = ROOT / "circuits/followups/aspectual_anchor_rank1_scalar_term_compression_split_v1_result.json"
DOSE = ROOT / "circuits/followups/aspectual_anchor_affine_carrier_dose_response_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_resid10_margin_to_carrier_gain_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.resid10_margin_to_carrier_gain_v1"
EXPECTED_PRIOR_SHA256 = "1c5ed420e4194260334776242757b096367c261726399e6581f1766eeaac8e0f"
EXPECTED = {
    PROGRAM_V11: "d8d96f3f0e6565d07fb8ca35f7aba9a361d5fb74a06dd153520131799e2ca309",
    V10: "cb1275e1b9449c52254a05efa9932aeaa916a2722c8ee72717231d9121e957fb",
    SCALAR: "4a55ef3da37b12722fabae41c9caaa7e8284fc0891ba4e15c5cfdeab40323b2d",
    DOSE: "4b3463aa095c03801a41453411ae663d14dad8340be6083ca42ff51264ec9a4f",
}
TOKEN_IDS = {"has": 468, "had": 550}
COUNTED_FORWARDS_MAX = 25
EXAMPLE_EVALUATIONS_MAX = 368


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_line(xs, ys):
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    ss_x = sum((x - x_mean) ** 2 for x in xs)
    if ss_x <= 1.0e-12:
        raise ExperimentError("degenerate resid10 calibration feature")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / ss_x
    intercept = y_mean - slope * x_mean
    ss_y = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / ss_y if ss_y > 0.0 else float("nan")
    return {"intercept": intercept, "slope": slope, "r2": r2, "count": len(xs)}


def summarize(records, key):
    values = [record[key] for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError(f"missing/nonfinite {key}")
    return {"count": len(values), f"mean_{key}": statistics.fmean(values), f"mean_absolute_{key}": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def fixed_confidence(backend, state, direction):
    current_id = TOKEN_IDS["has"] if direction == "present_to_past" else TOKEN_IDS["had"]
    other_id = TOKEN_IDS["had"] if direction == "present_to_past" else TOKEN_IDS["has"]
    current, other = affine.parent.pair_logits(backend, state, current_id, other_id)
    return current - other


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    lexical_rows, lexical_spec, _fresh_rows, _fresh_spec, rank1, _reference = scalar_parent.validate_static()
    confirm_rows, confirm_spec, confirm_rank1 = prospective.validate_static()
    prior = json.loads(PRIOR.read_text())
    v11 = json.loads(PROGRAM_V11.read_text())
    v10 = json.loads(V10.read_text())
    scalar = json.loads(SCALAR.read_text())
    dose = json.loads(DOSE.read_text())
    selection_ids = {record["row_id"] for record in scalar["term_amplitudes"] if record["phase"] == "selection"}
    calibration_rows = [row for row in lexical_rows if str(row["row_id"]) in selection_ids]
    alpha_by_id = {record["row_id"]: record["alpha"] for record in v10["intervention_records"] if record["kind"] == "A" and record["row_id"] in selection_ids}
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or prior["frozen_design"]["fixed_token_ids"] != TOKEN_IDS
        or v11.get("terminal") != "release"
        or dose.get("terminal") != "screen"
        or len(calibration_rows) != 16
        or len(alpha_by_id) != 16
        or len(confirm_rows) != 64
        or rank1["basis"]["sha256"] != confirm_rank1["basis"]["sha256"]
    ):
        raise ExperimentError("candidate, authority terminal, calibration alignment, confirmation rows, or basis changed")
    return calibration_rows, lexical_spec, alpha_by_id, confirm_rows, confirm_spec, rank1, dose


def main() -> None:
    calibration_rows, lexical_spec, alpha_by_id, rows, confirm_spec, rank1, dose = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_resid10_margin_to_carrier_gain_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "calibration_rows": 16,
        "confirmation_rows": 64,
        "feature_site": "resid:10",
        "write_site": "resid:18",
        "fixed_token_ids": TOKEN_IDS,
        "confirmation_resid18_margin_used_to_select_alpha": False,
        "confirmation_donor_activation_used_to_select_alpha": False,
        "counted_forwards_max": COUNTED_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "selected_head_pair_evaluations": 240,
        "grid_evaluations": 0,
        "model_backwards": 0,
        "model_updates": 0,
        "fit_parameters": 4,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = affine.parent.scalar_parent.empirical.component_parent.utc_now(), time.perf_counter()
    backend = affine.parent.producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    q = torch.tensor(rank1["basis"]["values_column_major"], device=backend.device, dtype=torch.float32)
    if q.shape != (1152,) or abs(float(q.norm()) - 1.0) > 1.0e-4 or hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != rank1["basis"]["sha256"]:
        raise ExperimentError("basis reconstruction failed")
    head10_ok, head10_error = affine.parent.das.verify_head(backend, calibration_rows[:8], "resid:10")
    head18_ok, head18_error = affine.parent.das.verify_head(backend, rows[:8], "resid:18")
    counted_forwards, evaluations, pair_evaluations = 2, 16, 0

    calibration_records = []
    for chunk in affine.parent.producer._chunks(calibration_rows, lexical_spec.batch_size):
        base10, _donor10, _ = affine.parent.das.capture_site(backend, chunk, "resid:10")
        counted_forwards += 2
        evaluations += 2 * len(chunk)
        for i, row in enumerate(chunk):
            direction = row["direction_id"]
            calibration_records.append({"row_id": str(row["row_id"]), "direction": direction, "resid10_confidence": fixed_confidence(backend, base10[i], direction), "target_guided_alpha": alpha_by_id[str(row["row_id"])]})
            pair_evaluations += 1
    fits = {}
    for direction in ("present_to_past", "past_to_present"):
        selected = [record for record in calibration_records if record["direction"] == direction]
        fits[direction] = fit_line([record["resid10_confidence"] for record in selected], [record["target_guided_alpha"] for record in selected])

    target_scale = float(rank1["score"]["families"]["target_scale"])
    capability_counts = {"A_base": [0, 0], "A_donor": [0, 0], "P_source": [0, 0], "C_actual_base": [0, 0]}
    records = []
    for family in ("A1", "A2", "P", "C"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in affine.parent.producer._chunks(family_rows, confirm_spec.batch_size):
            base10, donor10, _ = affine.parent.das.capture_site(backend, chunk, "resid:10")
            base18, donor18, _ = affine.parent.das.capture_site(backend, chunk, "resid:18")
            counted_forwards += 4
            evaluations += 4 * len(chunk)
            for i, row in enumerate(chunk):
                direction = affine.direction_for(row, family)
                source10 = donor10[i] if family == "P" else base10[i]
                source18 = donor18[i] if family == "P" else base18[i]
                confidence10 = fixed_confidence(backend, source10, direction)
                alpha = fits[direction]["intercept"] + fits[direction]["slope"] * confidence10
                if family in ("A1", "A2"):
                    current_id = TOKEN_IDS["has"] if direction == "present_to_past" else TOKEN_IDS["had"]
                    other_id = TOKEN_IDS["had"] if direction == "present_to_past" else TOKEN_IDS["has"]
                    base_original = affine.parent.pair_logits(backend, base18[i], current_id, other_id)
                    donor_original = affine.parent.pair_logits(backend, donor18[i], other_id, current_id)
                    base_margin = -(base_original[0] - base_original[1])
                    donor_margin = donor_original[0] - donor_original[1]
                    answer_id, foil_id = other_id, current_id
                    capability_counts["A_base"][0] += int(base_margin < 0.0)
                    capability_counts["A_base"][1] += 1
                    capability_counts["A_donor"][0] += int(donor_margin > 0.0)
                    capability_counts["A_donor"][1] += 1
                    pair_evaluations += 4
                elif family == "P":
                    current_id = TOKEN_IDS["has"] if direction == "present_to_past" else TOKEN_IDS["had"]
                    other_id = TOKEN_IDS["had"] if direction == "present_to_past" else TOKEN_IDS["has"]
                    original = affine.parent.pair_logits(backend, source18, current_id, other_id)
                    current_margin = original[0] - original[1]
                    base_margin, donor_margin = -current_margin, None
                    answer_id, foil_id = other_id, current_id
                    capability_counts["P_source"][0] += int(current_margin > 0.0)
                    capability_counts["P_source"][1] += 1
                    pair_evaluations += 3
                else:
                    original = affine.parent.pair_logits(backend, source18, row["base_answer_id"], row["base_foil_id"])
                    base_margin, donor_margin = original[0] - original[1], None
                    answer_id, foil_id = row["base_answer_id"], row["base_foil_id"]
                    capability_counts["C_actual_base"][0] += int(base_margin > 0.0)
                    capability_counts["C_actual_base"][1] += 1
                    pair_evaluations += 3
                patched = affine.parent.pair_logits(backend, source18 + alpha * q, answer_id, foil_id)
                patched_margin = patched[0] - patched[1]
                record = {"family": family, "row_id": str(row["row_id"]), "direction": direction, "resid10_confidence": confidence10, "alpha": alpha, "base_margin": base_margin, "patched_margin": patched_margin, "confirmation_resid18_margin_used_to_select_alpha": False, "confirmation_donor_activation_used_to_select_alpha": False, "row_target_or_foil_used_to_select_alpha": False}
                if family in ("A1", "A2"):
                    record["donor_reference_margin"] = donor_margin
                    record["recovery"] = (patched_margin - base_margin) / (donor_margin - base_margin)
                elif family == "P":
                    record["margin_reflection_fraction"] = (patched_margin - base_margin) / (-2.0 * base_margin)
                else:
                    record["normalized_unrelated_effect"] = abs(patched_margin - base_margin) / target_scale
                records.append(record)
            counted_forwards += 1
            evaluations += len(chunk)

    by_family = {family: [record for record in records if record["family"] == family] for family in ("A1", "A2", "P", "C")}
    summaries = {"A1": summarize(by_family["A1"], "recovery"), "A2": summarize(by_family["A2"], "recovery"), "P": summarize(by_family["P"], "margin_reflection_fraction"), "C": summarize(by_family["C"], "normalized_unrelated_effect")}
    reference = {"A1": dose["score"]["mean_response"]["A1"]["1.0"], "A2": dose["score"]["mean_response"]["A2"]["1.0"], "P": dose["score"]["mean_response"]["P"]["1.0"]}
    retention = {family: (summaries[family]["mean_recovery"] if family in ("A1", "A2") else summaries[family]["mean_margin_reflection_fraction"]) / reference[family] for family in ("A1", "A2", "P")}
    capability = all(correct == total for correct, total in capability_counts.values())
    fit_finite = all(all(math.isfinite(fits[direction][key]) for key in ("intercept", "slope", "r2")) and fits[direction]["count"] == 8 for direction in fits)
    actuator_identity = all(math.isfinite(record["alpha"]) and not record["confirmation_resid18_margin_used_to_select_alpha"] and not record["confirmation_donor_activation_used_to_select_alpha"] and not record["row_target_or_foil_used_to_select_alpha"] and record["alpha"] == fits[record["direction"]]["intercept"] + fits[record["direction"]]["slope"] * record["resid10_confidence"] for record in records)
    pred_a = capability and head10_ok and head18_ok and max(head10_error, head18_error) <= 1.0e-3
    pred_b = fit_finite and all(fits[direction]["r2"] >= 0.50 for direction in fits) and actuator_identity
    pred_c = all(summaries[family]["mean_recovery"] >= 0.75 and summaries[family]["direction_fraction"] >= 0.75 and retention[family] >= 0.65 for family in ("A1", "A2"))
    pred_d = summaries["P"]["mean_margin_reflection_fraction"] >= 0.75 and summaries["P"]["direction_fraction"] >= 0.75 and retention["P"] >= 0.65
    pred_e = summaries["C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_f = len(calibration_records) == 16 and len(records) == 64 and len({record["row_id"] for record in records}) == 64 and counted_forwards <= COUNTED_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX and pair_evaluations == 240
    predictions = {"pred_a_authority_alignment_capability_and_exact_heads": pred_a, "pred_b_upstream_fit_and_actuator_well_formed": pred_b, "pred_c_new_lexicon_A_prediction": pred_c, "pred_d_new_lexicon_P_generalization": pred_d, "pred_e_new_lexicon_C_selectivity": pred_e, "pred_f_exact_coverage_and_price": pred_f}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and fit_finite and actuator_identity and pred_f else "invalid")
    reason = {"screen": "resid10_scalar_predicts_lexicon_heldout_calibrated_carrier_write", "null": "resid10_fit_or_A_P_prediction_or_C_selectivity_fails", "invalid": "authority_alignment_capability_head_fit_actuator_or_coverage_invalid"}[terminal]
    result = {"schema": "aspectual_anchor_resid10_margin_to_carrier_gain_result_v1", "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only", "started_utc": started_utc, "finished_utc": affine.parent.scalar_parent.empirical.component_parent.utc_now(), "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256, "basis_sha256": rank1["basis"]["sha256"], "fixed_token_ids": TOKEN_IDS, "feature_site": "resid:10", "write_site": "resid:18", "fits": fits, "head_controls": {"resid10": {"passed": head10_ok, "max_abs_difference": head10_error}, "resid18": {"passed": head18_ok, "max_abs_difference": head18_error}}, "capability_counts": {key: {"correct": value[0], "total": value[1]} for key, value in capability_counts.items()}, "predictions": predictions, "score": {"families": summaries, "retention_vs_final_margin_controller": retention, "target_scale": target_scale, "counted_forwards": counted_forwards, "example_evaluations": evaluations, "selected_head_pair_evaluations": pair_evaluations, "grid_evaluations": 0, "calibration_record_count": len(calibration_records), "confirmation_record_count": len(records), "model_backwards": 0, "model_updates": 0, "fit_parameters": 4}, "calibration_records": calibration_records, "intervention_records": records, "terminal": terminal, "reason": reason, "next_action": "freeze upstream coefficients and validate on a second unseen lexicon" if terminal == "screen" else "retain final-margin controller and add a second upstream v11 variable"}
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "fits": fits, "families": summaries, "retention": retention, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
