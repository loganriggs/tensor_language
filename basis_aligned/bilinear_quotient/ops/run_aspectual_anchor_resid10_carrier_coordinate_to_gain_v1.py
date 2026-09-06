#!/usr/bin/env python3
"""Test whether the resid18 carrier is also a weight-free resid10 gain reader."""

# BQGATE: EXPERIMENT pred_a_authority_population_capability_and_exactness pred_b_weight_free_fit_and_actuator_well_formed pred_c_A_prediction pred_d_P_generalization pred_e_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_aspectual_fresh_lexicon_v5 as fresh
import run_aspectual_anchor_base_margin_affine_carrier_actuation_v1 as affine
import run_aspectual_anchor_base_margin_affine_fresh_lexicon_v2 as prospective
import run_aspectual_anchor_resid10_frozen_gain_fresh_lexicon_v2 as v5
import run_aspectual_anchor_resid10_margin_to_carrier_gain_v1 as upstream
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_resid10_carrier_coordinate_to_gain_v1.json"
V12 = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v12_result.json"
UPSTREAM = ROOT / "circuits/followups/aspectual_anchor_resid10_margin_to_carrier_gain_v1_result.json"
V5 = ROOT / "circuits/followups/aspectual_anchor_resid10_frozen_gain_fresh_lexicon_v2_result.json"
SCALAR = ROOT / "circuits/followups/aspectual_anchor_rank1_scalar_term_compression_split_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_fresh_lexicon_v5.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_resid10_carrier_coordinate_to_gain_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.resid10_carrier_coordinate_to_gain_v1"
EXPECTED_PRIOR_SHA256 = "95a75a058309d433058c6ebbcb696b7e94d0c17b91013e097fdfa366fdda4346"
EXPECTED = {
    V12: "ed6afb3455bf5bfeea6e36f65ce33e9199290cb26540ee94da2c42accc785e7c",
    UPSTREAM: "e7a9fbb41a3330bdf7cc7f8e13545ccb3773cb803ca47250f87691f355d2839a",
    V5: "a4cee1818acd6a28e999eda26d0447f94d080b913f4061d0e6dab4914cb3802c",
    SCALAR: "4a55ef3da37b12722fabae41c9caaa7e8284fc0891ba4e15c5cfdeab40323b2d",
    BUILDER: "ae624913c5adfe07cf028acf6549cd5fe2debd4b090c71659218fe158089fe2c",
}
EXPECTED_ROWS_SHA256 = "296c2186f477a6d450bbbb87fda5ba89b999eb4d3ac0dc18e31496ca47d5caf7"


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    calibration_rows, lexical_spec, alpha_by_id, _old_rows, _old_spec, rank1, _dose = upstream.validate_static()
    _ignored, screen_spec, confirm_rank1 = prospective.validate_static()
    rows = fresh.build_rows()
    prior = json.loads(PRIOR.read_text())
    v12 = json.loads(V12.read_text())
    known_v5 = json.loads(V5.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or v12.get("terminal") != "release"
        or known_v5.get("terminal") != "screen"
        or not all(cell["passed"] for cell in known_v5["capability_cells"])
        or len(calibration_rows) != 16
        or len(alpha_by_id) != 16
        or len(rows) != 64
        or fresh.validate_rows(rows) != EXPECTED_ROWS_SHA256
        or rank1["basis"]["sha256"] != confirm_rank1["basis"]["sha256"]
    ):
        raise ExperimentError("candidate, terminal, population, rows, or basis changed")
    return calibration_rows, lexical_spec, alpha_by_id, rows, screen_spec, rank1, known_v5


def direction_for(row, family):
    return affine.direction_for(row, family)


def token_pair(direction):
    current = upstream.TOKEN_IDS["has"] if direction == "present_to_past" else upstream.TOKEN_IDS["had"]
    other = upstream.TOKEN_IDS["had"] if direction == "present_to_past" else upstream.TOKEN_IDS["has"]
    return current, other


def main() -> None:
    calibration_rows, lexical_spec, alpha_by_id, rows, screen_spec, rank1, known_v5 = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_resid10_carrier_coordinate_to_gain_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "calibration_rows": 16,
        "confirmation_rows": 64,
        "feature": "dot(rank1_carrier_q,resid10_source)",
        "lm_head_weights_used_by_feature": False,
        "counted_forwards_max": 25,
        "example_evaluations_max": 360,
        "selected_head_pair_evaluations": 224,
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
    head_ok, head_error = affine.parent.das.verify_head(backend, rows[:8], "resid:18")
    counted_forwards, evaluations, pair_evaluations = 1, 8, 0

    calibration_records = []
    for chunk in affine.parent.producer._chunks(calibration_rows, lexical_spec.batch_size):
        base10, _donor10, _ = affine.parent.das.capture_site(backend, chunk, "resid:10")
        counted_forwards += 2
        evaluations += 2 * len(chunk)
        for i, row in enumerate(chunk):
            feature = float(torch.dot(q, base10[i]))
            calibration_records.append({"row_id": str(row["row_id"]), "direction": row["direction_id"], "resid10_carrier_coordinate": feature, "target_guided_alpha": alpha_by_id[str(row["row_id"])]})
    fits = {}
    for direction in ("present_to_past", "past_to_present"):
        selected = [record for record in calibration_records if record["direction"] == direction]
        fits[direction] = upstream.fit_line([record["resid10_carrier_coordinate"] for record in selected], [record["target_guided_alpha"] for record in selected])

    target_scale = float(rank1["score"]["families"]["target_scale"])
    capability_records, records = [], []
    for family in ("A1", "A2", "P", "C"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in affine.parent.producer._chunks(family_rows, screen_spec.batch_size):
            base10, donor10, _ = affine.parent.das.capture_site(backend, chunk, "resid:10")
            base18, donor18, _ = affine.parent.das.capture_site(backend, chunk, "resid:18")
            counted_forwards += 4
            evaluations += 4 * len(chunk)
            for i, row in enumerate(chunk):
                direction = direction_for(row, family)
                source10 = donor10[i] if family == "P" else base10[i]
                source18 = donor18[i] if family == "P" else base18[i]
                feature = float(torch.dot(q, source10))
                coefficients = fits[direction]
                alpha = coefficients["intercept"] + coefficients["slope"] * feature
                if family in ("A1", "A2"):
                    current_id, other_id = token_pair(direction)
                    base_original = affine.parent.pair_logits(backend, base18[i], current_id, other_id)
                    donor_original = affine.parent.pair_logits(backend, donor18[i], other_id, current_id)
                    base_margin = -(base_original[0] - base_original[1])
                    donor_margin = donor_original[0] - donor_original[1]
                    answer_id, foil_id = other_id, current_id
                    capability_records.extend([
                        {"family": family, "direction": direction, "side": "base", "correct": base_margin < 0.0},
                        {"family": family, "direction": direction, "side": "donor", "correct": donor_margin > 0.0},
                    ])
                    pair_evaluations += 4
                elif family == "P":
                    current_id, other_id = token_pair(direction)
                    original = affine.parent.pair_logits(backend, source18, current_id, other_id)
                    current_margin = original[0] - original[1]
                    base_margin, donor_margin = -current_margin, None
                    answer_id, foil_id = other_id, current_id
                    capability_records.append({"family": family, "direction": direction, "side": "source", "correct": current_margin > 0.0})
                    pair_evaluations += 3
                else:
                    original = affine.parent.pair_logits(backend, source18, row["base_answer_id"], row["base_foil_id"])
                    base_margin, donor_margin = original[0] - original[1], None
                    answer_id, foil_id = row["base_answer_id"], row["base_foil_id"]
                    capability_records.append({"family": family, "direction": direction, "side": "actual_base", "correct": base_margin > 0.0})
                    pair_evaluations += 3
                patched = affine.parent.pair_logits(backend, source18 + alpha * q, answer_id, foil_id)
                patched_margin = patched[0] - patched[1]
                record = {"family": family, "row_id": str(row["row_id"]), "direction": direction, "resid10_carrier_coordinate": feature, "alpha": alpha, "base_margin": base_margin, "patched_margin": patched_margin, "lm_head_weights_used_by_feature": False, "confirmation_resid18_margin_used_to_select_alpha": False, "confirmation_donor_activation_used_to_select_alpha": False, "row_outcome_ids_used_to_select_alpha": False}
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

    summaries = {
        "A1": upstream.summarize([r for r in records if r["family"] == "A1"], "recovery"),
        "A2": upstream.summarize([r for r in records if r["family"] == "A2"], "recovery"),
        "P": upstream.summarize([r for r in records if r["family"] == "P"], "margin_reflection_fraction"),
        "C": upstream.summarize([r for r in records if r["family"] == "C"], "normalized_unrelated_effect"),
    }
    capability_cells = []
    for family in ("A1", "A2", "P", "C"):
        for direction in sorted({r["direction"] for r in capability_records if r["family"] == family}):
            cell = [r for r in capability_records if r["family"] == family and r["direction"] == direction]
            accuracy = sum(r["correct"] for r in cell) / len(cell)
            threshold = 0.75 if family == "C" else 0.85
            capability_cells.append({"family": family, "direction": direction, "correct": sum(r["correct"] for r in cell), "total": len(cell), "accuracy": accuracy, "threshold": threshold, "passed": accuracy >= threshold})
    fit_finite = all(fit["count"] == 8 and all(math.isfinite(fit[key]) for key in ("intercept", "slope", "r2")) for fit in fits.values())
    actuator_ok = all(math.isfinite(r["alpha"]) and r["alpha"] == fits[r["direction"]]["intercept"] + fits[r["direction"]]["slope"] * r["resid10_carrier_coordinate"] and not r["lm_head_weights_used_by_feature"] and not r["confirmation_resid18_margin_used_to_select_alpha"] and not r["confirmation_donor_activation_used_to_select_alpha"] and not r["row_outcome_ids_used_to_select_alpha"] for r in records)
    pred_a = all(cell["passed"] for cell in capability_cells) and head_ok and head_error <= 1.0e-3
    pred_b = fit_finite and actuator_ok and all(fit["r2"] >= 0.50 for fit in fits.values())
    pred_c = all(summaries[f]["mean_recovery"] >= 0.65 and summaries[f]["direction_fraction"] >= 0.75 for f in ("A1", "A2"))
    pred_d = summaries["P"]["mean_margin_reflection_fraction"] >= 0.65 and summaries["P"]["direction_fraction"] >= 0.75
    pred_e = summaries["C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_f = len(calibration_records) == 16 and len(records) == 64 and len({r["row_id"] for r in records}) == 64 and counted_forwards <= 25 and evaluations <= 360 and pair_evaluations == 224
    predictions = {"pred_a_authority_population_capability_and_exactness": pred_a, "pred_b_weight_free_fit_and_actuator_well_formed": pred_b, "pred_c_A_prediction": pred_c, "pred_d_P_generalization": pred_d, "pred_e_C_selectivity": pred_e, "pred_f_exact_coverage_and_price": pred_f}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and fit_finite and actuator_ok and pred_f else "invalid")
    reason = {"screen": "same_carrier_coordinate_reads_upstream_and_writes_downstream", "null": "upstream_reader_and_downstream_writer_coordinates_split", "invalid": "authority_capability_head_basis_fit_actuator_or_coverage_invalid"}[terminal]
    value = {"schema": "aspectual_anchor_resid10_carrier_coordinate_to_gain_result_v1", "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only", "started_utc": started_utc, "finished_utc": affine.parent.scalar_parent.empirical.component_parent.utc_now(), "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256, "basis_sha256": rank1["basis"]["sha256"], "feature_site": "resid:10", "write_site": "resid:18", "feature": "dot(rank1_carrier_q,resid10_source)", "fits": fits, "capability_cells": capability_cells, "head_control": {"passed": head_ok, "max_abs_difference": head_error}, "predictions": predictions, "score": {"families": summaries, "target_scale": target_scale, "counted_forwards": counted_forwards, "example_evaluations": evaluations, "selected_head_pair_evaluations": pair_evaluations, "grid_evaluations": 0, "calibration_record_count": len(calibration_records), "record_count": len(records), "model_backwards": 0, "model_updates": 0, "fit_parameters": 4}, "calibration_records": calibration_records, "intervention_records": records, "terminal": terminal, "reason": reason, "evidence_scope": "weight_free_geometry_screen_on_known_v5_population", "next_action": "freeze on fresh lexicon if screen; otherwise identify a distinct weight-free upstream reader direction"}
    payload = atomic_create_json(OUT, value)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "fits": fits, "families": summaries, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
