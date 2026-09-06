#!/usr/bin/env python3
"""Fit a four-scalar resid10 reader to the donor-free q_is root gain."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_and_exact_heads pred_b_upstream_affine_fit pred_c_design_seen_A2 pred_d_design_seen_P pred_e_design_seen_C pred_f_exact_coverage_and_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v4 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_rank1_donor_free_margin_reflection_v1 as head


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_resid10_margin_to_root_gain_v1.json"
ROOT_ACTUATOR = ROOT / "circuits/followups/tense_auxiliary_is_was_bracketed_root_actuator_fresh_lexicon_v1_result.json"
Q_IS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v4.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_resid10_margin_to_root_gain_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.resid10_margin_to_root_gain_v1"
TOKEN_IDS = {"is": 318, "was": 373}
EXPECTED_PRIOR_SHA256 = "684845793e76d5cd46b164a00545667d38747f9bc09cbbc4657281a104c91192"
EXPECTED = {
    ROOT_ACTUATOR: "342817d6f102497bcb48272337deb138f7c638c864b1213013516ef1454511cb",
    Q_IS: "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    BUILDER: "1d90b1b7feebcf4eb467b41b9b4b168a6ecc62b3a4c4178a91a502bc7923b74b",
}
EXPECTED_ROWS_SHA256 = "c62c2f1eeb311afad1631f4ccd0077211121a4e493cf772676d59ba33e01f4b2"
MODEL_FORWARDS_MAX = 16
EXAMPLE_EVALUATIONS_MAX = 240


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


def confidence(backend, state, direction):
    current, other = current_ids(direction)
    return float(head.selected_margin(backend, state[None, :], [current], [other])[0])


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
    root, qi = json.loads(ROOT_ACTUATOR.read_text()), json.loads(Q_IS.read_text())
    rows = fresh.build_rows()
    alphas = {record["row_id"]: record["alpha"] for record in root["intervention_records"] if record["family"] == "A1"}
    a1 = [row for row in rows if row["family"] == "A1"]
    ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and prior["authority"]["fixed_token_ids"] == TOKEN_IDS
        and root.get("terminal") == "screen"
        and qi.get("terminal") == "screen"
        and root["basis_sha256"] == qi["basis"]["sha256"]
        and fresh.validate_rows(rows) == EXPECTED_ROWS_SHA256
        and len(alphas) == len(a1) == 16
        and set(alphas) == {str(row["row_id"]) for row in a1}
    )
    if not ok:
        raise ExperimentError("candidate, ids, root, basis, rows, or alpha alignment changed")
    return rows, root, qi, alphas


def main():
    rows, root, qi, alpha_by_id = validate_static()
    plan = {
        "schema": "tense_auxiliary_is_was_resid10_margin_to_root_gain_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "feature_site": "resid:10",
        "write_site": "resid:18", "fixed_token_ids": TOKEN_IDS,
        "calibration_records": 16, "causal_records": 48,
        "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "fit_parameters": 4, "grid_evaluations": 0, "transformer_backwards": 0, "model_updates": 0,
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
    by_family = {family: [row for row in rows if row["family"] == family] for family in ("A1", "A2", "P", "C")}
    head18_ok, head18_error = das.verify_head(backend, by_family["A2"][:8], "resid:18")
    forward_calls, evaluations = 1, 8
    base10, _donor10, _ = das.capture_site(backend, by_family["A1"], "resid:10")
    forward_calls += 2
    evaluations += 32
    full10 = das.head_logits(backend, base10)
    local_error = 0.0
    calibration = []
    for i, row in enumerate(by_family["A1"]):
        direction = direction_for(row)
        current, other = current_ids(direction)
        feature = confidence(backend, base10[i], direction)
        local_error = max(local_error, abs(feature - float(full10[i, current] - full10[i, other])))
        calibration.append({"row_id": str(row["row_id"]), "direction": direction, "resid10_confidence": feature, "root_alpha": float(alpha_by_id[str(row["row_id"])])})
    fits = {}
    for direction in ("present_to_past", "past_to_present"):
        selected = [record for record in calibration if record["direction"] == direction]
        fits[direction] = fit_line([record["resid10_confidence"] for record in selected], [record["root_alpha"] for record in selected])

    records = []
    target_scale = float(qi["score"]["families"]["target_scale"])
    for family in ("A2", "P", "C"):
        family_rows = by_family[family]
        base10, donor10, _ = das.capture_site(backend, family_rows, "resid:10")
        base18, donor18, _ = das.capture_site(backend, family_rows, "resid:18")
        forward_calls += 4
        evaluations += 4 * len(family_rows)
        for i, row in enumerate(family_rows):
            direction = direction_for(row)
            source10 = donor10[i] if family == "P" else base10[i]
            source18 = donor18[i] if family == "P" else base18[i]
            feature = confidence(backend, source10, direction)
            alpha = fits[direction]["intercept"] + fits[direction]["slope"] * feature
            patched = source18 + alpha * q
            target, foil = requested_ids(direction)
            base_target = float(head.selected_margin(backend, source18[None, :], [target], [foil])[0])
            patched_target = float(head.selected_margin(backend, patched[None, :], [target], [foil])[0])
            record = {
                "family": family, "row_id": str(row["row_id"]), "direction": direction,
                "resid10_confidence": feature, "alpha": alpha,
                "base_target_margin": base_target, "patched_target_margin": patched_target,
                "root_search_used": False, "donor_activation_used_to_select_alpha": False,
                "donor_margin_used_to_select_alpha": False, "row_target_or_foil_used_to_select_alpha": False,
            }
            if family == "A2":
                donor_reference = float(head.selected_margin(backend, donor18[i:i + 1], [target], [foil])[0])
                record["donor_reference_margin"] = donor_reference
                record["recovery"] = (patched_target - base_target) / (donor_reference - base_target)
            elif family == "P":
                record["margin_reflection_fraction"] = (patched_target - base_target) / (-2.0 * base_target)
            else:
                before_c = float(head.selected_margin(backend, source18[None, :], [row["base_answer_id"]], [row["base_foil_id"]])[0])
                after_c = float(head.selected_margin(backend, patched[None, :], [row["base_answer_id"]], [row["base_foil_id"]])[0])
                record["normalized_unrelated_effect"] = abs(after_c - before_c) / target_scale
            records.append(record)

    grouped_records = {family: [record for record in records if record["family"] == family] for family in ("A2", "P", "C")}
    summaries = {
        "A2": summarize(grouped_records["A2"], "recovery"),
        "P": summarize(grouped_records["P"], "margin_reflection_fraction"),
        "C": summarize(grouped_records["C"], "normalized_unrelated_effect"),
    }
    pred_a = head18_ok and head18_error <= 1.0e-3 and local_error <= 1.0e-4
    fit_ok = all(fits[direction]["count"] == 8 and all(math.isfinite(fits[direction][key]) for key in ("intercept", "slope", "r2")) for direction in fits)
    identity_ok = all(math.isfinite(record["alpha"]) and record["alpha"] == fits[record["direction"]]["intercept"] + fits[record["direction"]]["slope"] * record["resid10_confidence"] and not record["root_search_used"] and not record["donor_activation_used_to_select_alpha"] and not record["donor_margin_used_to_select_alpha"] and not record["row_target_or_foil_used_to_select_alpha"] for record in records)
    pred_b = fit_ok and all(fits[direction]["r2"] >= 0.50 for direction in fits) and identity_ok
    pred_c = summaries["A2"]["mean_recovery"] >= 0.75 and summaries["A2"]["direction_fraction"] >= 0.75
    pred_d = summaries["P"]["mean_margin_reflection_fraction"] >= 0.75 and summaries["P"]["direction_fraction"] >= 0.75
    pred_e = summaries["C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_f = len(calibration) == 16 and len(records) == 48 and len({record["row_id"] for record in records}) == 48 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    predictions = {
        "pred_a_authority_alignment_and_exact_heads": pred_a,
        "pred_b_upstream_affine_fit": pred_b,
        "pred_c_design_seen_A2": pred_c,
        "pred_d_design_seen_P": pred_d,
        "pred_e_design_seen_C": pred_e,
        "pred_f_exact_coverage_and_price": pred_f,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and fit_ok and identity_ok and pred_f else "invalid")
    reason = {"screen": "resid10_is_was_margin_computes_selective_q_is_write", "null": "affine_fit_or_A2_P_C_effect_misses", "invalid": "authority_alignment_head_fit_identity_or_coverage_invalid"}[terminal]
    result = {
        "schema": "tense_auxiliary_is_was_resid10_margin_to_root_gain_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "basis_sha256": qi["basis"]["sha256"], "fixed_token_ids": TOKEN_IDS,
        "feature_site": "resid:10", "write_site": "resid:18", "fits": fits,
        "head_controls": {"resid10_local_selected_vs_full": {"passed": local_error <= 1.0e-4, "max_abs_difference": local_error}, "resid18_native": {"passed": head18_ok, "max_abs_difference": head18_error}},
        "calibration_records": calibration, "intervention_records": records,
        "score": {"families": summaries, "model_forwards": forward_calls, "example_evaluations": evaluations, "calibration_records": len(calibration), "causal_records": len(records), "fit_parameters": 4, "grid_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
        "predictions": predictions, "terminal": terminal, "reason": reason,
        "next_action": "freeze four coefficients and validate on a fifth disjoint capability-qualified lexicon" if terminal == "screen" else "retain root controller and do not add features or rank",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "fits": fits, "families": summaries, "price": result["score"], "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
