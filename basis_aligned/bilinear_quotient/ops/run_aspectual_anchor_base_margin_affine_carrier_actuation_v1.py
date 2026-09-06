#!/usr/bin/env python3
"""Apply a four-parameter source-margin predictor to the aspectual rank-one carrier."""

# BQGATE: EXPERIMENT pred_a_authority_fit_source_capability_and_exact_head pred_b_fixed_input_conditioned_actuator pred_c_fresh_A_prediction pred_d_fresh_P_generalization pred_e_fresh_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import run_aspectual_anchor_direction_prototype_carrier_actuation_v2 as parent
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_base_margin_affine_carrier_actuation_v1.json"
SCALAR = ROOT / "circuits/followups/aspectual_anchor_rank1_scalar_term_compression_split_v1_result.json"
V2_RESULT = ROOT / "circuits/followups/aspectual_anchor_direction_prototype_carrier_actuation_v2_result.json"
V2_AUDIT = ROOT / "circuits/followups/aspectual_anchor_direction_prototype_carrier_actuation_v2_instrument_audit_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_base_margin_affine_carrier_actuation_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.base_margin_affine_carrier_actuation_v1"
EXPECTED_PRIOR_SHA256 = "9cfdb00f19464c978cd128c800400d6263e064300e517242fdf02dab957fc83d"
EXPECTED = {
    SCALAR: "4a55ef3da37b12722fabae41c9caaa7e8284fc0891ba4e15c5cfdeab40323b2d",
    V2_RESULT: "a740fbcb5dc42aed876294dc980cb5deb6fc8e8fa21090b30db0946597d5ad23",
    V2_AUDIT: "a379c2a25b372e9b734c59cd4cf5af6d8d97a031cb7cdc6652155b8b7437ee23",
}
COEFFICIENTS = {
    "present_to_past": {"intercept": -132.28506295990428, "slope": -2035.2755463854235},
    "past_to_present": {"intercept": -383.97355829906564, "slope": 2426.378924892433},
}
TOKEN_IDS = {"has": 468, "had": 550}
MODEL_FORWARDS_MAX = 25
EXAMPLE_EVALUATIONS_MAX = 200


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_line(records):
    xs = [-record["base_target_margin"] for record in records]
    ys = [record["alpha"] for record in records]
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum((x - x_mean) ** 2 for x in xs)
    return {"intercept": y_mean - slope * x_mean, "slope": slope}


def summarize(records, key):
    values = [record[key] for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError(f"missing/nonfinite {key}")
    return {
        "count": len(values),
        f"mean_{key}": statistics.fmean(values),
        f"mean_absolute_{key}": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    fresh_rows, fresh_spec, rank1, v10 = parent.validate_static()
    lexical_rows, _lexical_spec, _fresh_rows, _fresh_spec, _rank1, _reference = parent.scalar_parent.validate_static()
    prior = json.loads(PRIOR.read_text())
    scalar = json.loads(SCALAR.read_text())
    v2 = json.loads(V2_RESULT.read_text())
    audit = json.loads(V2_AUDIT.read_text())
    selection_ids = {record["row_id"] for record in scalar["term_amplitudes"] if record["phase"] == "selection"}
    direction_by_id = {str(row["row_id"]): row["direction_id"] for row in lexical_rows}
    calibration = [record for record in v10["intervention_records"] if record["kind"] == "A" and record["row_id"] in selection_ids]
    observed = {
        direction: fit_line([record for record in calibration if direction_by_id[record["row_id"]] == direction])
        for direction in COEFFICIENTS
    }
    coefficient_error = max(abs(observed[d][k] - COEFFICIENTS[d][k]) for d in COEFFICIENTS for k in ("intercept", "slope"))
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or prior["frozen_predictor"]["coefficients"] != COEFFICIENTS
        or len(selection_ids) != 16
        or len(calibration) != 16
        or coefficient_error > 1.0e-9
        or audit.get("scientific_disposition") != "fixed_two_scalar_direction_prototypes_null"
        or v2.get("terminal") != "invalid"
    ):
        raise ExperimentError("candidate, fit population, coefficient reconstruction, or v2 disposition changed")
    return fresh_rows, fresh_spec, rank1, v10, v2, coefficient_error


def direction_for(row, family):
    if family in ("A1", "A2"):
        return row["direction_id"]
    return "present_to_past" if row["group_number"] % 2 == 0 else "past_to_present"


def fixed_confidence(backend, state, direction):
    if direction == "present_to_past":
        current_id, other_id = TOKEN_IDS["has"], TOKEN_IDS["had"]
    else:
        current_id, other_id = TOKEN_IDS["had"], TOKEN_IDS["has"]
    current, other = parent.pair_logits(backend, state, current_id, other_id)
    return current - other


def main() -> None:
    fresh_rows, fresh_spec, rank1, v10, v2, coefficient_error = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_base_margin_affine_carrier_actuation_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "rows": 64,
        "coefficients": COEFFICIENTS,
        "fixed_token_ids": TOKEN_IDS,
        "confirmation_donor_activation_used": False,
        "row_target_or_foil_used_to_select_alpha": False,
        "target_guided_alpha_search": False,
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "selected_head_pair_evaluations": 176,
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

    started_utc, started = parent.scalar_parent.empirical.component_parent.utc_now(), time.perf_counter()
    backend = parent.producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    q = torch.tensor(rank1["basis"]["values_column_major"], device=backend.device, dtype=torch.float32)
    if q.shape != (1152,) or abs(float(q.norm()) - 1.0) > 1.0e-4 or hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != rank1["basis"]["sha256"]:
        raise ExperimentError("basis reconstruction failed")
    a1_rows = [row for row in fresh_rows if row["transform_id"] == "A1"]
    head_ok, head_error = parent.das.verify_head(backend, a1_rows[:8], "resid:18")
    forward_calls, evaluations, head_evaluations = 1, 8, 0
    target_scale = float(rank1["score"]["families"]["target_scale"])
    records, capability = [], True

    for family in ("A1", "A2", "P", "C"):
        rows = [row for row in fresh_rows if row["transform_id"] == family]
        for chunk in parent.producer._chunks(rows, fresh_spec.batch_size):
            base, donor, _ = parent.das.capture_site(backend, chunk, "resid:18")
            forward_calls += 2
            evaluations += 2 * len(chunk)
            for i, row in enumerate(chunk):
                direction = direction_for(row, family)
                source = donor[i] if family == "P" else base[i]
                confidence = fixed_confidence(backend, source, direction)
                alpha = COEFFICIENTS[direction]["intercept"] + COEFFICIENTS[direction]["slope"] * confidence
                if family in ("A1", "A2"):
                    current_id = TOKEN_IDS["has"] if direction == "present_to_past" else TOKEN_IDS["had"]
                    other_id = TOKEN_IDS["had"] if direction == "present_to_past" else TOKEN_IDS["has"]
                    if (row["base_answer_id"], row["base_foil_id"], row["donor_answer_id"], row["donor_foil_id"]) != (current_id, other_id, other_id, current_id):
                        raise ExperimentError("A row does not match fixed has/had interface")
                    donor_original = parent.pair_logits(backend, donor[i], other_id, current_id)
                    capability = capability and confidence > 0.0 and donor_original[0] > donor_original[1]
                    answer_id, foil_id = other_id, current_id
                    base_target = (-confidence, 0.0)
                    donor_margin = donor_original[0] - donor_original[1]
                    head_evaluations += 3
                elif family == "P":
                    capability = capability and confidence > 0.0
                    answer_id = TOKEN_IDS["had"] if direction == "present_to_past" else TOKEN_IDS["has"]
                    foil_id = TOKEN_IDS["has"] if direction == "present_to_past" else TOKEN_IDS["had"]
                    base_target, donor_margin = (-confidence, 0.0), None
                    head_evaluations += 2
                else:
                    original = parent.pair_logits(backend, source, row["base_answer_id"], row["base_foil_id"])
                    capability = capability and original[0] > original[1] and confidence > 0.0
                    answer_id, foil_id = row["base_answer_id"], row["base_foil_id"]
                    base_target, donor_margin = (original[0] - original[1], 0.0), None
                    head_evaluations += 3
                patched = parent.pair_logits(backend, source + alpha * q, answer_id, foil_id)
                base_margin, patched_margin = base_target[0], patched[0] - patched[1]
                record = {
                    "family": family,
                    "row_id": str(row["row_id"]),
                    "direction": direction,
                    "fixed_has_had_confidence": confidence,
                    "alpha": alpha,
                    "base_margin": base_margin,
                    "patched_margin": patched_margin,
                    "confirmation_donor_activation_used_by_actuator": False,
                    "row_target_or_foil_used_to_select_alpha": False,
                }
                if family in ("A1", "A2"):
                    record["donor_reference_margin"] = donor_margin
                    record["recovery"] = (patched_margin - base_margin) / (donor_margin - base_margin)
                elif family == "P":
                    record["margin_reflection_fraction"] = (patched_margin - base_margin) / (-2.0 * base_margin)
                else:
                    record["normalized_unrelated_effect"] = abs(patched_margin - base_margin) / target_scale
                records.append(record)
            forward_calls += 1
            evaluations += len(chunk)

    by_family = {family: [record for record in records if record["family"] == family] for family in ("A1", "A2", "P", "C")}
    summaries = {
        "A1": summarize(by_family["A1"], "recovery"),
        "A2": summarize(by_family["A2"], "recovery"),
        "P": summarize(by_family["P"], "margin_reflection_fraction"),
        "C": summarize(by_family["C"], "normalized_unrelated_effect"),
    }
    v10_means = {"A1": v10["score"]["panels"]["fresh_A1"]["mean_recovery"], "A2": v10["score"]["panels"]["fresh_A2"]["mean_recovery"]}
    v2_means = {"A1": v2["score"]["families"]["A1"]["mean_recovery"], "A2": v2["score"]["families"]["A2"]["mean_recovery"], "P": v2["score"]["families"]["P"]["mean_margin_reflection_fraction"]}
    versus_v10 = {family: summaries[family]["mean_recovery"] / v10_means[family] for family in ("A1", "A2")}
    versus_v2 = {family: summaries[family]["mean_recovery"] / v2_means[family] for family in ("A1", "A2")}
    versus_v2["P"] = summaries["P"]["mean_margin_reflection_fraction"] / v2_means["P"]
    pred_a = capability and coefficient_error <= 1.0e-9 and head_ok and head_error <= 1.0e-3
    pred_b = len(COEFFICIENTS) == 2 and sum(len(value) for value in COEFFICIENTS.values()) == 4 and all(math.isfinite(record["alpha"]) and abs(record["alpha"]) <= 10000.0 and not record["confirmation_donor_activation_used_by_actuator"] and not record["row_target_or_foil_used_to_select_alpha"] for record in records)
    pred_c = all(summaries[family]["mean_recovery"] >= 0.75 and summaries[family]["direction_fraction"] >= 0.75 and versus_v10[family] >= 0.65 and versus_v2[family] >= 2.0 for family in ("A1", "A2"))
    pred_d = summaries["P"]["mean_margin_reflection_fraction"] >= 0.75 and summaries["P"]["direction_fraction"] >= 0.75 and versus_v2["P"] >= 2.0
    pred_e = summaries["C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_f = len(records) == 64 and len({record["row_id"] for record in records}) == 64 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX and head_evaluations == 176
    predictions = {
        "pred_a_authority_fit_source_capability_and_exact_head": pred_a,
        "pred_b_fixed_input_conditioned_actuator": pred_b,
        "pred_c_fresh_A_prediction": pred_c,
        "pred_d_fresh_P_generalization": pred_d,
        "pred_e_fresh_C_selectivity": pred_e,
        "pred_f_exact_coverage_and_price": pred_f,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_f else "invalid")
    reason = {"screen": "base_margin_affine_rule_supplies_input_dependent_carrier_gain", "null": "base_margin_affine_rule_fails_A_or_P_prediction_or_C_selectivity", "invalid": "authority_fit_source_capability_head_actuator_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_base_margin_affine_carrier_actuation_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": parent.scalar_parent.empirical.component_parent.utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "basis_sha256": rank1["basis"]["sha256"],
        "coefficient_reconstruction_max_abs": coefficient_error,
        "coefficients": COEFFICIENTS,
        "fixed_token_ids": TOKEN_IDS,
        "head_control": {"passed": head_ok, "max_abs_difference": head_error},
        "predictions": predictions,
        "score": {
            "families": summaries,
            "recovery_fraction_vs_v10": versus_v10,
            "improvement_factor_vs_v2": versus_v2,
            "target_scale": target_scale,
            "forward_calls": forward_calls,
            "example_evaluations": evaluations,
            "selected_head_pair_evaluations": head_evaluations,
            "grid_evaluations": 0,
            "record_count": len(records),
            "model_backwards": 0,
            "model_updates": 0,
            "fit_parameters": 4,
        },
        "intervention_records": records,
        "terminal": terminal,
        "reason": reason,
        "evidence_scope": "design_seen_causal_screen_not_prospective_identification",
        "next_action": "preregister genuinely new text for prospective affine-actuator validation" if terminal == "screen" else "retain v10 target-guided actuator and test a richer upstream state variable",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "families": summaries, "versus_v10": versus_v10, "versus_v2": versus_v2, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
