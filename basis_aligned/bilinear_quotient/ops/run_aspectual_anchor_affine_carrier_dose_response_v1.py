#!/usr/bin/env python3
"""Dose-response test for the prospectively validated aspectual affine controller."""

# BQGATE: EXPERIMENT pred_a_authority_source_capability_and_exact_head pred_b_frozen_three_dose_actuator pred_c_ordered_causal_dose_response pred_d_calibrated_gain_not_broad_plateau pred_e_all_dose_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import aspectual_anchor_affine_dose_executor as dose_executor
import run_aspectual_anchor_base_margin_affine_carrier_actuation_v1 as affine
import run_aspectual_anchor_base_margin_affine_fresh_lexicon_v2 as prospective
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_affine_carrier_dose_response_v1.json"
EXECUTOR = ROOT / "ops/aspectual_anchor_affine_dose_executor.py"
PROSPECTIVE_RESULT = ROOT / "circuits/followups/aspectual_anchor_base_margin_affine_fresh_lexicon_v2_result.json"
BOUNDARY_AUDIT = ROOT / "circuits/followups/aspectual_anchor_affine_gain_identification_boundary_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_affine_carrier_dose_response_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.affine_carrier_dose_response_v1"
EXPECTED_PRIOR_SHA256 = "a16387d65ec1cd56437fd6db11b46e958b4bce3e8a66d3fe8989a73365e2f343"
EXPECTED = {
    EXECUTOR: "31d129f7b94ba44abb8d833c246b4c2b7aa4821a050bd9b3be7e93b6fcf5eb57",
    PROSPECTIVE_RESULT: "a498295f1986f7ca80379fb193d3d961777e3eda1ef11fcf90515953dfc54921",
    BOUNDARY_AUDIT: "d8862dd13ca290b75fbb52ab611ec0482feb5a8700e836ad5922183490bd0cc3",
}
DOSES = (0.5, 1.0, 1.5)
COUNTED_FORWARDS_MAX = 25
EXAMPLE_EVALUATIONS_MAX = 328


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
    rows, screen_spec, rank1 = prospective.validate_static()
    prior = json.loads(PRIOR.read_text())
    result = json.loads(PROSPECTIVE_RESULT.read_text())
    boundary = json.loads(BOUNDARY_AUDIT.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or tuple(prior["frozen_design"]["dose_multipliers"]) != DOSES
        or prior["frozen_design"]["coefficients"] != affine.COEFFICIENTS
        or prior["frozen_design"]["fixed_token_ids"] != affine.TOKEN_IDS
        or result.get("terminal") != "screen"
        or boundary.get("disposition") != "causal_control_law_not_natural_amplitude_identification"
    ):
        raise ExperimentError("candidate, doses, frozen actuator, prospective result, or boundary changed")
    return rows, screen_spec, rank1


def main() -> None:
    rows, screen_spec, rank1 = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_affine_carrier_dose_response_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "rows": 64,
        "doses": list(DOSES),
        "coefficients": affine.COEFFICIENTS,
        "fixed_token_ids": affine.TOKEN_IDS,
        "counted_forwards_max": COUNTED_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "selected_head_pair_evaluations": 304,
        "record_count": 192,
        "grid_evaluations": 0,
        "model_backwards": 0,
        "model_updates": 0,
        "inherited_fit_parameters": 4,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    measured = dose_executor.measure(rows, screen_spec, rank1, coefficients=affine.COEFFICIENTS, token_ids=affine.TOKEN_IDS, doses=DOSES)
    summaries, records = measured["families"], measured["records"]
    response = {family: {dose: summaries[family][str(dose)]["mean_response"] for dose in DOSES} for family in dose_executor.FAMILIES}
    pred_a = measured["capability"] and measured["head_control"]["passed"] and measured["head_control"]["max_abs_difference"] <= 1.0e-3
    pred_b = (
        len(records) == 192
        and all({record["dose"] for record in records if record["row_id"] == row_id} == set(DOSES) for row_id in {record["row_id"] for record in records})
        and all(math.isfinite(record["alpha"]) and record["applied_alpha"] == record["dose"] * record["alpha"] and not record["confirmation_donor_activation_used_by_actuator"] and not record["row_target_or_foil_used_to_select_alpha"] for record in records)
    )
    pred_c = all(response[family][0.5] < response[family][1.0] < response[family][1.5] and measured["row_ordering_fraction"][family] >= 0.75 for family in ("A1", "A2", "P"))
    calibration = {
        family: {
            "dose_1_error": abs(response[family][1.0] - 1.0),
            "dose_half_error": abs(response[family][0.5] - 1.0),
            "dose_one_half_error": abs(response[family][1.5] - 1.0),
            "half_to_one_half_span": response[family][1.5] - response[family][0.5],
        }
        for family in ("A1", "A2", "P")
    }
    pred_d = all(values["dose_1_error"] + 0.10 <= min(values["dose_half_error"], values["dose_one_half_error"]) and values["half_to_one_half_span"] >= 0.50 for values in calibration.values())
    pred_e = all(response["C"][dose] <= 0.20 for dose in DOSES)
    pred_f = len(records) == 192 and len({(record["row_id"], record["dose"]) for record in records}) == 192 and measured["counted_forwards"] <= COUNTED_FORWARDS_MAX and measured["example_evaluations"] <= EXAMPLE_EVALUATIONS_MAX and measured["selected_head_pair_evaluations"] == 304
    predictions = {
        "pred_a_authority_source_capability_and_exact_head": pred_a,
        "pred_b_frozen_three_dose_actuator": pred_b,
        "pred_c_ordered_causal_dose_response": pred_c,
        "pred_d_calibrated_gain_not_broad_plateau": pred_d,
        "pred_e_all_dose_C_selectivity": pred_e,
        "pred_f_exact_coverage_and_price": pred_f,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_f else "invalid")
    reason = {"screen": "affine_alpha_is_resolved_calibrated_causal_gain_over_registered_doses", "null": "dose_order_calibration_span_or_selectivity_fails", "invalid": "authority_capability_head_dose_identity_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_affine_carrier_dose_response_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": measured["started_utc"],
        "finished_utc": measured["finished_utc"],
        "serial_seconds": measured["serial_seconds"],
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "basis_sha256": rank1["basis"]["sha256"],
        "executor_sha256": EXPECTED[EXECUTOR],
        "doses": list(DOSES),
        "coefficients": affine.COEFFICIENTS,
        "fixed_token_ids": affine.TOKEN_IDS,
        "capability_counts": measured["capability_counts"],
        "head_control": measured["head_control"],
        "predictions": predictions,
        "score": {
            "families": summaries,
            "mean_response": {family: {str(dose): value for dose, value in response[family].items()} for family in response},
            "row_ordering_fraction": measured["row_ordering_fraction"],
            "calibration": calibration,
            "target_scale": measured["target_scale"],
            "counted_forwards": measured["counted_forwards"],
            "example_evaluations": measured["example_evaluations"],
            "selected_head_pair_evaluations": measured["selected_head_pair_evaluations"],
            "grid_evaluations": 0,
            "record_count": len(records),
            "model_backwards": 0,
            "model_updates": 0,
            "inherited_fit_parameters": 4,
        },
        "intervention_records": records,
        "terminal": terminal,
        "reason": reason,
        "next_action": "promote calibrated lexicon-held-out actuator and test upstream quotient gain" if terminal == "screen" else "retain robust-control semantics and move to upstream quotient gain prediction",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "responses": response, "ordering": measured["row_ordering_fraction"], "calibration": calibration, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
