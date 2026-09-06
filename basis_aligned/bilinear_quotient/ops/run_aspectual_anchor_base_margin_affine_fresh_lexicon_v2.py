#!/usr/bin/env python3
"""Runtime-type repair for prospective new-lexicon affine carrier actuation."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_source_capability_and_exact_head pred_b_frozen_input_conditioned_actuator pred_c_new_lexicon_A_prediction pred_d_new_lexicon_P_generalization pred_e_new_lexicon_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import aspectual_anchor_affine_carrier_executor as executor
import run_aspectual_anchor_base_margin_affine_carrier_actuation_v1 as affine
import run_aspectual_anchor_base_margin_affine_fresh_lexicon_v1 as v1
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_base_margin_affine_fresh_lexicon_v2.json"
FAILURE = ROOT / "circuits/followups/aspectual_anchor_base_margin_affine_fresh_lexicon_v1_failure.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_base_margin_affine_fresh_lexicon_v2_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.base_margin_affine_fresh_lexicon_v2"
EXPECTED_PRIOR_SHA256 = "fd45fc1a48667ff411bb9911fae31e6c80127952c789a7b32df6aa0015c73ca4"
EXPECTED_FAILURE_SHA256 = "ce85e47e20d8a0a30792a624bc5a00b57d310fa839b0dc81c1bbb99ef76457f9"


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or sha(FAILURE) != EXPECTED_FAILURE_SHA256:
        raise ExperimentError("v2 prior or v1 failure changed")
    rows, _bad_metadata_spec, rank1 = v1.validate_static()
    _old_rows, screen_spec, _rank1, _v10, _v2, coefficient_error = affine.validate_static()
    prior = json.loads(PRIOR.read_text())
    failure = json.loads(FAILURE.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or prior["frozen_predictor"]["coefficients"] != affine.COEFFICIENTS
        or prior["frozen_predictor"]["fixed_token_ids"] != affine.TOKEN_IDS
        or failure.get("terminal") != "invalid"
        or failure.get("scientific_outcomes_observed") is not False
        or failure.get("population_intervention_forwards_completed") != 0
        or coefficient_error > 1.0e-9
        or not hasattr(screen_spec, "batch_size")
    ):
        raise ExperimentError("candidate, frozen predictor, v1 failure boundary, coefficient, or screen spec changed")
    return rows, screen_spec, rank1


def main() -> None:
    rows, screen_spec, rank1 = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_base_margin_affine_fresh_lexicon_dryrun_v2",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "rows_sha256": v1.EXPECTED_ROWS_SHA256,
        "rows": 64,
        "runtime_spec_type": type(screen_spec).__name__,
        "sole_correction": "validated screen spec replaces metadata-only task spec",
        "coefficients": affine.COEFFICIENTS,
        "fixed_token_ids": affine.TOKEN_IDS,
        "confirmation_donor_activation_used": False,
        "row_target_or_foil_used_to_select_alpha": False,
        "target_guided_alpha_search": False,
        "model_forwards_max": v1.MODEL_FORWARDS_MAX,
        "example_evaluations_max": v1.EXAMPLE_EVALUATIONS_MAX,
        "selected_head_pair_evaluations": 176,
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

    measured = executor.measure(rows, screen_spec, rank1, coefficients=affine.COEFFICIENTS, token_ids=affine.TOKEN_IDS)
    summaries, records = measured["families"], measured["records"]
    pred_a = measured["capability"] and measured["head_control"]["passed"] and measured["head_control"]["max_abs_difference"] <= 1.0e-3
    pred_b = all(
        math.isfinite(record["alpha"])
        and abs(record["alpha"]) <= 10000.0
        and not record["confirmation_donor_activation_used_by_actuator"]
        and not record["row_target_or_foil_used_to_select_alpha"]
        and record["alpha"] == affine.COEFFICIENTS[record["direction"]]["intercept"] + affine.COEFFICIENTS[record["direction"]]["slope"] * record["fixed_has_had_confidence"]
        for record in records
    )
    pred_c = all(summaries[family]["mean_recovery"] >= 0.75 and summaries[family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_d = summaries["P"]["mean_margin_reflection_fraction"] >= 0.75 and summaries["P"]["direction_fraction"] >= 0.75
    pred_e = summaries["C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_f = len(records) == 64 and len({record["row_id"] for record in records}) == 64 and measured["forward_calls"] <= v1.MODEL_FORWARDS_MAX and measured["example_evaluations"] <= v1.EXAMPLE_EVALUATIONS_MAX and measured["selected_head_pair_evaluations"] == 176
    predictions = {
        "pred_a_authority_novelty_source_capability_and_exact_head": pred_a,
        "pred_b_frozen_input_conditioned_actuator": pred_b,
        "pred_c_new_lexicon_A_prediction": pred_c,
        "pred_d_new_lexicon_P_generalization": pred_d,
        "pred_e_new_lexicon_C_selectivity": pred_e,
        "pred_f_exact_coverage_and_price": pred_f,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_f else "invalid")
    reason = {"screen": "frozen_source_margin_gain_predicts_selective_new_lexicon_actuation", "null": "frozen_source_margin_gain_fails_new_lexicon_A_or_P_prediction_or_C_selectivity", "invalid": "authority_novelty_source_capability_head_actuator_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_base_margin_affine_fresh_lexicon_result_v2",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": measured["started_utc"],
        "finished_utc": measured["finished_utc"],
        "serial_seconds": measured["serial_seconds"],
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "rows_sha256": v1.EXPECTED_ROWS_SHA256,
        "basis_sha256": rank1["basis"]["sha256"],
        "executor_sha256": v1.EXPECTED[v1.ENGINE],
        "coefficients": affine.COEFFICIENTS,
        "fixed_token_ids": affine.TOKEN_IDS,
        "runtime_spec_type": type(screen_spec).__name__,
        "capability_counts": measured["capability_counts"],
        "head_control": measured["head_control"],
        "predictions": predictions,
        "score": {
            "families": summaries,
            "target_scale": measured["target_scale"],
            "forward_calls": measured["forward_calls"],
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
        "evidence_scope": "prospective_new_lexicon_within_tested_constructions",
        "next_action": "promote lexicon-held-out actuator scope and test an upstream quotient predictor or different surface readout" if terminal == "screen" else "retain design-seen scope and localize which new lexical contexts break the gain rule",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "capability": measured["capability_counts"], "families": summaries, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
