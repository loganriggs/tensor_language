#!/usr/bin/env python3
"""Prospective new-lexicon test of the frozen aspectual affine carrier actuator."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_source_capability_and_exact_head pred_b_frozen_input_conditioned_actuator pred_c_new_lexicon_A_prediction pred_d_new_lexicon_P_generalization pred_e_new_lexicon_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import aspectual_anchor_affine_carrier_executor as executor
import circuit_candidate_aspectual_fresh_lexicon_v3 as fresh
import run_aspectual_anchor_base_margin_affine_carrier_actuation_v1 as affine
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_base_margin_affine_fresh_lexicon_v1.json"
ENGINE = ROOT / "ops/aspectual_anchor_affine_carrier_executor.py"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_fresh_lexicon_v3.py"
AFFINE_RESULT = ROOT / "circuits/followups/aspectual_anchor_base_margin_affine_carrier_actuation_v1_result.json"
AFFINE_AUDIT = ROOT / "circuits/followups/aspectual_anchor_base_margin_affine_carrier_actuation_v1_instrument_audit_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_base_margin_affine_fresh_lexicon_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.base_margin_affine_fresh_lexicon_v1"
EXPECTED_PRIOR_SHA256 = "ed7c5300c7e7ce867092c6a3698f2546464275001d48dd4a98508775479cfde3"
EXPECTED = {
    ENGINE: "2777c63dd7205ba54e0e728e0cbf1324d90b24f9c638fa96eb0ff971d5404180",
    BUILDER: "4d884af47ec7e9e2b00effc91754948514b8e3bb893a58b6a98c4653ef37ab3f",
    AFFINE_RESULT: "4358d288756b7baf0e6d06377f314024905b17a845f8c5cbe5d490fd1bba2c25",
    AFFINE_AUDIT: "f5f26d0c156467e32d8e5256a22b4d5c9b498fc2ff710b06832b53bc7d21ab8a",
}
EXPECTED_ROWS_SHA256 = "1127a313c7657ef2cace839219ff5cd6366df47151dcf63f89450abc85eaa02e"
MODEL_FORWARDS_MAX = 25
EXAMPLE_EVALUATIONS_MAX = 200


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
    old_rows, _old_spec, rank1, _v10, _v2, coefficient_error = affine.validate_static()
    rows, spec = fresh.build_rows(), fresh.TASK_SPEC
    prior = json.loads(PRIOR.read_text())
    parent_result = json.loads(AFFINE_RESULT.read_text())
    parent_audit = json.loads(AFFINE_AUDIT.read_text())
    old_ids = {str(row["row_id"]) for row in old_rows}
    new_ids = {str(row["row_id"]) for row in rows}
    old_agents = set(affine.parent.fresh_parent.fresh._AGENTS)
    old_periods = set(affine.parent.fresh_parent.fresh._PERIODS)
    novelty = not (old_ids & new_ids) and not (old_agents & set(fresh._AGENTS)) and not (old_periods & set(fresh._PERIODS))
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or prior["frozen_predictor"]["coefficients"] != affine.COEFFICIENTS
        or prior["frozen_predictor"]["fixed_token_ids"] != affine.TOKEN_IDS
        or fresh.validate_rows(rows) != EXPECTED_ROWS_SHA256
        or len(rows) != 64
        or len(new_ids) != 64
        or not novelty
        or coefficient_error > 1.0e-9
        or parent_result.get("terminal") != "invalid"
        or parent_audit.get("scientific_disposition") != "design_seen_base_margin_affine_carrier_screen"
    ):
        raise ExperimentError("candidate, frozen predictor, new rows, novelty, coefficient, or parent disposition changed")
    return rows, spec, rank1


def main() -> None:
    rows, spec, rank1 = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_base_margin_affine_fresh_lexicon_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "rows": 64,
        "new_agents": 16,
        "new_periods": 16,
        "prior_row_id_overlap": 0,
        "coefficients": affine.COEFFICIENTS,
        "fixed_token_ids": affine.TOKEN_IDS,
        "confirmation_donor_activation_used": False,
        "row_target_or_foil_used_to_select_alpha": False,
        "target_guided_alpha_search": False,
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
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

    measured = executor.measure(rows, spec, rank1, coefficients=affine.COEFFICIENTS, token_ids=affine.TOKEN_IDS)
    summaries = measured["families"]
    records = measured["records"]
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
    pred_f = len(records) == 64 and len({record["row_id"] for record in records}) == 64 and measured["forward_calls"] <= MODEL_FORWARDS_MAX and measured["example_evaluations"] <= EXAMPLE_EVALUATIONS_MAX and measured["selected_head_pair_evaluations"] == 176
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
        "schema": "aspectual_anchor_base_margin_affine_fresh_lexicon_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": measured["started_utc"],
        "finished_utc": measured["finished_utc"],
        "serial_seconds": measured["serial_seconds"],
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "basis_sha256": rank1["basis"]["sha256"],
        "executor_sha256": EXPECTED[ENGINE],
        "coefficients": affine.COEFFICIENTS,
        "fixed_token_ids": affine.TOKEN_IDS,
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
