#!/usr/bin/env python3
"""Population-capable third-lexicon test of frozen upstream aspectual gain."""

# BQGATE: EXPERIMENT pred_a_authority_population_capability_and_local_exactness pred_b_frozen_upstream_program pred_c_new_lexicon_A_prediction pred_d_new_lexicon_P_generalization pred_e_new_lexicon_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import aspectual_anchor_upstream_gain_executor as executor
import circuit_candidate_aspectual_fresh_lexicon_v5 as fresh
import run_aspectual_anchor_base_margin_affine_fresh_lexicon_v2 as prospective
import run_aspectual_anchor_resid10_frozen_gain_fresh_lexicon_v1 as v1
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_resid10_frozen_gain_fresh_lexicon_v2.json"
EXECUTOR = ROOT / "ops/aspectual_anchor_upstream_gain_executor.py"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_fresh_lexicon_v5.py"
V1_RESULT = ROOT / "circuits/followups/aspectual_anchor_resid10_frozen_gain_fresh_lexicon_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_resid10_frozen_gain_fresh_lexicon_v2_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.resid10_frozen_gain_fresh_lexicon_v2"
EXPECTED_PRIOR_SHA256 = "d56f0ea2c75461d2ff88df9d7b44e3685e43027ae459a0a4e5eedf6f2c153ba6"
EXPECTED = {EXECUTOR: "13cff25e46bb523e40655e74439c9ee376cc77e337f0548257795d931713e53a", BUILDER: "ae624913c5adfe07cf028acf6549cd5fe2debd4b090c71659218fe158089fe2c", V1_RESULT: "519fe4c9ec41e0ba4fa462a18fe92f18fd2bf1f8ad0ecbe4401ebe27af19b869"}
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
    _old_rows, screen_spec, rank1 = prospective.validate_static()
    rows = fresh.build_rows()
    prior = json.loads(PRIOR.read_text())
    invalid = json.loads(V1_RESULT.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID or prior["frozen_program"]["coefficients"] != v1.COEFFICIENTS or fresh.validate_rows(rows) != EXPECTED_ROWS_SHA256 or len(rows) != 64 or invalid.get("terminal") != "invalid":
        raise ExperimentError("candidate, frozen program, rows, or v1 boundary changed")
    return rows, screen_spec, rank1


def main() -> None:
    rows, screen_spec, rank1 = validate_static()
    dryrun = {"schema": "aspectual_anchor_resid10_frozen_gain_fresh_lexicon_dryrun_v2", "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only", "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS_SHA256, "rows": 64, "capability_policy": {"A_P": 0.85, "C": 0.75, "all_rows_retained": True}, "feature_site": "resid:10", "write_site": "resid:18", "coefficients": v1.COEFFICIENTS, "fixed_token_ids": v1.upstream.TOKEN_IDS, "counted_forwards_max": 25, "example_evaluations_max": 328, "selected_head_pair_evaluations": 224, "grid_evaluations": 0, "model_backwards": 0, "model_updates": 0, "inherited_fit_parameters": 4}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    measured = executor.measure(rows, screen_spec, rank1, coefficients=v1.COEFFICIENTS)
    summaries, records = measured["families"], measured["records"]
    actuator_ok = all(math.isfinite(record["alpha"]) and record["alpha"] == v1.COEFFICIENTS[record["direction"]]["intercept"] + v1.COEFFICIENTS[record["direction"]]["slope"] * record["resid10_unembedding_contrast"] and not record["confirmation_resid18_margin_used_to_select_alpha"] and not record["confirmation_donor_activation_used_to_select_alpha"] and not record["row_target_or_foil_used_to_select_alpha"] for record in records)
    pred_a = measured["capability"] and measured["local_error"] <= 1.0e-4 and measured["head_control"]["passed"] and measured["head_control"]["max_abs_difference"] <= 1.0e-3
    pred_b = actuator_ok
    pred_c = all(summaries[family]["mean_recovery"] >= 0.75 and summaries[family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_d = summaries["P"]["mean_margin_reflection_fraction"] >= 0.75 and summaries["P"]["direction_fraction"] >= 0.75
    pred_e = summaries["C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_f = len(records) == 64 and len({record["row_id"] for record in records}) == 64 and measured["counted_forwards"] <= 25 and measured["example_evaluations"] <= 328 and measured["selected_head_pair_evaluations"] == 224
    predictions = {"pred_a_authority_population_capability_and_local_exactness": pred_a, "pred_b_frozen_upstream_program": pred_b, "pred_c_new_lexicon_A_prediction": pred_c, "pred_d_new_lexicon_P_generalization": pred_d, "pred_e_new_lexicon_C_selectivity": pred_e, "pred_f_exact_coverage_and_price": pred_f}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_f else "invalid")
    reason = {"screen": "frozen_upstream_program_predicts_population_capable_third_lexicon", "null": "frozen_upstream_program_fails_A_P_prediction_or_C_selectivity", "invalid": "authority_population_capability_local_exactness_head_program_or_coverage_invalid"}[terminal]
    result = {"schema": "aspectual_anchor_resid10_frozen_gain_fresh_lexicon_result_v2", "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only", "started_utc": measured["started_utc"], "finished_utc": measured["finished_utc"], "serial_seconds": measured["serial_seconds"], "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS_SHA256, "basis_sha256": rank1["basis"]["sha256"], "executor_sha256": EXPECTED[EXECUTOR], "feature_site": "resid:10", "write_site": "resid:18", "coefficients": v1.COEFFICIENTS, "fixed_token_ids": v1.upstream.TOKEN_IDS, "capability_cells": measured["capability_cells"], "local_unembedding_pair_max_abs_difference": measured["local_error"], "head_control": measured["head_control"], "predictions": predictions, "score": {"families": summaries, "target_scale": measured["target_scale"], "counted_forwards": measured["counted_forwards"], "example_evaluations": measured["example_evaluations"], "selected_head_pair_evaluations": measured["selected_head_pair_evaluations"], "grid_evaluations": 0, "record_count": len(records), "model_backwards": 0, "model_updates": 0, "inherited_fit_parameters": 4}, "intervention_records": records, "terminal": terminal, "reason": reason, "evidence_scope": "prospective_third_lexicon_population_capability_within_tested_constructions", "next_action": "compile upstream gain into program v12 and test syntax or different surface readout" if terminal == "screen" else "add a second upstream v11 variable or retain final-margin controller"}
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "capability": measured["capability_cells"], "families": summaries, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
