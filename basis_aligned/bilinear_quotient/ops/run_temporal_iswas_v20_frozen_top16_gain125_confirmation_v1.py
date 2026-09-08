#!/usr/bin/env python3
"""Prospective v20 confirmation of the frozen top-16, gain-1.25 M11 program."""

# BQGATE: EXPERIMENT pred_a_authority_formula_replays_hooks_finiteness_and_exact_price pred_b_fixed_program_recovers_both_pristine_v20_target_panels pred_c_program_is_construction_stable_and_selective pred_d_finite_dose_response_is_locally_linear pred_e_no_v20_fit_reorder_dose_or_postselection
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v20 as fresh
import circuit_fast_screen_producer as producer
import run_temporal_iswas_top16_fixed_gain_redundancy_compiler_v1 as compiler
import run_temporal_iswas_v18_frozen_shared_tensor_transfer_v1 as transfer


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v20_frozen_top16_gain125_confirmation_v1.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v20_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v20.py"
TASK_RESULT = ROOT / "circuits/followups/temporal_iswas_m11_crossconstruction_task_tangent_factor_order_v1_result.json"
CALIBRATION = ROOT / "circuits/followups/temporal_iswas_top16_fixed_gain_redundancy_compiler_v1_result.json"
COMPILER = ROOT / "ops/run_temporal_iswas_top16_fixed_gain_redundancy_compiler_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v20_frozen_top16_gain125_confirmation_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v20_frozen_top16_gain125_confirmation_v1"
EXPECTED = {
    "prior": "3517e4dee9016f5470a7952b436948df5b3982c8b8159ffb19fd16a1bdf3b415",
    "capability": "d03534d18f9d3662e5765f7d79294f9ea5f45d09f59d891a093c1739bd20c2ef",
    "builder": "cafe9120c4ba8fdbeff27c5a2d05a247c2000641fd401529a4ca0a089137d1d9",
    "task_result": "e0af6cc965d2d6a560dbde548e191268c77fcac7a5817ad54ea2ccaefe7050b7",
    "calibration": "a5b493763cea7fb0270f179a2f7e4cf5e8ca45354c037844d9d77ff6859d9b0e",
    "compiler": "04c3f2877531b8c33aadc3dae953139ca41bee6a555a541b17fcd1d70117acba",
}
PANELS, TARGETS, CONTROLS = ("A1", "A2", "P", "C"), ("A1", "A2"), ("P", "C")
PRICE = {"checkpoint_loads": 1, "model_forwards": 32, "sequence_evaluations": 512,
         "scored_token_positions": 1024, "panels": 4, "forwards_per_panel": 8,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
BARS = {"replay": 1e-5, "formula": 1e-5, "output_rse": 1e-8,
        "recovery_low": .80, "recovery_high": 1.20, "cosine": .95,
        "direction": .875, "recovery_gap": .25, "control_ratio": .25,
        "linearity_relative_vector_error": .10}
PREDICTION_KEYS = (
    "pred_a_authority_formula_replays_hooks_finiteness_and_exact_price",
    "pred_b_fixed_program_recovers_both_pristine_v20_target_panels",
    "pred_c_program_is_construction_stable_and_selective",
    "pred_d_finite_dose_response_is_locally_linear",
    "pred_e_no_v20_fit_reorder_dose_or_postselection",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    paths = {"prior": PRIOR, "capability": CAPABILITY, "builder": BUILDER,
             "task_result": TASK_RESULT, "calibration": CALIBRATION, "compiler": COMPILER}
    observed = {name: sha(path) for name, path in paths.items()}
    capability = json.loads(CAPABILITY.read_text())
    task_result = json.loads(TASK_RESULT.read_text())
    calibration = json.loads(CALIBRATION.read_text())
    all_rows = fresh.build_rows()
    jointly = {family: set(capability["jointly_capable_row_ids"][family]) for family in TARGETS}
    rows = {family: [row for row in all_rows if row["transform_id"] == family and (
        family in CONTROLS or row["row_id"] in jointly[family])] for family in PANELS}
    frozen = calibration["frozen_program"]
    order_tuple = tuple(int(index) for index in frozen["factor_indices"])
    authority_ok = bool(observed == EXPECTED
        and capability.get("terminal") == "screen" and capability.get("causal_outcomes_opened") is False
        and task_result.get("terminal") == "compact_task_read_factor_program"
        and calibration.get("terminal") == "fixed_gain_calibration_screen"
        and calibration.get("evidence_status") == "retrospective_calibration_only_not_confirmation"
        and order_tuple == tuple(int(index) for index in task_result["stable_factor_order_first256"][:16])
        and float(frozen["gain"]) == compiler.GAIN == 1.25
        and all(len(panel) == 16 for panel in rows.values()))
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "rows": {key: len(value) for key, value in rows.items()},
           "frozen_program": frozen, "price": PRICE, "evidence_status": "prospective_confirmation"}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("v20 fixed-program confirmation authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda")
    chosen = backend.torch.tensor(order_tuple, dtype=backend.torch.long, device=backend.device)
    reports, effects, instruments = {}, {}, {}
    with backend.torch.no_grad():
        for family in PANELS:
            reports[family], panel_effects, instruments[family] = compiler.panel_run(
                backend, rows[family], chosen)
            effects[family] = {key: value.tolist() for key, value in panel_effects.items()}
    target_rms = transfer.rms(np.concatenate([
        np.asarray(effects[family]["top16_gain125"], dtype=np.float64) for family in TARGETS]))
    control_ratios = {family: transfer.rms(effects[family]["top16_gain125"])
                      / max(target_rms, 1e-30) for family in CONTROLS}
    recovery = {family: reports[family]["top16_gain125"]["signed_recovery"] for family in TARGETS}
    recovery_gap = abs(recovery["A1"] - recovery["A2"])
    linearity = {}
    for family in TARGETS:
        native = np.asarray(effects[family]["top16_native"], dtype=np.float64)
        gained = np.asarray(effects[family]["top16_gain125"], dtype=np.float64)
        linearity[family] = float(np.linalg.norm(gained - compiler.GAIN * native)
                                  / max(np.linalg.norm(gained), 1e-30))
    A = bool(authority_ok and all(compiler.calls_ok(value["calls"]) for value in instruments.values())
        and all(value["writer_replay_max_abs_margin_error"] <= BARS["replay"]
                and value["all_hidden_margin_max_abs_error"] <= BARS["formula"]
                and value["all_hidden_output_relative_squared_error"] <= BARS["output_rse"]
                for value in instruments.values())
        and finite({"reports": reports, "controls": control_ratios, "linearity": linearity})
        and PRICE["model_forwards"] == PRICE["panels"] * PRICE["forwards_per_panel"]
        and PRICE["sequence_evaluations"] == PRICE["model_forwards"] * 16)
    B = all(BARS["recovery_low"] <= recovery[family] <= BARS["recovery_high"]
        and reports[family]["top16_gain125"]["cosine"] >= BARS["cosine"]
        and reports[family]["top16_gain125"]["direction_agreement"] >= BARS["direction"]
        for family in TARGETS)
    C = recovery_gap <= BARS["recovery_gap"] and all(
        ratio <= BARS["control_ratio"] for ratio in control_ratios.values())
    D = all(error <= BARS["linearity_relative_vector_error"] for error in linearity.values())
    E = True
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A else "fixed_program_nonselective_or_unstable" if not C
        else "finite_dose_nonlinear" if not D else "v20_fixed_program_confirmed" if B
        else "global_scalar_confirmation_failed")
    result = {"schema": "temporal_iswas_v20_frozen_top16_gain125_confirmation_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "evidence_status": "prospective_confirmation", "frozen_program": frozen,
        "v20_fit_updates": 0, "v20_selected_parameters": 0,
        "reports": reports, "effect_vectors": effects, "target_recovery": recovery,
        "target_recovery_gap": recovery_gap, "control_normalized_rms_ratios": control_ratios,
        "finite_dose_relative_vector_error": linearity, "instrument": instruments,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("evidence_status", "frozen_program",
        "target_recovery", "target_recovery_gap", "control_normalized_rms_ratios",
        "finite_dose_relative_vector_error", "reports", "instrument", "predictions",
        "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
