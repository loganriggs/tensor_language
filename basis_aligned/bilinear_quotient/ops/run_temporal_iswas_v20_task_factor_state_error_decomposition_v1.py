#!/usr/bin/env python3
"""Decompose held-out raw Q state error into amplitude and shape terms."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_folds_formula_finiteness_and_exact_price pred_b_remaining_cell_error_is_mainly_scalar_amplitude pred_c_decomposition_is_exact_and_oracle_sign_is_stable pred_d_scope_remains_diagnostic_only
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import factorial_tensor_state as state

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v20_task_factor_state_error_decomposition_v1.json"
TENSOR = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1.npz"
RAW_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_task_factor_raw_state_program_crossfit_v1_result.json"
LIBRARY = ROOT / "ops/factorial_tensor_state.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v20_task_factor_state_error_decomposition_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v20_task_factor_state_error_decomposition_v1"
EXPECTED = {"prior": "f00a955cd6d2cdc300df169c6c73063396add0c6b24f173fe3f77d61de70a598",
    "tensor": "a79ff1b0e8eac3e3b135a8cb02eaa58e83c317ade23f0d65950b90b2ef2dff12",
    "raw_result": "730575d554ef69ed31e4d19e1fa0612c33defb69daa7ce7ff45a0713ea1a28d2",
    "library": "f0ce2a482da038abe14c98c2521675710cbfcc0c678feda089ae43a83641833e"}
BARS = {"amplitude_fraction_of_fixed_sse": .50, "shape_dominance_fraction": .60,
        "oracle_scalar_relative_residual": .25, "decomposition_relative_error": 1e-10,
        "oracle_positive_fraction": 1.0}
PRICE = {"gpu_forwards": 0, "transformer_backwards": 0, "model_updates": 0,
         "fitted_cell_vectors": 12, "heldout_oracle_scalar_reads": 48}
PREDICTION_KEYS = ("pred_a_authority_folds_formula_finiteness_and_exact_price",
    "pred_b_remaining_cell_error_is_mainly_scalar_amplitude",
    "pred_c_decomposition_is_exact_and_oracle_sign_is_stable",
    "pred_d_scope_remains_diagnostic_only")

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_write(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")

def main():
    paths = {"prior": PRIOR, "tensor": TENSOR, "raw_result": RAW_RESULT, "library": LIBRARY}
    observed = {name: sha(path) for name, path in paths.items()}
    data = np.load(TENSOR, allow_pickle=False); Q = data["Q"].astype(np.float64)
    authority_ok = observed == EXPECTED and Q.shape == (4, 16, 4608)
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "tensor_shape": list(Q.shape), "bars": BARS, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok or not np.isfinite(Q).all(): raise RuntimeError("error decomposition authority invalid")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter(); fold_reports = {}
    totals = {"fixed_sse": 0.0, "shape_sse": 0.0, "amplitude_sse": 0.0, "reference_sse": 0.0}
    amplitudes = []
    for label, (fit, test) in state.balanced_mod4_folds(16).items():
        cells = state.fit_factorial_states(Q[:3], fit).cell
        fold = {key: 0.0 for key in totals}; fold_amplitudes = []
        for panel in range(3):
            for row in test:
                mean = cells[(panel, row % 2)]; magnitude = float(np.linalg.norm(mean))
                if magnitude <= 0: raise RuntimeError("zero fitted cell mean")
                direction = mean / magnitude; y = Q[panel, row]
                amplitude = float(y @ direction); orthogonal = y - amplitude * direction
                fixed = y - mean
                fold["fixed_sse"] += float(fixed @ fixed)
                fold["shape_sse"] += float(orthogonal @ orthogonal)
                fold["amplitude_sse"] += float((amplitude - magnitude) ** 2)
                fold["reference_sse"] += float(y @ y)
                fold_amplitudes.append(amplitude); amplitudes.append(amplitude)
        for key in totals: totals[key] += fold[key]
        fold["decomposition_relative_error"] = abs(fold["fixed_sse"] - fold["shape_sse"] - fold["amplitude_sse"]) / max(fold["fixed_sse"], 1e-30)
        fold["amplitude_fraction_of_fixed_sse"] = fold["amplitude_sse"] / max(fold["fixed_sse"], 1e-30)
        fold["oracle_scalar_relative_residual"] = np.sqrt(fold["shape_sse"] / max(fold["reference_sse"], 1e-30))
        fold["oracle_positive_fraction"] = float(np.mean(np.asarray(fold_amplitudes) > 0))
        fold_reports[label] = fold
    decomposition_error = abs(totals["fixed_sse"] - totals["shape_sse"] - totals["amplitude_sse"]) / max(totals["fixed_sse"], 1e-30)
    summary = {**totals,
        "decomposition_relative_error": decomposition_error,
        "amplitude_fraction_of_fixed_sse": totals["amplitude_sse"] / max(totals["fixed_sse"], 1e-30),
        "shape_fraction_of_fixed_sse": totals["shape_sse"] / max(totals["fixed_sse"], 1e-30),
        "oracle_scalar_relative_residual": float(np.sqrt(totals["shape_sse"] / max(totals["reference_sse"], 1e-30))),
        "oracle_positive_fraction": float(np.mean(np.asarray(amplitudes) > 0))}
    A = bool(authority_ok and all(np.isfinite(list(report.values())).all() for report in fold_reports.values())
             and PRICE["fitted_cell_vectors"] == 2 * 6 and PRICE["heldout_oracle_scalar_reads"] == len(amplitudes))
    B = bool(summary["amplitude_fraction_of_fixed_sse"] >= BARS["amplitude_fraction_of_fixed_sse"]
             and summary["oracle_scalar_relative_residual"] <= BARS["oracle_scalar_relative_residual"])
    C = bool(summary["decomposition_relative_error"] <= BARS["decomposition_relative_error"]
             and summary["oracle_positive_fraction"] >= BARS["oracle_positive_fraction"])
    D = True; predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A or not C else "scalar_amplitude_is_next_state_variable" if B
                else "additional_shape_is_required" if summary["shape_fraction_of_fixed_sse"] >= BARS["shape_dominance_fraction"]
                or summary["oracle_scalar_relative_residual"] > BARS["oracle_scalar_relative_residual"] else "mixed_error")
    result = {"schema": "temporal_iswas_v20_task_factor_state_error_decomposition_result_v1",
        "candidate_id": CANDIDATE_ID,
        "started_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "evidence_status": "cpu_heldout_error_attribution_diagnostic_only",
        "fold_reports": fold_reports, "pooled": summary, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE,
        "heldout_oracle_amplitude_used_by_executable": False,
        "hr_outcome_opened": False, "v21_causal_outcomes_opened": False}
    atomic_write(OUT, result)
    print(json.dumps({key: result[key] for key in ("pooled", "predictions", "terminal")}, sort_keys=True))

if __name__ == "__main__": main()
