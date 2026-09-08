#!/usr/bin/env python3
"""Cross-fit raw factorial states of the v20 task-functional tensor."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_raw_formula_balanced_folds_finiteness_and_exact_price pred_b_raw_cell_states_predict_heldout_q_magnitude pred_c_raw_additive_hierarchy_is_nearly_sufficient pred_d_scope_is_upper_bound_screen_only
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
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v20_task_factor_raw_state_program_crossfit_v1.json"
TENSOR_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1_result.json"
TENSOR = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1.npz"
NORMALIZED_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_task_factor_factorial_prototype_crossfit_v1_result.json"
LIBRARY = ROOT / "ops/factorial_tensor_state.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v20_task_factor_raw_state_program_crossfit_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v20_task_factor_raw_state_program_crossfit_v1"
EXPECTED = {
    "prior": "d52d00d2f101fd3e1ac611b75d644dbded97c40e1739a5d2bb87298df73e0b1a",
    "tensor_result": "da97abea6f3dae8721c2cda383e0b7af844217051e199046a23c6b9c264da396",
    "tensor": "a79ff1b0e8eac3e3b135a8cb02eaa58e83c317ade23f0d65950b90b2ef2dff12",
    "normalized_result": "08d2e08094ed1ae28e8b0975d6a4261a5f85cfbf135b9e597700d9208ae387d1",
    "library": "f0ce2a482da038abe14c98c2521675710cbfcc0c678feda089ae43a83641833e",
}
PANELS, MODELS = ("A1", "A2", "P"), ("cell", "additive")
BARS = {"cell_pooled_cosine": .85, "cell_pooled_recovery_min": .50,
        "cell_pooled_recovery_max": 1.50, "cell_pooled_relative_residual": .75,
        "cell_each_cell_cosine": .75, "cell_row_positive_fraction": 1.0,
        "additive_pooled_cosine": .80, "additive_pooled_relative_residual": .85,
        "cell_cosine_advantage_max": .08, "additive_residual_disadvantage_max": .15}
PRICE = {"gpu_forwards": 0, "transformer_backwards": 0, "model_updates": 0,
         "fitted_state_vectors": 24, "heldout_vector_predictions": 96}
PREDICTION_KEYS = ("pred_a_authority_raw_formula_balanced_folds_finiteness_and_exact_price",
    "pred_b_raw_cell_states_predict_heldout_q_magnitude",
    "pred_c_raw_additive_hierarchy_is_nearly_sufficient",
    "pred_d_scope_is_upper_bound_screen_only")

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics(predictions, references):
    predictions, references = np.stack(predictions), np.stack(references)
    flat_p, flat_r = predictions.reshape(-1), references.reshape(-1)
    rr, pp, dot = float(flat_r @ flat_r), float(flat_p @ flat_p), float(flat_p @ flat_r)
    return {"count": int(len(predictions)), "cosine": dot / np.sqrt(max(pp * rr, 1e-30)),
            "signed_recovery": dot / max(rr, 1e-30),
            "relative_residual": float(np.linalg.norm(flat_p - flat_r) / max(np.linalg.norm(flat_r), 1e-30)),
            "row_positive_fraction": float(np.mean(np.sum(predictions * references, axis=1) > 0))}

def atomic_write(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")

def main():
    paths = {"prior": PRIOR, "tensor_result": TENSOR_RESULT, "tensor": TENSOR,
             "normalized_result": NORMALIZED_RESULT, "library": LIBRARY}
    observed = {name: sha(path) for name, path in paths.items()}
    data = np.load(TENSOR, allow_pickle=False); Q = data["Q"].astype(np.float64)
    authority_ok = (observed == EXPECTED and Q.shape == (4, 16, 4608)
                    and tuple(str(x) for x in data["panels"]) == ("A1", "A2", "P", "C"))
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "tensor_shape": list(Q.shape), "bars": BARS, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok or not np.isfinite(Q).all(): raise RuntimeError("raw Q authority invalid")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter(); folds = state.balanced_mod4_folds(16)
    fold_reports, pooled_values = {}, {model: ([], []) for model in MODELS}
    all_cells = [(panel, direction) for panel in range(3) for direction in (0, 1)]
    for label, (fit, test) in folds.items():
        fitted = state.fit_factorial_states(Q[:3], fit); fold_reports[label] = {}
        for model in MODELS:
            mapping = getattr(fitted, model); per_cell, all_p, all_r = {}, [], []
            for cell in all_cells:
                panel, direction = cell; indices = [row for row in test if row % 2 == direction]
                predictions = [mapping[cell] for _row in indices]
                references = [Q[panel, row] for row in indices]
                per_cell[f"{PANELS[panel]}:{direction}"] = metrics(predictions, references)
                all_p.extend(predictions); all_r.extend(references)
                pooled_values[model][0].extend(predictions); pooled_values[model][1].extend(references)
            fold_reports[label][model] = {"pooled": metrics(all_p, all_r), "cells": per_cell}
    pooled = {model: metrics(*values) for model, values in pooled_values.items()}
    cosine_advantage = pooled["cell"]["cosine"] - pooled["additive"]["cosine"]
    residual_disadvantage = pooled["additive"]["relative_residual"] - pooled["cell"]["relative_residual"]
    A = bool(authority_ok and np.isfinite(Q).all()
             and PRICE["fitted_state_vectors"] == len(folds) * 12
             and PRICE["heldout_vector_predictions"] == len(folds) * len(MODELS) * 3 * 8)
    B = bool(pooled["cell"]["cosine"] >= BARS["cell_pooled_cosine"]
             and BARS["cell_pooled_recovery_min"] <= pooled["cell"]["signed_recovery"] <= BARS["cell_pooled_recovery_max"]
             and pooled["cell"]["relative_residual"] <= BARS["cell_pooled_relative_residual"]
             and pooled["cell"]["row_positive_fraction"] >= BARS["cell_row_positive_fraction"]
             and all(fold_reports[f]["cell"]["cells"][c]["cosine"] >= BARS["cell_each_cell_cosine"]
                     for f in folds for c in fold_reports[f]["cell"]["cells"]))
    C = bool(pooled["additive"]["cosine"] >= BARS["additive_pooled_cosine"]
             and pooled["additive"]["relative_residual"] <= BARS["additive_pooled_relative_residual"]
             and cosine_advantage <= BARS["cell_cosine_advantage_max"]
             and residual_disadvantage <= BARS["additive_residual_disadvantage_max"])
    D = True; predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A else "raw_additive_state_program" if B and C
                else "raw_coupled_cell_program" if B else "amplitude_is_document_continuous")
    result = {"schema": "temporal_iswas_v20_task_factor_raw_state_program_crossfit_result_v1",
        "candidate_id": CANDIDATE_ID,
        "started_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "evidence_status": "cpu_raw_q_state_program_upper_bound_screen_only",
        "fold_reports": fold_reports, "pooled_reports": pooled,
        "cell_cosine_advantage": float(cosine_advantage),
        "additive_residual_disadvantage": float(residual_disadvantage),
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE,
        "hr_outcome_opened": False, "v21_causal_outcomes_opened": False}
    atomic_write(OUT, result)
    print(json.dumps({key: result[key] for key in ("pooled_reports", "cell_cosine_advantage",
        "additive_residual_disadvantage", "predictions", "terminal")}, sort_keys=True))

if __name__ == "__main__": main()
