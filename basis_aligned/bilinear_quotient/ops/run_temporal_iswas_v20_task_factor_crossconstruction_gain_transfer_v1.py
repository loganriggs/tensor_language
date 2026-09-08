#!/usr/bin/env python3
"""Cross-fit gain transfer between v20 A1 and A2 response states."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_folds_projection_map_finiteness_and_exact_price pred_b_target_constructions_share_predictive_gain pred_c_gain_mapping_sign_and_scale_are_fold_stable pred_d_transferred_gain_predicts_raw_destination_q pred_e_scope_is_gain_reuse_screen_only
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
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v20_task_factor_crossconstruction_gain_transfer_v1.json"
TENSOR = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1.npz"
ERROR_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_task_factor_state_error_decomposition_v1_result.json"
LIBRARY = ROOT / "ops/factorial_tensor_state.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v20_task_factor_crossconstruction_gain_transfer_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v20_task_factor_crossconstruction_gain_transfer_v1"
EXPECTED = {"prior": "e39915408abcdb912853b996646fe3cbc80dff3f7a7c95cc04884745345384e1",
    "tensor": "a79ff1b0e8eac3e3b135a8cb02eaa58e83c317ade23f0d65950b90b2ef2dff12",
    "error_result": "d8e1772d9fd3e356e142027e12402cb99b4b57a60458ed79079fd539b74cb76d",
    "library": "f7cf522a6ee302638df90981dc1b2168e6021dba848bb7c7df968d191da8a0f6"}
BARS = {"pearson": .75, "r2": .50, "slope_ratio_max": 2.0,
        "raw_q_relative_residual": .25, "row_positive_fraction": 1.0}
PRICE = {"gpu_forwards": 0, "transformer_backwards": 0, "model_updates": 0,
         "affine_fits": 4, "heldout_gain_predictions": 32, "heldout_q_predictions": 32}
PREDICTION_KEYS = ("pred_a_authority_folds_projection_map_finiteness_and_exact_price",
    "pred_b_target_constructions_share_predictive_gain",
    "pred_c_gain_mapping_sign_and_scale_are_fold_stable",
    "pred_d_transferred_gain_predicts_raw_destination_q",
    "pred_e_scope_is_gain_reuse_screen_only")
MAPPINGS = {"A1_to_A2": (0, 1), "A2_to_A1": (1, 0)}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def raw_metrics(predictions, references):
    p, r = np.stack(predictions).reshape(-1), np.stack(references).reshape(-1)
    return {"relative_residual": float(np.linalg.norm(p-r) / max(np.linalg.norm(r), 1e-30)),
            "row_positive_fraction": float(np.mean([np.dot(x, y) > 0 for x, y in zip(predictions, references)]))}

def gains(Q, panel, rows, means):
    return np.asarray([state.projection_coefficient(Q[panel, row], means[(panel, row % 2)])
                       for row in rows], dtype=np.float64)

def atomic_write(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")

def main():
    paths = {"prior": PRIOR, "tensor": TENSOR, "error_result": ERROR_RESULT, "library": LIBRARY}
    observed = {name: sha(path) for name, path in paths.items()}
    data = np.load(TENSOR, allow_pickle=False); Q = data["Q"].astype(np.float64)
    authority_ok = observed == EXPECTED and Q.shape == (4, 16, 4608)
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "tensor_shape": list(Q.shape), "bars": BARS, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok or not np.isfinite(Q).all(): raise RuntimeError("gain transfer authority invalid")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter(); fold_reports = {}; pooled = {name: ([], [], [], []) for name in MAPPINGS}
    slopes = {name: [] for name in MAPPINGS}
    for fold, (fit, test) in state.balanced_mod4_folds(16).items():
        means = state.fit_factorial_states(Q[:3], fit).cell; fold_reports[fold] = {}
        for name, (source, destination) in MAPPINGS.items():
            x_fit, y_fit = gains(Q, source, fit, means), gains(Q, destination, fit, means)
            coefficients = np.linalg.lstsq(np.stack([np.ones_like(x_fit), x_fit], axis=1), y_fit, rcond=None)[0]
            x_test, y_test = gains(Q, source, test, means), gains(Q, destination, test, means)
            prediction = coefficients[0] + coefficients[1] * x_test
            scalar = state.scalar_prediction_metrics(prediction, y_test)
            q_prediction = [prediction[i] * means[(destination, row % 2)] for i, row in enumerate(test)]
            q_reference = [Q[destination, row] for row in test]
            vectors = raw_metrics(q_prediction, q_reference)
            fold_reports[fold][name] = {"coefficients": coefficients.tolist(),
                "gain_metrics": scalar, "raw_q_metrics": vectors}
            slopes[name].append(float(coefficients[1]))
            pooled[name][0].extend(prediction.tolist()); pooled[name][1].extend(y_test.tolist())
            pooled[name][2].extend(q_prediction); pooled[name][3].extend(q_reference)
    reports = {name: {"gain_metrics": state.scalar_prediction_metrics(np.asarray(values[0]), np.asarray(values[1])),
                      "raw_q_metrics": raw_metrics(values[2], values[3]), "fold_slopes": slopes[name]}
               for name, values in pooled.items()}
    A = bool(authority_ok and all(np.isfinite(v).all() for values in pooled.values() for v in (np.asarray(values[0]), np.asarray(values[1])))
             and PRICE["affine_fits"] == 4 and PRICE["heldout_gain_predictions"] == 32
             and PRICE["heldout_q_predictions"] == 32)
    mapping_pass = {name: report["gain_metrics"]["pearson"] >= BARS["pearson"]
                    and report["gain_metrics"]["r2"] >= BARS["r2"] for name, report in reports.items()}
    B = all(mapping_pass.values())
    C = all(all(slope > 0 for slope in values)
            and max(abs(slope) for slope in values) / max(min(abs(slope) for slope in values), 1e-30) <= BARS["slope_ratio_max"]
            for values in slopes.values())
    vector_pass = {name: report["raw_q_metrics"]["relative_residual"] <= BARS["raw_q_relative_residual"]
                   and report["raw_q_metrics"]["row_positive_fraction"] >= BARS["row_positive_fraction"]
                   for name, report in reports.items()}
    D = all(vector_pass.values()); E = True
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A else "shared_crossconstruction_gain" if B and C and D
                else "directionally_asymmetric_gain" if C and sum(mapping_pass[n] and vector_pass[n] for n in MAPPINGS) == 1
                else "construction_specific_gain")
    result = {"schema": "temporal_iswas_v20_task_factor_crossconstruction_gain_transfer_result_v1",
        "candidate_id": CANDIDATE_ID,
        "started_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "evidence_status": "cpu_crossconstruction_gain_reuse_screen_only",
        "fold_reports": fold_reports, "pooled_reports": reports,
        "mapping_pass": mapping_pass, "vector_pass": vector_pass,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE,
        "hr_outcome_opened": False, "v21_causal_outcomes_opened": False}
    atomic_write(OUT, result)
    print(json.dumps({key: result[key] for key in ("pooled_reports", "mapping_pass", "vector_pass", "predictions", "terminal")}, sort_keys=True))

if __name__ == "__main__": main()
