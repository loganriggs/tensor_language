#!/usr/bin/env python3
"""Cross-fit independently averaged M11 writer/reader factorial states."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_future_authority_exact_split_alignment_folds_finiteness_and_price pred_b_independent_cell_hr_states_predict_heldout_q pred_c_additive_hr_hierarchy_is_nearly_sufficient pred_d_writer_or_reader_structure_is_reporter_stable pred_e_scope_remains_computational_screen_only
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
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v20_m11_hr_factorial_state_program_crossfit_v1.json"
SPLIT_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_writer_reader_factor_split_v1_result.json"
TENSOR = ROOT / "circuits/followups/temporal_iswas_v20_m11_writer_reader_factor_split_v1.npz"
LIBRARY = ROOT / "ops/factorial_tensor_state.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v20_m11_hr_factorial_state_program_crossfit_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v20_m11_hr_factorial_state_program_crossfit_v1"
EXPECTED = {
    "prior": "d961ad31f523573501dea9d4e27155dc4d2c715a6c765a688d4a2ce6356feb70",
    "split_result": "e334a580b72806a3976408cc16efb36aa05d066c3a7db28cb74da8f96c5ccc71",
    "tensor": "0af2a5e263b0a227e1f88a4d91d7b1ac2191e89734d4882458ba5f46b492a5eb",
    "library": "02d291fa5ce70c0807f264a9a303c996bca992d7d337900d5ff9373d0c0885d4",
}
PANELS, MODELS = ("A1", "A2", "P"), ("cell", "additive")
BARS = {"cell_program_pooled_q_cosine": .85, "cell_program_each_cell_mean_q_cosine": .75,
        "cell_program_direction_fraction": 1.0, "cell_program_signed_recovery_min": .50,
        "cell_program_signed_recovery_max": 1.50, "additive_program_pooled_q_cosine": .80,
        "cell_advantage_over_additive_max": .08, "marginal_cell_pooled_cosine": .85,
        "marginal_crossfold_stability_mean": .90}
PRICE = {"gpu_forwards": 0, "transformer_backwards": 0, "model_updates": 0,
         "reporter_folds": 2, "heldout_q_predictions_per_program": 48, "programs": 2}
PREDICTION_KEYS = ("pred_a_future_authority_exact_split_alignment_folds_finiteness_and_price",
    "pred_b_independent_cell_hr_states_predict_heldout_q",
    "pred_c_additive_hr_hierarchy_is_nearly_sufficient",
    "pred_d_writer_or_reader_structure_is_reporter_stable",
    "pred_e_scope_remains_computational_screen_only")

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def unit_rows(values):
    flat = values.reshape(values.shape[:2] + (-1,)).astype(np.float64)
    norms = np.linalg.norm(flat, axis=-1, keepdims=True)
    if np.any(norms <= 0): raise RuntimeError("zero marginal H/R row")
    return (flat / norms).reshape(values.shape)

def vector_metrics(predictions, references):
    p, r = np.stack(predictions), np.stack(references)
    fp, fr = p.reshape(-1), r.reshape(-1)
    pp, rr, dot = float(fp @ fp), float(fr @ fr), float(fp @ fr)
    row_dot = np.sum(p.reshape(len(p), -1) * r.reshape(len(r), -1), axis=1)
    return {"count": int(len(p)), "cosine": dot / np.sqrt(max(pp * rr, 1e-30)),
            "signed_recovery": dot / max(rr, 1e-30),
            "relative_residual": float(np.linalg.norm(fp-fr) / max(np.linalg.norm(fr), 1e-30)),
            "row_positive_fraction": float(np.mean(row_dot > 0))}

def cosine(left, right):
    left, right = np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)
    return float(left @ right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-30)

def atomic_write(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")

def main():
    paths = {"prior": PRIOR, "split_result": SPLIT_RESULT, "tensor": TENSOR, "library": LIBRARY}
    observed = {name: sha(path) for name, path in paths.items()}
    data = np.load(TENSOR, allow_pickle=False)
    H0, R0, mask, Q = (data[k] for k in ("H", "R", "mask", "Q"))
    panels = tuple(str(x) for x in data["panels"])
    H, common_h = state.suffix_align(H0, mask, panel_count=3)
    R, common_r = state.suffix_align(R0, mask, panel_count=3)
    reconstructed = np.sum(H0.astype(np.float64) * R0.astype(np.float64), axis=2)
    exact_error = float(np.max(np.abs(reconstructed-Q.astype(np.float64))))
    folds = state.balanced_mod4_folds(16)
    balanced = all(all(np.sum(rows % 2 == d) == 4 for rows in pair for d in (0, 1))
                   for pair in folds.values())
    authority_ok = (observed == EXPECTED and H0.shape == R0.shape == (4, 16, 16, 4608)
                    and mask.shape == (4, 16, 16) and Q.shape == (4, 16, 4608)
                    and panels == ("A1", "A2", "P", "C") and common_h == common_r
                    and exact_error <= 1e-6 and balanced)
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "common_suffix_positions": common_h, "bars": BARS, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok or not all(np.isfinite(x).all() for x in (H, R, Q)):
        raise RuntimeError("H/R state-program authority invalid")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter(); Hn, Rn = unit_rows(H), unit_rows(R)
    fold_reports, pooled = {}, {model: ([], []) for model in MODELS}
    marginal_pooled = {side: {model: ([], []) for model in MODELS} for side in ("writer", "reader")}
    fitted_marginals = {"writer": {}, "reader": {}}
    cell_keys = [(p, d) for p in range(3) for d in (0, 1)]
    pooled_cells = {model: {f"{PANELS[p]}:{d}": ([], []) for p, d in cell_keys}
                    for model in MODELS}
    for fold, (fit, test) in folds.items():
        hs, rs = state.fit_factorial_states(H, fit), state.fit_factorial_states(R, fit)
        hns, rns = state.fit_factorial_states(Hn, fit), state.fit_factorial_states(Rn, fit)
        fitted_marginals["writer"][fold], fitted_marginals["reader"][fold] = hns, rns
        fold_reports[fold] = {"q_programs": {}, "marginals": {}}
        for model in MODELS:
            program = state.contract_state_program(hs, rs, model)
            per_cell, all_p, all_r = {}, [], []
            for p, d in cell_keys:
                rows = test[test % 2 == d]
                predictions = [program[(p, d)] for _ in rows]
                references = [Q[p, row] for row in rows]
                per_cell[f"{PANELS[p]}:{d}"] = vector_metrics(predictions, references)
                all_p.extend(predictions); all_r.extend(references)
                pooled[model][0].extend(predictions); pooled[model][1].extend(references)
                pooled_cells[model][f"{PANELS[p]}:{d}"][0].extend(predictions)
                pooled_cells[model][f"{PANELS[p]}:{d}"][1].extend(references)
            fold_reports[fold]["q_programs"][model] = {"pooled": vector_metrics(all_p, all_r), "cells": per_cell}
        for side, values, fitted in (("writer", Hn, hns), ("reader", Rn, rns)):
            fold_reports[fold]["marginals"][side] = {}
            for model in MODELS:
                mapping = getattr(fitted, model); all_p, all_r = [], []
                for p, d in cell_keys:
                    rows = test[test % 2 == d]
                    all_p.extend([mapping[(p, d)] for _ in rows]); all_r.extend([values[p, row] for row in rows])
                report = vector_metrics(all_p, all_r)
                fold_reports[fold]["marginals"][side][model] = report
                marginal_pooled[side][model][0].extend(all_p); marginal_pooled[side][model][1].extend(all_r)
    pooled_q = {model: vector_metrics(*values) for model, values in pooled.items()}
    pooled_cell_reports = {model: {cell: vector_metrics(*values) for cell, values in cells.items()}
                           for model, cells in pooled_cells.items()}
    pooled_marginal = {side: {model: vector_metrics(*values) for model, values in models.items()}
                       for side, models in marginal_pooled.items()}
    stability = {}
    fold_names = tuple(folds)
    for side in ("writer", "reader"):
        values = [cosine(fitted_marginals[side][fold_names[0]].cell[key],
                         fitted_marginals[side][fold_names[1]].cell[key]) for key in cell_keys]
        stability[side] = {"mean_cosine": float(np.mean(values)), "minimum_cosine": float(np.min(values)),
                           "cell_cosines": values}
    advantage = pooled_q["cell"]["cosine"] - pooled_q["additive"]["cosine"]
    A = bool(authority_ok and all(np.isfinite(x).all() for x in (H, R, Q))
             and PRICE["reporter_folds"] == len(folds)
             and PRICE["heldout_q_predictions_per_program"] == len(folds) * 3 * 8)
    B = bool(pooled_q["cell"]["cosine"] >= BARS["cell_program_pooled_q_cosine"]
             and BARS["cell_program_signed_recovery_min"] <= pooled_q["cell"]["signed_recovery"] <= BARS["cell_program_signed_recovery_max"]
             and pooled_q["cell"]["row_positive_fraction"] >= BARS["cell_program_direction_fraction"]
             and all(report["cosine"] >= BARS["cell_program_each_cell_mean_q_cosine"]
                     for report in pooled_cell_reports["cell"].values()))
    C = bool(pooled_q["additive"]["cosine"] >= BARS["additive_program_pooled_q_cosine"]
             and advantage <= BARS["cell_advantage_over_additive_max"])
    stable_side = {side: pooled_marginal[side]["cell"]["cosine"] >= BARS["marginal_cell_pooled_cosine"]
                   and stability[side]["mean_cosine"] >= BARS["marginal_crossfold_stability_mean"]
                   for side in ("writer", "reader")}
    D, E = any(stable_side.values()), True
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A else "factorized_additive_weight_state_program" if B and C and D
                else "factorized_coupled_cell_program" if B and D else "positionwise_or_document_covariance_required")
    result = {"schema": "temporal_iswas_v20_m11_hr_factorial_state_program_crossfit_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": observed,
        "evidence_status": "cpu_heldout_hr_weight_tensor_program_screen_only",
        "common_suffix_positions": common_h, "split_reconstruction_max_abs": exact_error,
        "fold_reports": fold_reports, "pooled_q_programs": pooled_q, "pooled_q_cells": pooled_cell_reports,
        "pooled_marginal_reports": pooled_marginal, "marginal_cell_crossfold_stability": stability,
        "stable_marginal_side": stable_side, "cell_cosine_advantage_over_additive": float(advantage),
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE,
        "gpu_accessed": False, "v21_outcome_opened": False, "causal_intervention_opened": False}
    atomic_write(OUT, result)
    print(json.dumps({k: result[k] for k in ("pooled_q_programs", "pooled_marginal_reports",
        "marginal_cell_crossfold_stability", "stable_marginal_side", "predictions", "terminal")}, sort_keys=True))

if __name__ == "__main__": main()
