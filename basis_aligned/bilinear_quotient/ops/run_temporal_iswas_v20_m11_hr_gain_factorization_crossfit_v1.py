#!/usr/bin/env python3
"""Cross-fit writer, reader, and bilinear models of M11 response gain."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_future_authority_exact_split_alignment_coefficients_folds_finiteness_and_price pred_b_native_hr_coefficients_predict_heldout_gain pred_c_gain_source_is_discriminated pred_d_predicted_gain_closes_most_nonshape_error pred_e_scope_remains_identification_screen_only
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
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v20_m11_hr_gain_factorization_crossfit_v1.json"
SPLIT_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_writer_reader_factor_split_v1_result.json"
TENSOR = ROOT / "circuits/followups/temporal_iswas_v20_m11_writer_reader_factor_split_v1.npz"
LIBRARY = ROOT / "ops/factorial_tensor_state.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v20_m11_hr_gain_factorization_crossfit_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v20_m11_hr_gain_factorization_crossfit_v1"
EXPECTED = {"prior": "3f463aa29edf38bc22f3f20f53bfbc40688e6aeb6680f519a0100d0278d2624e",
    "split_result": "e334a580b72806a3976408cc16efb36aa05d066c3a7db28cb74da8f96c5ccc71",
    "tensor": "0af2a5e263b0a227e1f88a4d91d7b1ac2191e89734d4882458ba5f46b492a5eb",
    "library": "02d291fa5ce70c0807f264a9a303c996bca992d7d337900d5ff9373d0c0885d4"}
MODELS = ("writer", "reader", "product", "joint")
BARS = {"gain_pearson": .75, "gain_r2": .50, "marginal_r2_floor": .40,
        "winner_r2_margin": .15, "raw_q_residual_improvement": .05,
        "winning_raw_q_relative_residual": .25, "fold_coefficient_sign_agreement": 1.0}
PRICE = {"gpu_forwards": 0, "transformer_backwards": 0, "model_updates": 0,
         "least_squares_fits": 8, "heldout_scalar_predictions": 192}
PREDICTION_KEYS = ("pred_a_future_authority_exact_split_alignment_coefficients_folds_finiteness_and_price",
    "pred_b_native_hr_coefficients_predict_heldout_gain", "pred_c_gain_source_is_discriminated",
    "pred_d_predicted_gain_closes_most_nonshape_error", "pred_e_scope_remains_identification_screen_only")

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def raw_metrics(predictions, references):
    p, r = np.stack(predictions).reshape(-1), np.stack(references).reshape(-1)
    row_p, row_r = np.stack(predictions), np.stack(references)
    return {"relative_residual": float(np.linalg.norm(p-r) / max(np.linalg.norm(r), 1e-30)),
            "row_positive_fraction": float(np.mean(np.sum(row_p*row_r, axis=1) > 0)), "count": len(predictions)}

def coefficients(H, R, Q, hs, rs, qs, rows):
    h, r, a, cells = [], [], [], []
    for panel in range(3):
        for row in rows:
            cell = (panel, int(row % 2)); cells.append(cell)
            h.append(state.projection_coefficient(H[panel, row], hs[cell]))
            r.append(state.projection_coefficient(R[panel, row], rs[cell]))
            a.append(state.projection_coefficient(Q[panel, row], qs[cell]))
    return np.asarray(h), np.asarray(r), np.asarray(a), cells

def atomic_write(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")

def main():
    paths = {"prior": PRIOR, "split_result": SPLIT_RESULT, "tensor": TENSOR, "library": LIBRARY}
    observed = {name: sha(path) for name, path in paths.items()}; data = np.load(TENSOR, allow_pickle=False)
    H0, R0, mask, Q = (data[k] for k in ("H", "R", "mask", "Q"))
    H, common_h = state.suffix_align(H0, mask, panel_count=3); R, common_r = state.suffix_align(R0, mask, panel_count=3)
    reconstructed = np.sum(H0.astype(np.float64)*R0.astype(np.float64), axis=2)
    exact_error = float(np.max(np.abs(reconstructed-Q.astype(np.float64))))
    folds = state.balanced_mod4_folds(16)
    authority_ok = (observed == EXPECTED and H0.shape == R0.shape == (4,16,16,4608)
                    and Q.shape == (4,16,4608) and tuple(str(x) for x in data["panels"]) == ("A1","A2","P","C")
                    and common_h == common_r and exact_error <= 1e-6
                    and all(all(np.sum(rows % 2 == d) == 4 for rows in pair for d in (0,1)) for pair in folds.values()))
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False,
           "queue_touched": False, "authority_ok": authority_ok, "common_suffix_positions": common_h,
           "bars": BARS, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok or not all(np.isfinite(x).all() for x in (H,R,Q)):
        raise RuntimeError("H/R gain authority invalid")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter(); fold_reports = {}; pooled = {m: ([], [], [], []) for m in MODELS}
    pooled_constant = ([], []); fold_coefficients = {m: [] for m in MODELS}
    for fold, (fit, test) in folds.items():
        hs = state.fit_factorial_states(H, fit).cell; rs = state.fit_factorial_states(R, fit).cell
        qs = state.fit_factorial_states(Q[:3], fit).cell
        hf, rf, af, _ = coefficients(H,R,Q,hs,rs,qs,fit)
        fitted = state.fit_gain_models(hf,rf,af)
        ht, rt, at, cells = coefficients(H,R,Q,hs,rs,qs,test)
        predicted = state.predict_gain_models(fitted,ht,rt)
        fold_reports[fold] = {"fit_coefficients": {m: fitted[m].tolist() for m in MODELS}, "models": {}}
        constant_vectors, references = [], []
        for index, cell in enumerate(cells):
            constant_vectors.append(qs[cell]); references.append(Q[cell[0], test[index % len(test)]])
        # `cells` is panel-major and each panel uses the same ordered test rows.
        references = [Q[panel,row] for panel in range(3) for row in test]
        pooled_constant[0].extend(constant_vectors); pooled_constant[1].extend(references)
        for model in MODELS:
            vectors = [predicted[model][i] * qs[cell] for i, cell in enumerate(cells)]
            fold_reports[fold]["models"][model] = {"gain_metrics": state.scalar_prediction_metrics(predicted[model],at),
                                                   "raw_q_metrics": raw_metrics(vectors,references)}
            fold_coefficients[model].append(fitted[model].tolist())
            pooled[model][0].extend(predicted[model].tolist()); pooled[model][1].extend(at.tolist())
            pooled[model][2].extend(vectors); pooled[model][3].extend(references)
    constant_raw = raw_metrics(*pooled_constant)
    reports = {m: {"gain_metrics": state.scalar_prediction_metrics(np.asarray(v[0]),np.asarray(v[1])),
                   "raw_q_metrics": raw_metrics(v[2],v[3]), "fold_coefficients": fold_coefficients[m]}
               for m,v in pooled.items()}
    improvements = {m: constant_raw["relative_residual"]-reports[m]["raw_q_metrics"]["relative_residual"] for m in MODELS}
    qualifies_b = {m: reports[m]["gain_metrics"]["pearson"] >= BARS["gain_pearson"]
                   and reports[m]["gain_metrics"]["r2"] >= BARS["gain_r2"]
                   and improvements[m] >= BARS["raw_q_residual_improvement"] for m in MODELS}
    r2 = {m: reports[m]["gain_metrics"]["r2"] for m in MODELS}; best_marginal = max(r2["writer"],r2["reader"])
    branch_candidates = {}
    if r2["writer"] >= BARS["marginal_r2_floor"] and r2["writer"] >= r2["reader"]+BARS["winner_r2_margin"]:
        branch_candidates["writer_gain"] = "writer"
    if r2["reader"] >= BARS["marginal_r2_floor"] and r2["reader"] >= r2["writer"]+BARS["winner_r2_margin"]:
        branch_candidates["reader_gain"] = "reader"
    for model in ("product","joint"):
        if r2[model] >= BARS["gain_r2"] and r2[model] >= best_marginal+BARS["winner_r2_margin"]:
            branch_candidates["bilinear_gain"] = max((branch_candidates.get("bilinear_gain",model),model), key=lambda x:r2[x])
    winner_branch, winner = (max(branch_candidates.items(),key=lambda item:r2[item[1]]) if branch_candidates else (None,None))
    sign_stability = {}
    for model, coefficients_by_fold in fold_coefficients.items():
        slopes = np.asarray(coefficients_by_fold)[:,1:]
        sign_stability[model] = {"fraction": float(np.mean(np.sign(slopes[0]) == np.sign(slopes[1]))),
                                 "stable": bool(np.all(np.sign(slopes[0]) == np.sign(slopes[1])))}
    A = bool(authority_ok and all(np.isfinite(x).all() for x in (H,R,Q))
             and PRICE["least_squares_fits"] == len(folds)*len(MODELS)
             and PRICE["heldout_scalar_predictions"] == len(folds)*24*len(MODELS))
    B = any(qualifies_b.values())
    C = bool(winner is not None and sign_stability[winner]["fraction"] >= BARS["fold_coefficient_sign_agreement"])
    D = bool(winner is not None and reports[winner]["raw_q_metrics"]["relative_residual"] <= BARS["winning_raw_q_relative_residual"])
    E = True; predictions = dict(zip(PREDICTION_KEYS,map(bool,(A,B,C,D,E))))
    terminal = "invalid_instrument" if not A else winner_branch if B and C and D else "gain_not_identified_by_marginal_strength"
    result = {"schema":"temporal_iswas_v20_m11_hr_gain_factorization_crossfit_result_v1",
        "candidate_id":CANDIDATE_ID,"started_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "serial_seconds":time.perf_counter()-started,"authority_sha256":observed,
        "evidence_status":"cpu_heldout_hr_gain_identification_screen_only","common_suffix_positions":common_h,
        "split_reconstruction_max_abs":exact_error,"fold_reports":fold_reports,"constant_raw_q_metrics":constant_raw,
        "pooled_reports":reports,"raw_q_residual_improvement":improvements,"model_pass":qualifies_b,
        "branch_candidates":branch_candidates,"winning_branch":winner_branch,"winning_model":winner,
        "coefficient_sign_stability":sign_stability,"predictions":predictions,"terminal":terminal,
        "bars":BARS,"price":PRICE,"gpu_accessed":False,"v21_outcome_opened":False,"causal_intervention_opened":False}
    atomic_write(OUT,result)
    print(json.dumps({k:result[k] for k in ("constant_raw_q_metrics","pooled_reports","raw_q_residual_improvement",
        "branch_candidates","winning_branch","winning_model","predictions","terminal")},sort_keys=True))

if __name__ == "__main__": main()
