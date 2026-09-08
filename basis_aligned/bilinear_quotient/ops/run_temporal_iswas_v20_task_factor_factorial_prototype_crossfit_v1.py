#!/usr/bin/env python3
"""Cross-fit a factorial discrete-state model of the v20 response tensor."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_identity_balanced_folds_formula_finiteness_and_exact_price pred_b_six_cell_model_predicts_heldout_restricted_tensor pred_c_additive_hierarchy_explains_cells_without_material_interaction pred_d_cell_identities_are_crossfit_stable pred_e_scope_is_heldout_structure_screen_only
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v20_task_factor_factorial_prototype_crossfit_v1.json"
TENSOR_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1_result.json"
TENSOR = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1.npz"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v20.py"
SIGNED_NULL = ROOT / "circuits/followups/temporal_iswas_v20_task_factor_signed_prototype_crossfit_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v20_task_factor_factorial_prototype_crossfit_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v20_task_factor_factorial_prototype_crossfit_v1"
EXPECTED = {
    "prior": "df23d7178c4a1e9de1c374bd12563d3ceb4c09e06558e15bd07522b9db89ffd9",
    "tensor_result": "da97abea6f3dae8721c2cda383e0b7af844217051e199046a23c6b9c264da396",
    "tensor": "a79ff1b0e8eac3e3b135a8cb02eaa58e83c317ade23f0d65950b90b2ef2dff12",
    "builder": "cafe9120c4ba8fdbeff27c5a2d05a247c2000641fd401529a4ca0a089137d1d9",
    "signed_null": "781f960478c66c836ba91710d97d0634e2bb01529c7eeb55e200f81790438214",
}
PANELS, MODELS = ("A1", "A2", "P"), ("global", "direction", "panel", "additive", "cell")
BARS = {"cell_pooled_cosine": .90, "cell_mean_cosine": .85, "cell_minimum_cosine": .70,
        "cell_positive_fraction": 1.0, "additive_pooled_cosine": .85,
        "additive_cell_mean_cosine": .75, "cell_advantage_over_additive": .05,
        "cell_prototype_stability_mean": .90, "cell_prototype_stability_minimum": .80}
PRICE = {"gpu_forwards": 0, "transformer_backwards": 0, "model_updates": 0,
         "prototype_vectors_fit": 24, "heldout_row_predictions": 240}
PREDICTION_KEYS = (
    "pred_a_authority_identity_balanced_folds_formula_finiteness_and_exact_price",
    "pred_b_six_cell_model_predicts_heldout_restricted_tensor",
    "pred_c_additive_hierarchy_explains_cells_without_material_interaction",
    "pred_d_cell_identities_are_crossfit_stable",
    "pred_e_scope_is_heldout_structure_screen_only",
)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def unit(vector):
    norm = float(np.linalg.norm(vector))
    if norm <= 0: raise RuntimeError("zero response/prototype")
    return vector / norm

def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "mean_cosine": float(values.mean()),
            "minimum_cosine": float(values.min()),
            "positive_fraction": float(np.mean(values > 0)), "cosines": values.tolist()}

def fit_models(Qn, fit):
    grand = Qn[:, fit].mean(axis=(0, 1))
    raw_panel = {p: Qn[i, fit].mean(axis=0) for i, p in enumerate(PANELS)}
    raw_direction = {d: Qn[:, [r for r in fit if r % 2 == d]].mean(axis=(0, 1)) for d in (0, 1)}
    raw_cell = {(p, d): Qn[i, [r for r in fit if r % 2 == d]].mean(axis=0)
                for i, p in enumerate(PANELS) for d in (0, 1)}
    return {"global": unit(grand),
            "direction": {d: unit(v) for d, v in raw_direction.items()},
            "panel": {p: unit(v) for p, v in raw_panel.items()},
            "additive": {(p, d): unit(raw_panel[p] + raw_direction[d] - grand)
                         for p in PANELS for d in (0, 1)},
            "cell": {key: unit(value) for key, value in raw_cell.items()}}

def prototype(model, fitted, panel, direction):
    if model == "global": return fitted[model]
    if model == "direction": return fitted[model][direction]
    if model == "panel": return fitted[model][panel]
    return fitted[model][(panel, direction)]

def atomic_write(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")

def main():
    paths = {"prior": PRIOR, "tensor_result": TENSOR_RESULT, "tensor": TENSOR,
             "builder": BUILDER, "signed_null": SIGNED_NULL}
    observed = {name: sha(path) for name, path in paths.items()}
    data = np.load(TENSOR, allow_pickle=False); Q = data["Q"].astype(np.float64)
    panels = tuple(str(value) for value in data["panels"])
    import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v20 as builder
    rows = builder.build_rows()
    by_panel = {p: [r for r in rows if r["transform_id"] == p] for p in panels}
    expected_ids = np.asarray([[r["row_id"] for r in by_panel[p]] for p in panels])
    label_ok = all(int(row["group_number"]) == i and bool(row["answer_changes"]) == (p != "P")
                   for p in PANELS for i, row in enumerate(by_panel[p]))
    authority_ok = (observed == EXPECTED and Q.shape == (4, 16, 4608)
                    and panels == ("A1", "A2", "P", "C")
                    and np.array_equal(data["row_ids"], expected_ids) and label_ok)
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "tensor_shape": list(Q.shape), "models": MODELS, "bars": BARS, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok or not np.isfinite(Q).all(): raise RuntimeError("authority or tensor invalid")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter(); Qn = np.stack([[unit(row) for row in panel] for panel in Q[:3]])
    folds = {"first_to_second": (np.array([i for i in range(16) if i % 4 < 2]),
                                  np.array([i for i in range(16) if i % 4 >= 2])),
             "second_to_first": (np.array([i for i in range(16) if i % 4 >= 2]),
                                  np.array([i for i in range(16) if i % 4 < 2]))}
    fitted, reports, pooled = {}, {}, {}
    balanced = True
    for fold, (fit, test) in folds.items():
        balanced &= all(sum(int(i % 2 == d) for i in fit) == 4 and sum(int(i % 2 == d) for i in test) == 4
                        for d in (0, 1))
        fitted[fold] = fit_models(Qn, fit); reports[fold] = {}
        for model in MODELS:
            reports[fold][model] = {}; all_values = []
            for panel_index, panel in enumerate(PANELS):
                for direction in (0, 1):
                    indices = [i for i in test if i % 2 == direction]
                    values = Qn[panel_index, indices] @ prototype(model, fitted[fold], panel, direction)
                    reports[fold][model][f"{panel}:{direction}"] = summarize(values)
                    all_values.extend(values.tolist())
            pooled.setdefault(model, []).extend(all_values)
    pooled_reports = {model: summarize(values) for model, values in pooled.items()}
    stability_values = [float(fitted["first_to_second"]["cell"][(p, d)]
                              @ fitted["second_to_first"]["cell"][(p, d)])
                        for p in PANELS for d in (0, 1)]
    stability = summarize(stability_values)
    A = bool(authority_ok and balanced and np.isfinite(Qn).all()
             and PRICE["prototype_vectors_fit"] == len(folds) * (1 + 2 + 3 + 6)
             and PRICE["heldout_row_predictions"] == len(folds) * len(MODELS) * 3 * 8)
    B = bool(pooled_reports["cell"]["mean_cosine"] >= BARS["cell_pooled_cosine"]
             and all(reports[f]["cell"][c]["mean_cosine"] >= BARS["cell_mean_cosine"]
                     and reports[f]["cell"][c]["minimum_cosine"] >= BARS["cell_minimum_cosine"]
                     and reports[f]["cell"][c]["positive_fraction"] >= BARS["cell_positive_fraction"]
                     for f in folds for c in reports[f]["cell"]))
    advantage = pooled_reports["cell"]["mean_cosine"] - pooled_reports["additive"]["mean_cosine"]
    C = bool(pooled_reports["additive"]["mean_cosine"] >= BARS["additive_pooled_cosine"]
             and advantage <= BARS["cell_advantage_over_additive"]
             and all(reports[f]["additive"][c]["mean_cosine"] >= BARS["additive_cell_mean_cosine"]
                     for f in folds for c in reports[f]["additive"]))
    D = bool(stability["mean_cosine"] >= BARS["cell_prototype_stability_mean"]
             and stability["minimum_cosine"] >= BARS["cell_prototype_stability_minimum"])
    E = True; predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A else "additive_discrete_hierarchy" if B and C and D
                else "coupled_six_state_tensor" if B and D else "continuous_or_positionwise_structure")
    result = {"schema": "temporal_iswas_v20_task_factor_factorial_prototype_crossfit_result_v1",
        "candidate_id": CANDIDATE_ID,
        "started_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "evidence_status": "cpu_heldout_restricted_tensor_structure_screen_only",
        "heldout_reports": reports, "pooled_reports": pooled_reports,
        "cell_prototype_crossfit_stability": stability,
        "cell_advantage_over_additive": float(advantage), "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE,
        "v21_causal_outcomes_opened": False, "causal_selector_identified": False}
    atomic_write(OUT, result)
    print(json.dumps({key: result[key] for key in ("pooled_reports", "cell_prototype_crossfit_stability",
        "cell_advantage_over_additive", "predictions", "terminal")}, sort_keys=True))

if __name__ == "__main__": main()
