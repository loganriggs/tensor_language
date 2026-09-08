#!/usr/bin/env python3
"""Cross-fit a discrete signed prototype to the v20 task-functional tensor."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_shape_row_identity_sign_formula_finiteness_and_exact_price pred_b_signed_two_point_model_predicts_each_heldout_panel pred_c_signed_coordinate_is_crossfit_stable_and_beats_unsigned_control pred_d_scope_remains_a_diagnostic_screen
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v20_task_factor_signed_prototype_crossfit_v1.json"
TENSOR_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1_result.json"
TENSOR = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1.npz"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v20.py"
ANSWER_SIGN = ROOT / "circuits/followups/temporal_iswas_v20_task_factor_answer_sign_analysis_v1.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v20_task_factor_signed_prototype_crossfit_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v20_task_factor_signed_prototype_crossfit_v1"
EXPECTED = {
    "prior": "fff6691ae03480020fedb7f1f82bfe684104c6fcf829adc5db4089e241d51306",
    "tensor_result": "da97abea6f3dae8721c2cda383e0b7af844217051e199046a23c6b9c264da396",
    "tensor": "a79ff1b0e8eac3e3b135a8cb02eaa58e83c317ade23f0d65950b90b2ef2dff12",
    "builder": "cafe9120c4ba8fdbeff27c5a2d05a247c2000641fd401529a4ca0a089137d1d9",
    "answer_sign": "09b2b5e58a8b5990a905707835f5424696cfeb7ac83fbfa18578ff24377540e1",
}
PANELS = ("A1", "A2", "P")
BARS = {"per_panel_mean_cosine": .80, "per_panel_minimum_cosine": .60,
        "per_panel_positive_fraction": .875, "crossfit_prototype_cosine": .90,
        "pooled_improvement_over_unsigned": .50,
        "per_fold_improvement_over_unsigned": .45}
PRICE = {"gpu_forwards": 0, "transformer_backwards": 0, "model_updates": 0,
         "prototype_fits": 4, "heldout_row_predictions": 96,
         "fitted_vector_values_per_prototype": 4608}
PREDICTION_KEYS = (
    "pred_a_authority_shape_row_identity_sign_formula_finiteness_and_exact_price",
    "pred_b_signed_two_point_model_predicts_each_heldout_panel",
    "pred_c_signed_coordinate_is_crossfit_stable_and_beats_unsigned_control",
    "pred_d_scope_remains_a_diagnostic_screen",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def unit_rows(array):
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norms <= 0): raise RuntimeError("zero task-functional row")
    return array / norms


def unit(vector):
    norm = np.linalg.norm(vector)
    if norm <= 0: raise RuntimeError("zero prototype")
    return vector / norm


def report(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "mean_cosine": float(values.mean()),
            "minimum_cosine": float(values.min()),
            "positive_fraction": float(np.mean(values > 0)),
            "cosines": values.tolist()}


def atomic_write(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")


def main():
    paths = {"prior": PRIOR, "tensor_result": TENSOR_RESULT, "tensor": TENSOR,
             "builder": BUILDER, "answer_sign": ANSWER_SIGN}
    observed = {name: sha(path) for name, path in paths.items()}
    data = np.load(TENSOR, allow_pickle=False)
    Q = data["Q"].astype(np.float64)
    panels = tuple(str(value) for value in data["panels"])
    import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v20 as builder
    rows = builder.build_rows()
    by_panel = {panel: [row for row in rows if row["transform_id"] == panel]
                for panel in panels}
    row_ids = np.asarray(data["row_ids"])
    expected_ids = np.asarray([[row["row_id"] for row in by_panel[panel]] for panel in panels])
    signs = np.asarray([[1.0 if row["donor_answer"] == " is" else -1.0
                         for row in by_panel[panel]] for panel in PANELS])
    sign_values_ok = all(row["donor_answer"] in {" is", " was"}
                         for panel in PANELS for row in by_panel[panel])
    authority_ok = (observed == EXPECTED and Q.shape == (4, 16, 4608)
                    and panels == ("A1", "A2", "P", "C")
                    and np.array_equal(row_ids, expected_ids) and sign_values_ok)
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "tensor_shape": list(Q.shape), "panels": PANELS, "bars": BARS, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok or not np.isfinite(Q).all():
        raise RuntimeError("signed prototype authority, identity, or finiteness changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    Qn = unit_rows(Q[:3])
    folds = {"even_to_odd": (np.arange(0, 16, 2), np.arange(1, 16, 2)),
             "odd_to_even": (np.arange(1, 16, 2), np.arange(0, 16, 2))}
    prototypes, heldout, fold_summary = {}, {}, {}
    for label, (fit, test) in folds.items():
        signed = unit((Qn[:, fit] * signs[:, fit, None]).mean(axis=(0, 1)))
        unsigned = unit(Qn[:, fit].mean(axis=(0, 1)))
        prototypes[label] = signed
        heldout[label] = {"signed": {}, "unsigned": {}}
        signed_all, unsigned_all = [], []
        for panel_index, panel in enumerate(PANELS):
            signed_cos = (Qn[panel_index, test]
                          @ (signs[panel_index, test, None] * signed).T).diagonal()
            unsigned_cos = Qn[panel_index, test] @ unsigned
            heldout[label]["signed"][panel] = report(signed_cos)
            heldout[label]["unsigned"][panel] = report(unsigned_cos)
            signed_all.extend(signed_cos.tolist()); unsigned_all.extend(unsigned_cos.tolist())
        signed_mean, unsigned_mean = float(np.mean(signed_all)), float(np.mean(unsigned_all))
        fold_summary[label] = {"signed_pooled_mean_cosine": signed_mean,
                               "unsigned_pooled_mean_cosine": unsigned_mean,
                               "improvement": signed_mean - unsigned_mean}
    prototype_cosine = float(prototypes["even_to_odd"] @ prototypes["odd_to_even"])
    signed_pooled = np.mean([fold_summary[key]["signed_pooled_mean_cosine"] for key in folds])
    unsigned_pooled = np.mean([fold_summary[key]["unsigned_pooled_mean_cosine"] for key in folds])
    pooled_improvement = float(signed_pooled - unsigned_pooled)
    A = bool(authority_ok and np.isfinite(np.stack(list(prototypes.values()))).all()
             and PRICE["prototype_fits"] == 2 * len(folds)
             and PRICE["heldout_row_predictions"] == 2 * len(folds) * len(PANELS) * 8)
    B = all(heldout[fold]["signed"][panel]["mean_cosine"] >= BARS["per_panel_mean_cosine"]
            and heldout[fold]["signed"][panel]["minimum_cosine"] >= BARS["per_panel_minimum_cosine"]
            and heldout[fold]["signed"][panel]["positive_fraction"] >= BARS["per_panel_positive_fraction"]
            for fold in folds for panel in PANELS)
    C = bool(prototype_cosine >= BARS["crossfit_prototype_cosine"]
             and pooled_improvement >= BARS["pooled_improvement_over_unsigned"]
             and all(summary["improvement"] >= BARS["per_fold_improvement_over_unsigned"]
                     for summary in fold_summary.values()))
    D = True
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A else "signed_discrete_carrier_screen"
                if B and C else "higher_structure_required")
    result = {"schema": "temporal_iswas_v20_task_factor_signed_prototype_crossfit_result_v1",
        "candidate_id": CANDIDATE_ID,
        "started_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "evidence_status": "cpu_crossfit_diagnostic_screen_only", "heldout": heldout,
        "fold_summary": fold_summary, "crossfit_prototype_cosine": prototype_cosine,
        "pooled_signed_mean_cosine": float(signed_pooled),
        "pooled_unsigned_mean_cosine": float(unsigned_pooled),
        "pooled_improvement_over_unsigned": pooled_improvement,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE,
        "v21_causal_outcomes_opened": False, "causal_selector_identified": False}
    atomic_write(OUT, result)
    print(json.dumps({key: result[key] for key in ("fold_summary", "crossfit_prototype_cosine",
        "pooled_signed_mean_cosine", "pooled_unsigned_mean_cosine",
        "pooled_improvement_over_unsigned", "predictions", "terminal")}, sort_keys=True))


if __name__ == "__main__": main()
