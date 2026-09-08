#!/usr/bin/env python3
"""CPU-only cross-fit test for a selective binary M11 factor subset."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_shape_finiteness_folds_formula_and_price pred_b_binary_top64_is_target_sufficient_and_control_selective_on_both_heldout_folds pred_c_binary_factor_identity_is_crossfit_stable pred_d_result_is_diagnostic_and_v21_remains_causally_sealed
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v20_task_factor_crossfit_selectivity_v1.json"
TENSOR_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1_result.json"
TENSOR = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1.npz"
LOWER_BOUND = ROOT / "circuits/followups/temporal_iswas_v20_task_factor_selectivity_lower_bound_v1.json"
V21_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v21.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v20_task_factor_crossfit_selectivity_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v20_task_factor_crossfit_selectivity_v1"
EXPECTED = {
    "prior": "a925c0469bf2c7c331536a45717ab2cd46133c7db398506653dd785ea1fa17fe",
    "tensor_result": "da97abea6f3dae8721c2cda383e0b7af844217051e199046a23c6b9c264da396",
    "tensor": "a79ff1b0e8eac3e3b135a8cb02eaa58e83c317ade23f0d65950b90b2ef2dff12",
    "lower_bound": "b58e22c2b3f52b8226e29df2d88df7e43adfc9d9353b3ac5e31854b2d078cf4b",
    "v21_builder": "4289795a9886c3c36561f47ca202cb790072dd05c324d96ca9fbd5e837a23dc8",
}
PANELS, PREFIXES = ("A1", "A2", "P", "C"), (8, 16, 32, 64)
PRICE = {"gpu_forwards": 0, "transformer_backwards": 0, "model_updates": 0,
         "factor_scores": 9216, "heldout_prefix_panel_evaluations": 32}
BARS = {"target_recovery": .50, "target_cosine": .90, "target_direction": .875,
        "control_ratio": .25, "top64_jaccard": .25}
PREDICTION_KEYS = (
    "pred_a_authority_shape_finiteness_folds_formula_and_price",
    "pred_b_binary_top64_is_target_sufficient_and_control_selective_on_both_heldout_folds",
    "pred_c_binary_factor_identity_is_crossfit_stable",
    "pred_d_result_is_diagnostic_and_v21_remains_causally_sealed",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(effect, reference):
    effect, reference = np.asarray(effect, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    denominator = max(float(reference @ reference), 1e-30)
    cosine_denominator = max(float(np.linalg.norm(effect) * np.linalg.norm(reference)), 1e-30)
    return {"count": int(reference.size), "signed_recovery": float(effect @ reference / denominator),
            "cosine": float(effect @ reference / cosine_denominator),
            "direction_agreement": float(np.mean(np.sign(effect) == np.sign(reference))),
            "relative_residual": float(np.linalg.norm(effect - reference)
                                       / max(np.linalg.norm(reference), 1e-30))}


def order_on(Q, fit):
    a1, a2, p = (Q[index, fit] for index in range(3))
    a1_den = max(abs(float(a1.sum(axis=1).mean())), 1e-30)
    a2_den = max(abs(float(a2.sum(axis=1).mean())), 1e-30)
    target_scale = math.sqrt(float(np.mean(np.concatenate([
        a1.sum(axis=1), a2.sum(axis=1)]) ** 2)))
    score = np.minimum(a1.mean(axis=0) / a1_den, a2.mean(axis=0) / a2_den)
    score -= 4.0 * np.abs(p.mean(axis=0) / max(target_scale, 1e-30))
    return np.argsort(-score, kind="stable"), score


def evaluate(Q, test, order):
    reference = {panel: Q[index, test].sum(axis=1) for index, panel in enumerate(PANELS)}
    target_rms = math.sqrt(float(np.mean(np.concatenate([
        reference["A1"], reference["A2"]]) ** 2)))
    reports, controls = {}, {}
    for prefix in PREFIXES:
        chosen = order[:prefix]
        effects = {panel: Q[index, test][:, chosen].sum(axis=1)
                   for index, panel in enumerate(PANELS)}
        reports[str(prefix)] = {panel: metrics(effects[panel], reference[panel])
                                for panel in ("A1", "A2")}
        controls[str(prefix)] = {panel: float(np.sqrt(np.mean(effects[panel] ** 2))
                                             / max(target_rms, 1e-30))
                                 for panel in ("P", "C")}
    return reports, controls


def atomic_write(path, payload):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    paths = {"prior": PRIOR, "tensor_result": TENSOR_RESULT, "tensor": TENSOR,
             "lower_bound": LOWER_BOUND, "v21_builder": V21_BUILDER}
    observed = {name: sha(path) for name, path in paths.items()}
    data = np.load(TENSOR, allow_pickle=False)
    Q = data["Q"].astype(np.float64)
    panels = tuple(str(value) for value in data["panels"])
    authority_ok = observed == EXPECTED and Q.shape == (4, 16, 4608) and panels == PANELS
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "tensor_shape": list(Q.shape), "prefixes": PREFIXES, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok or not np.isfinite(Q).all():
        raise RuntimeError("crossfit factor tensor authority or finiteness changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    folds = {"even_to_odd": (np.arange(0, 16, 2), np.arange(1, 16, 2)),
             "odd_to_even": (np.arange(1, 16, 2), np.arange(0, 16, 2))}
    orders, scores, reports, controls = {}, {}, {}, {}
    for label, (fit, test) in folds.items():
        order, score = order_on(Q, fit)
        orders[label] = order
        scores[label] = score
        reports[label], controls[label] = evaluate(Q, test, order)
    sets = [set(orders[label][:64].tolist()) for label in folds]
    jaccard = len(sets[0] & sets[1]) / len(sets[0] | sets[1])
    A = bool(authority_ok and np.isfinite(np.stack(list(scores.values()))).all()
             and PRICE["factor_scores"] == len(folds) * Q.shape[2]
             and PRICE["heldout_prefix_panel_evaluations"] == len(folds) * len(PREFIXES) * len(PANELS))
    B = all(reports[fold]["64"][panel]["signed_recovery"] >= BARS["target_recovery"]
            and reports[fold]["64"][panel]["cosine"] >= BARS["target_cosine"]
            and reports[fold]["64"][panel]["direction_agreement"] >= BARS["target_direction"]
            for fold in folds for panel in ("A1", "A2")) and all(
            controls[fold]["64"][panel] <= BARS["control_ratio"]
            for fold in folds for panel in ("P", "C"))
    C = jaccard >= BARS["top64_jaccard"]
    D = True
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A else "stable_binary_factor_split_screen" if B and C
                else "fixed_binary_factor_split_closed")
    full_order = order_on(Q, np.arange(16))[0][:64].tolist() if B and C else None
    result = {"schema": "temporal_iswas_v20_task_factor_crossfit_selectivity_result_v1",
        "candidate_id": CANDIDATE_ID,
        "started_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "evidence_status": "post_v20_cpu_crossfit_diagnostic_only",
        "orders_first64": {label: orders[label][:64].tolist() for label in folds},
        "top64_jaccard": jaccard, "heldout_reports": reports,
        "heldout_control_normalized_rms": controls,
        "full_v20_order_first64_only_if_screen_passes": full_order,
        "v21_causal_outcomes_opened": False, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_write(OUT, result)
    print(json.dumps({key: result[key] for key in ("top64_jaccard", "heldout_reports",
        "heldout_control_normalized_rms", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
