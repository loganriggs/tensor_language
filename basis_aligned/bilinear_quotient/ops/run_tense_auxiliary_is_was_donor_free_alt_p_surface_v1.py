#!/usr/bin/env python3
"""Alternative-P robustness test for the frozen donor-free q_is actuator."""

# BQGATE: EXPERIMENT pred_a_row_identity_capability_and_exact_head pred_b_frozen_donor_free_actuator pred_c_alternative_P_robustness pred_d_exact_coverage_and_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_candidate_aspectual_different_readout_is_was_v2 as parent_rows
import circuit_candidate_tense_auxiliary_is_was_alt_p_surface_v1 as alt_rows
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_rank1_donor_free_margin_reflection_v1 as actuator


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_donor_free_alt_p_surface_v1.json"
V1 = ROOT / "circuits/followups/tense_auxiliary_is_was_rank1_donor_free_margin_reflection_v1_result.json"
V2 = ROOT / "circuits/followups/tense_auxiliary_is_was_rank1_donor_free_margin_reflection_v2_result.json"
Q_IS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
PARENT_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v2.py"
ALT_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_alt_p_surface_v1.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_donor_free_alt_p_surface_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.donor_free_alt_p_surface_v1"
SITE = "resid:18"
GRID_POINTS = 257
BUDGET = 7218.272705078125
EXPECTED_PRIOR_SHA256 = "b38d39f605831091c3b1319eb10b9a5c0b61ac7414aae78eede301988199f9d3"
EXPECTED = {
    V1: "8e49101dc5fe1e2086488868e39b45c7914c9255a2b2637066b0ecc20e9840f8",
    V2: "e69acc81c970a6794f3d7cac64a0fc43805afa253164b223833649e5fb46d580",
    Q_IS: "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    PARENT_BUILDER: "4ed62d06e2ffe5c471efc70eeaa35c7524a66923f19c64c803faef7200d4a62f",
    ALT_BUILDER: "1feebda644c925f56dfe171a2cf6c61f879f8c1070828b18283e39d66a4347c2",
}
MODEL_FORWARDS_EXACT = 2
EXAMPLE_EVALUATIONS_EXACT = 32
HEAD_GRID_EVALUATIONS_EXACT = 4112


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
    prior = json.loads(PRIOR.read_text())
    v1, v2_result, qi = json.loads(V1.read_text()), json.loads(V2.read_text()), json.loads(Q_IS.read_text())
    parent, rows = parent_rows.build_rows(), alt_rows.build_rows()
    old_non_p = [row for row in parent if row["family"] != "P"]
    new_non_p = [row for row in rows if row["family"] != "P"]
    p_rows = [row for row in rows if row["family"] == "P"]
    ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and prior["authority"]["frozen_budget"] == BUDGET
        and prior["authority"]["grid_points"] == GRID_POINTS
        and v1.get("terminal") == "invalid"
        and v2_result.get("terminal") == "null"
        and qi.get("terminal") == "screen"
        and qi["basis"]["shape"] == [1152, 1]
        and qi["basis"]["sha256"] == v1["basis_sha256"]
        and v1["basis_sha256"] == prior["authority"]["frozen_basis_sha256"]
        and v1["actuator"]["budget"] == BUDGET
        and old_non_p == new_non_p
        and len(p_rows) == 16
        and alt_rows.validate_rows(rows) == "c68bcc1c395102d92684870002ddb942ec3ae191e8e4e234ab3d97c01b3a8a61"
    )
    if not ok:
        raise ExperimentError("candidate, frozen actuator, parent identity, or rows changed")
    return p_rows, v1, qi


def main():
    rows, v1, qi = validate_static()
    plan = {
        "schema": "tense_auxiliary_is_was_donor_free_alt_p_surface_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "site": SITE, "grid_points": GRID_POINTS,
        "budget": BUDGET, "execution_policy": "managed_queue_only_capability_first",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "model_forwards_exact": MODEL_FORWARDS_EXACT,
        "example_evaluations_exact": EXAMPLE_EVALUATIONS_EXACT,
        "head_grid_evaluations_exact": HEAD_GRID_EVALUATIONS_EXACT,
        "fit_parameters": 0, "transformer_backwards": 0, "model_updates": 0,
        "A_or_C_causal_reevaluations": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    q = torch.as_tensor(qi["basis"]["values_column_major"], device=backend.device, dtype=torch.float32)
    if q.shape != (1152,) or hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != v1["basis_sha256"]:
        raise ExperimentError("basis reconstruction failed")

    outputs, states = {}, {}
    forward_calls, evaluations = 0, 0
    head_error = 0.0
    for side in ("base", "donor"):
        batch = das._batch(backend, rows, side=side)
        output = backend.native(batch, capture=True)
        forward_calls += 1
        evaluations += len(rows)
        outputs[side] = output
        states[side] = torch.stack([torch.as_tensor(output.captured[(row["row_id"], SITE)]) for row in rows]).to(backend.device).float()
        answer_ids = [row[f"{side}_answer_id"] for row in rows]
        foil_ids = [row[f"{side}_foil_id"] for row in rows]
        exact = actuator.selected_margin(backend, states[side], answer_ids, foil_ids)
        for i in range(len(rows)):
            head_error = max(head_error, abs(float(exact[i]) - (float(output.answer_foil[i][0]) - float(output.answer_foil[i][1]))))

    capability_cells = []
    capability_ok = True
    for direction in sorted({row["direction_id"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["direction_id"] == direction]
        for side in ("base", "donor"):
            margins = actuator.selected_margin(
                backend, states[side][indices],
                [rows[i][f"{side}_answer_id"] for i in indices], [rows[i][f"{side}_foil_id"] for i in indices],
            )
            accuracy = float((margins > 0.0).float().mean())
            passed = accuracy >= 0.85
            capability_ok = capability_ok and passed
            capability_cells.append({"direction": direction, "side": side, "correct": int((margins > 0.0).sum()), "total": len(indices), "accuracy": accuracy, "threshold": 0.85, "passed": passed})

    records = []
    grid_evaluations = 0
    if capability_ok:
        target_ids = [row["base_foil_id"] for row in rows]
        foil_ids = [row["base_answer_id"] for row in rows]
        base_margin, patched_margin, alpha, grid_index = actuator.actuate(backend, states["donor"], q, target_ids, foil_ids, BUDGET)
        grid_evaluations = GRID_POINTS * len(rows)
        for i, row in enumerate(rows):
            reflection = (float(patched_margin[i]) - float(base_margin[i])) / (-2.0 * float(base_margin[i]))
            records.append({
                "row_id": str(row["row_id"]), "direction": str(row["direction_id"]),
                "base_target_margin": float(base_margin[i]), "patched_target_margin": float(patched_margin[i]),
                "alpha": float(alpha[i]), "grid_index": int(grid_index[i]),
                "margin_reflection_fraction": reflection,
                "confirmation_donor_activation_used_by_actuator": False,
                "confirmation_donor_margin_used_by_actuator": False,
                "row_outcome_used_by_actuator": False,
            })
    summary = None
    if records:
        values = [record["margin_reflection_fraction"] for record in records]
        summary = {
            "count": len(values), "mean_margin_reflection_fraction": statistics.fmean(values),
            "mean_absolute_margin_reflection_fraction": statistics.fmean(abs(value) for value in values),
            "direction_fraction": sum(value > 0.0 for value in values) / len(values),
            "endpoint_fraction": sum(record["grid_index"] in (0, GRID_POINTS - 1) for record in records) / len(records),
        }
    pred_a = capability_ok and head_error <= 1.0e-3
    pred_b = (
        v1["actuator"]["budget"] == BUDGET and v1["actuator"]["grid_points"] == GRID_POINTS
        and all(math.isfinite(record["alpha"]) and abs(record["alpha"]) <= BUDGET * (1.0 + 1.0e-6) for record in records)
        and all(not record["confirmation_donor_activation_used_by_actuator"] and not record["confirmation_donor_margin_used_by_actuator"] and not record["row_outcome_used_by_actuator"] for record in records)
    )
    pred_c = bool(summary) and summary["mean_margin_reflection_fraction"] >= 0.75 and summary["direction_fraction"] >= 0.75
    pred_d = len(records) == 16 and len({record["row_id"] for record in records}) == 16 and forward_calls == MODEL_FORWARDS_EXACT and evaluations == EXAMPLE_EVALUATIONS_EXACT and grid_evaluations == HEAD_GRID_EVALUATIONS_EXACT
    predictions = {
        "pred_a_row_identity_capability_and_exact_head": pred_a,
        "pred_b_frozen_donor_free_actuator": pred_b,
        "pred_c_alternative_P_robustness": pred_c,
        "pred_d_exact_coverage_and_price": pred_d,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_d else "invalid")
    reason = {"screen": "donor_free_additive_q_is_robust_to_alternative_P", "null": "alternative_P_reflection_misses_unchanged_bar", "invalid": "row_capability_head_actuator_or_coverage_invalid"}[terminal]
    result = {
        "schema": "tense_auxiliary_is_was_donor_free_alt_p_surface_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only_capability_first",
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "authority_sha256": alt_rows.authority_sha256(), "basis_sha256": v1["basis_sha256"],
        "actuator": {"site": SITE, "budget": BUDGET, "grid_points": GRID_POINTS, "inherited_fit_parameters": 0},
        "head_control": {"max_abs_difference": head_error}, "capability_cells": capability_cells,
        "score": {"alternative_P": summary, "model_forwards": forward_calls, "example_evaluations": evaluations, "head_grid_evaluations": grid_evaluations, "A_or_C_causal_reevaluations": 0, "fit_parameters": 0, "transformer_backwards": 0, "model_updates": 0},
        "intervention_records": records, "predictions": predictions,
        "terminal": terminal, "reason": reason,
        "next_action": "fit and prospectively validate a local is/was upstream gain controller" if terminal == "screen" else "retain selective projected q_is and close donor-free additive promotion",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "capability_cells": capability_cells, "alternative_P": summary, "price": result["score"], "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
