#!/usr/bin/env python3
"""Prospective capability-gated bracket/root controller for q_is."""

# BQGATE: EXPERIMENT pred_a_fresh_authority_capability_and_exact_head pred_b_donor_free_root_controller pred_c_fresh_A_actuation pred_d_fresh_P_generalization pred_e_unrelated_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v4 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_rank1_donor_free_margin_reflection_v1 as head


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_bracketed_root_actuator_fresh_lexicon_v1.json"
Q_IS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
FIXED_NULL = ROOT / "circuits/followups/tense_auxiliary_is_was_rank1_donor_free_margin_reflection_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v4.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_bracketed_root_actuator_fresh_lexicon_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.bracketed_root_actuator_fresh_lexicon_v1"
SITE = "resid:18"
TOKEN_IDS = {"is": 318, "was": 373}
BRACKET_ROUNDS = 24
BISECTION_ROUNDS = 32
INITIAL_DIVISOR = 1024.0
EXPECTED_PRIOR_SHA256 = "e1e0ca6cb5b0ceb9b09a8b483bb99efecf40636ff6942cf8f45ad70f00483f80"
EXPECTED = {
    Q_IS: "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    FIXED_NULL: "e69acc81c970a6794f3d7cac64a0fc43805afa253164b223833649e5fb46d580",
    BUILDER: "1d90b1b7feebcf4eb467b41b9b4b168a6ecc62b3a4c4178a91a502bc7923b74b",
}
EXPECTED_ROWS_SHA256 = "c62c2f1eeb311afad1631f4ccd0077211121a4e493cf772676d59ba33e01f4b2"
MODEL_FORWARDS_EXACT = 2
EXAMPLE_EVALUATIONS_EXACT = 128
ROOT_HEAD_EVALUATIONS_MAX = 3776


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direction_for(row):
    return row["direction_id"] if row["family"] in ("A1", "A2") else ("present_to_past" if row["group_number"] % 2 == 0 else "past_to_present")


def requested_ids(direction):
    return (TOKEN_IDS["was"], TOKEN_IDS["is"]) if direction == "present_to_past" else (TOKEN_IDS["is"], TOKEN_IDS["was"])


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    qi, fixed = json.loads(Q_IS.read_text()), json.loads(FIXED_NULL.read_text())
    rows = fresh.build_rows()
    ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and prior["frozen_design"]["root_head_evaluations_max"] == ROOT_HEAD_EVALUATIONS_MAX
        and qi.get("terminal") == "screen"
        and fixed.get("terminal") == "null"
        and qi["basis"]["shape"] == [1152, 1]
        and fresh.validate_rows(rows) == EXPECTED_ROWS_SHA256
        and tuple(prior["fresh_authority"]["agents"]) == fresh._AGENTS
    )
    if not ok:
        raise ExperimentError("candidate, controller, basis, null, rows, or agents changed")
    return rows, qi


def root_actuate(backend, states, q, target_ids, foil_ids):
    """Fixed dyadic bracket plus bisection; returns exact head-only accounting."""
    torch = backend.torch
    n = states.shape[0]
    base_margin = head.selected_margin(backend, states, target_ids, foil_ids)
    target_margin = -base_margin
    step = torch.linalg.vector_norm(states, dim=1) / INITIAL_DIVISOR
    plus = head.selected_margin(backend, states + step[:, None] * q, target_ids, foil_ids)
    minus = head.selected_margin(backend, states - step[:, None] * q, target_ids, foil_ids)
    choose_plus = (plus - target_margin).abs() <= (minus - target_margin).abs()
    sign = torch.where(choose_plus, torch.ones_like(step), -torch.ones_like(step))
    lo = torch.zeros_like(step)
    hi = sign * step
    f_lo = base_margin - target_margin
    f_hi = torch.where(choose_plus, plus, minus) - target_margin
    bracketed = f_lo * f_hi <= 0.0
    bracket_round = torch.where(bracketed, torch.zeros_like(step, dtype=torch.long), torch.full_like(step, -1, dtype=torch.long))
    head_evaluations = 3 * n
    # Round zero intentionally re-evaluates the chosen signed initial point. This
    # makes every round use the same vectorized path and pins exact accounting.
    for round_index in range(BRACKET_ROUNDS):
        candidate = torch.where(bracketed, hi, sign * step * float(2 ** round_index))
        candidate_margin = head.selected_margin(backend, states + candidate[:, None] * q, target_ids, foil_ids)
        candidate_f = candidate_margin - target_margin
        newly = (~bracketed) & (f_lo * candidate_f <= 0.0)
        hi = torch.where(bracketed, hi, candidate)
        f_hi = torch.where(bracketed, f_hi, candidate_f)
        bracket_round = torch.where(newly, torch.full_like(bracket_round, round_index), bracket_round)
        bracketed = bracketed | newly
        head_evaluations += n
    # Maintain the invariant f(lo)*f(hi)<=0 on bracketed rows.
    hi_margin = f_hi + target_margin
    for _ in range(BISECTION_ROUNDS):
        mid = (lo + hi) / 2.0
        mid_margin = head.selected_margin(backend, states + mid[:, None] * q, target_ids, foil_ids)
        f_mid = mid_margin - target_margin
        replace_hi = bracketed & (f_lo * f_mid <= 0.0)
        replace_lo = bracketed & (~replace_hi)
        hi = torch.where(replace_hi, mid, hi)
        f_hi = torch.where(replace_hi, f_mid, f_hi)
        hi_margin = torch.where(replace_hi, mid_margin, hi_margin)
        lo = torch.where(replace_lo, mid, lo)
        f_lo = torch.where(replace_lo, f_mid, f_lo)
        head_evaluations += n
    alpha = hi
    chosen_margin = hi_margin
    return base_margin, chosen_margin, alpha, bracketed, bracket_round, head_evaluations


def summarize(records, key):
    values = [record[key] for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError(f"missing/nonfinite {key}")
    return {"count": len(values), f"mean_{key}": statistics.fmean(values), f"mean_absolute_{key}": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def main():
    rows, qi = validate_static()
    plan = {
        "schema": "tense_auxiliary_is_was_bracketed_root_actuator_fresh_lexicon_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "site": SITE, "execution_policy": "managed_queue_only_capability_first",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS_SHA256,
        "bracket_rounds": BRACKET_ROUNDS, "bisection_rounds": BISECTION_ROUNDS,
        "initial_divisor": INITIAL_DIVISOR, "model_forwards_exact": MODEL_FORWARDS_EXACT,
        "example_evaluations_exact": EXAMPLE_EVALUATIONS_EXACT,
        "root_head_evaluations_max": ROOT_HEAD_EVALUATIONS_MAX,
        "fit_parameters": 0, "grid_points": 0, "transformer_backwards": 0, "model_updates": 0,
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
    if q.shape != (1152,) or hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != qi["basis"]["sha256"]:
        raise ExperimentError("basis reconstruction failed")

    outputs, states = {}, {}
    head_error = 0.0
    for side in ("base", "donor"):
        batch = das._batch(backend, rows, side=side)
        output = backend.native(batch, capture=True)
        outputs[side] = output
        states[side] = torch.stack([torch.as_tensor(output.captured[(row["row_id"], SITE)]) for row in rows]).to(backend.device).float()
        exact = head.selected_margin(backend, states[side], [row[f"{side}_answer_id"] for row in rows], [row[f"{side}_foil_id"] for row in rows])
        for i in range(len(rows)):
            head_error = max(head_error, abs(float(exact[i]) - (float(output.answer_foil[i][0]) - float(output.answer_foil[i][1]))))

    native = []
    for i, row in enumerate(rows):
        direction = direction_for(row)
        for side in ("base", "donor"):
            correct = float(outputs[side].answer_foil[i][0]) > float(outputs[side].answer_foil[i][1])
            native.append({"family": row["family"], "direction": direction, "side": side, "row_id": str(row["row_id"]), "correct": correct})
    cells = []
    for family in ("A1", "A2", "P", "C"):
        for direction in ("past_to_present", "present_to_past"):
            selected = [record for record in native if record["family"] == family and record["direction"] == direction]
            accuracy = sum(record["correct"] for record in selected) / len(selected)
            threshold = 0.75 if family == "C" else 0.85
            cells.append({"family": family, "direction": direction, "correct": sum(record["correct"] for record in selected), "total": len(selected), "accuracy": accuracy, "threshold": threshold, "passed": accuracy >= threshold})
    capability_ok = all(cell["passed"] for cell in cells)

    records = []
    head_evaluations = 0
    causal_outcomes_opened = False
    if capability_ok:
        causal_outcomes_opened = True
        source = torch.stack([states["donor"][i] if row["family"] == "P" else states["base"][i] for i, row in enumerate(rows)])
        directions = [direction_for(row) for row in rows]
        target_ids, foil_ids = zip(*(requested_ids(direction) for direction in directions))
        base_margin, patched_margin, alpha, bracketed, bracket_round, head_evaluations = root_actuate(backend, source, q, target_ids, foil_ids)
        target_scale = float(qi["score"]["families"]["target_scale"])
        for i, row in enumerate(rows):
            patched_state = source[i] + alpha[i] * q
            record = {
                "family": row["family"], "row_id": str(row["row_id"]), "direction": directions[i],
                "base_target_margin": float(base_margin[i]), "patched_target_margin": float(patched_margin[i]),
                "alpha": float(alpha[i]), "bracketed": bool(bracketed[i]), "bracket_round": int(bracket_round[i]),
                "margin_reflection_fraction": (float(patched_margin[i]) - float(base_margin[i])) / (-2.0 * float(base_margin[i])),
                "margin_reflection_absolute_error": abs(float(patched_margin[i]) + float(base_margin[i])),
                "donor_activation_used_by_controller": False, "donor_margin_used_by_controller": False,
                "row_outcome_used_by_controller": False, "learned_budget_used": False,
            }
            if row["family"] in ("A1", "A2"):
                donor_reference = head.selected_margin(backend, states["donor"][i:i + 1], [target_ids[i]], [foil_ids[i]])[0]
                record["donor_reference_margin"] = float(donor_reference)
                record["recovery"] = (float(patched_margin[i]) - float(base_margin[i])) / (float(donor_reference) - float(base_margin[i]))
            elif row["family"] == "C":
                before_c = head.selected_margin(backend, source[i:i + 1], [row["base_answer_id"]], [row["base_foil_id"]])[0]
                after_c = head.selected_margin(backend, patched_state[None, :], [row["base_answer_id"]], [row["base_foil_id"]])[0]
                record["normalized_unrelated_effect"] = abs(float(after_c) - float(before_c)) / target_scale
            records.append(record)
    families = {family: [record for record in records if record["family"] == family] for family in ("A1", "A2", "P", "C")}
    summaries = None
    if records:
        summaries = {
            "A1": summarize(families["A1"], "recovery"),
            "A2": summarize(families["A2"], "recovery"),
            "P": summarize(families["P"], "margin_reflection_fraction"),
            "C": summarize(families["C"], "normalized_unrelated_effect"),
        }
    pred_a = fresh.authority_sha256() == EXPECTED_ROWS_SHA256 and capability_ok and head_error <= 1.0e-3
    a_p_records = [record for record in records if record["family"] in ("A1", "A2", "P")]
    pred_b = (
        len(a_p_records) == 48 and all(record["bracketed"] and record["margin_reflection_absolute_error"] <= 1.0e-4 for record in a_p_records)
        and all(not record["donor_activation_used_by_controller"] and not record["donor_margin_used_by_controller"] and not record["row_outcome_used_by_controller"] and not record["learned_budget_used"] for record in records)
    )
    pred_c = bool(summaries) and all(summaries[family]["mean_recovery"] >= 0.75 and summaries[family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_d = bool(summaries) and summaries["P"]["mean_margin_reflection_fraction"] >= 0.95 and summaries["P"]["direction_fraction"] == 1.0
    pred_e = bool(summaries) and summaries["C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_f = len(records) == 64 and len({record["row_id"] for record in records}) == 64 and head_evaluations <= ROOT_HEAD_EVALUATIONS_MAX
    predictions = {
        "pred_a_fresh_authority_capability_and_exact_head": pred_a,
        "pred_b_donor_free_root_controller": pred_b,
        "pred_c_fresh_A_actuation": pred_c,
        "pred_d_fresh_P_generalization": pred_d,
        "pred_e_unrelated_C_selectivity": pred_e,
        "pred_f_exact_coverage_and_price": pred_f,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_f else "invalid")
    reason = {"screen": "fresh_donor_free_root_q_is_actuator_is_selective", "null": "valid_root_actuator_misses_fresh_A_P_or_C", "invalid": "authority_capability_head_controller_root_or_coverage_invalid"}[terminal]
    result = {
        "schema": "tense_auxiliary_is_was_bracketed_root_actuator_fresh_lexicon_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only_capability_first",
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "rows_sha256": EXPECTED_ROWS_SHA256, "basis_sha256": qi["basis"]["sha256"],
        "controller": {"site": SITE, "fixed_token_ids": TOKEN_IDS, "initial_divisor": INITIAL_DIVISOR, "bracket_rounds": BRACKET_ROUNDS, "bisection_rounds": BISECTION_ROUNDS, "donor_free": True, "learned_budget": False},
        "head_control": {"max_abs_difference": head_error}, "capability_cells": cells,
        "causal_outcomes_opened": causal_outcomes_opened, "score": {"families": summaries, "model_forwards": 2, "example_evaluations": 128, "intervention_records": len(records), "selected_head_root_evaluations": head_evaluations, "fit_parameters": 0, "grid_points": 0, "transformer_backwards": 0, "model_updates": 0},
        "intervention_records": records, "predictions": predictions,
        "terminal": terminal, "reason": reason,
        "next_action": "fit local resid10 is/was read to root alpha and validate on another fresh authority" if terminal == "screen" else "retain selective projected q_is and do not raise rank",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "capability_cells": cells, "causal_outcomes_opened": causal_outcomes_opened, "families": summaries, "price": result["score"], "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
