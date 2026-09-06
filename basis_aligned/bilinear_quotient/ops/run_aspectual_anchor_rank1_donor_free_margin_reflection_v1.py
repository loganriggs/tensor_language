#!/usr/bin/env python3
"""Donor-free target-guided actuation through the identified aspectual rank-one carrier."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_exact_head pred_b_budget_and_actuator_well_formed pred_c_heldout_lexical_actuation pred_d_cross_construction_actuation pred_e_noun_shift_generalization pred_f_unrelated_output_selectivity pred_g_exact_coverage_and_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_candidate_aspectual_fresh_construction_v2 as fresh
import circuit_candidate_aspectual_lexical_holdout_v5 as lexical
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_rank1_donor_free_margin_reflection_v1.json"
RANK1 = ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json"
LEXICAL_BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
FRESH_BUILDER = ROOT / "ops/circuit_candidate_aspectual_fresh_construction_v2.py"
DAS_LIBRARY = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_rank1_donor_free_margin_reflection_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.rank1_donor_free_margin_reflection_v1"
EXPECTED_PRIOR_SHA256 = "d8142f1498306a9206b5235c3d451b62056e66a7b16c35fd80feec7c8b49d31a"
EXPECTED = {
    RANK1: "58b83a2714ae8d53cc799d5e6ae96c61cc476f22e09019e6e1f620581ff9a278",
    LEXICAL_BUILDER: "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859",
    FRESH_BUILDER: "848332a12c22bf523573e015b6f8f0a38b5865db8b77434dcbe6a176d98370ac",
    DAS_LIBRARY: "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
}
SITE = "resid:18"
GRID_POINTS = 257
MODEL_FORWARDS_MAX = 33
EXAMPLE_EVALUATIONS_MAX = 264
HEAD_GRID_EVALUATIONS = 30_840


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
    rank1 = json.loads(RANK1.read_text())
    lexical_rows, fresh_rows = lexical.build_rows(), fresh.build_rows()
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or prior["frozen_design"]["grid_points"] != GRID_POINTS
        or rank1.get("terminal") != "screen"
        or rank1.get("rank") != 1
        or rank1["basis"]["shape"] != [1152, 1]
        or rank1["basis"]["sha256"] != "123c6e098fcccf68bd9b881bb81c6b95858a258baa688b79a947a3043bb61e39"
        or lexical.validate_rows(lexical_rows) != "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
        or fresh.validate_rows(fresh_rows) != "3c30019fdcc087c0e7410cd82d02458307bc6987ff9d23349dcf97d076f797d7"
    ):
        raise ExperimentError("authority, terminal, rank, grid, or row hash changed")
    return lexical_rows, fresh_rows, rank1


def selected_margin(backend, states, target_ids, foil_ids):
    """Exact soft-capped two-token margin without materializing full-vocabulary logits."""
    torch, F, model = backend.torch, backend.F, backend.model
    target = torch.as_tensor(target_ids, device=backend.device, dtype=torch.long)
    foil = torch.as_tensor(foil_ids, device=backend.device, dtype=torch.long)
    normalized = F.rms_norm(states, (model.config.n_embd,))
    target_raw = (normalized * model.lm_head.weight[target]).sum(dim=-1)
    foil_raw = (normalized * model.lm_head.weight[foil]).sum(dim=-1)
    return 30.0 * torch.tanh(target_raw / 30.0) - 30.0 * torch.tanh(foil_raw / 30.0)


def actuate(backend, states, q, target_ids, foil_ids, budget):
    """Choose alpha using only states, q, requested tokens, and the fixed head."""
    torch = backend.torch
    grid = torch.linspace(-budget, budget, GRID_POINTS, device=backend.device, dtype=states.dtype)
    with torch.no_grad():
        base_margin = selected_margin(backend, states, target_ids, foil_ids)
        candidates = states[:, None, :] + grid[None, :, None] * q[None, None, :]
        expanded_target = torch.as_tensor(target_ids, device=backend.device, dtype=torch.long)[:, None].expand(-1, GRID_POINTS).reshape(-1)
        expanded_foil = torch.as_tensor(foil_ids, device=backend.device, dtype=torch.long)[:, None].expand(-1, GRID_POINTS).reshape(-1)
        margins = selected_margin(backend, candidates.reshape(-1, states.shape[-1]), expanded_target, expanded_foil).reshape(states.shape[0], GRID_POINTS)
        chosen_index = (margins + base_margin[:, None]).abs().argmin(dim=1)
        row_index = torch.arange(states.shape[0], device=backend.device)
        chosen_margin = margins[row_index, chosen_index]
        chosen_alpha = grid[chosen_index]
    return base_margin, chosen_margin, chosen_alpha, chosen_index


def summarize(records, key):
    values = [record[key] for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError(f"missing/nonfinite summary: {key}")
    return {
        "count": len(values),
        f"mean_{key}": statistics.fmean(values),
        f"mean_absolute_{key}": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
        "endpoint_fraction": sum(record["grid_index"] in (0, GRID_POINTS - 1) for record in records) / len(records),
    }


def main() -> None:
    lexical_rows, fresh_rows, rank1 = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_rank1_donor_free_margin_reflection_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "site": SITE,
        "rank": 1,
        "grid_points": GRID_POINTS,
        "calibration_rows": 8,
        "confirmation_rows": 120,
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "two_token_head_grid_evaluations": HEAD_GRID_EVALUATIONS,
        "transformer_backwards": 0,
        "model_updates": 0,
        "fit_parameters": 1,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    q = torch.tensor(rank1["basis"]["values_column_major"], device=backend.device, dtype=torch.float32)
    if q.shape != (1152,) or abs(float(q.norm()) - 1.0) > 1.0e-4 or hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != rank1["basis"]["sha256"]:
        raise ExperimentError("basis reconstruction failed")

    old = {family: [row for row in lexical_rows if row["family"] == family] for family in ("A1", "A2", "P", "C")}
    new = {family: [row for row in fresh_rows if row["family"] == family] for family in ("A1", "A2", "P", "C")}
    fit_rows = old["A1"][:8]
    head_ok, native_head_error = das.verify_head(backend, fit_rows, SITE)
    fit_base, fit_donor, _ = das.capture_site(backend, fit_rows, SITE)
    fit_coefficients = (fit_donor - fit_base) @ q
    budget = 1.25 * float(fit_coefficients.abs().max())
    if not math.isfinite(budget) or budget <= 0.0:
        raise ExperimentError("nonpositive/nonfinite actuator budget")
    full_head = das.head_logits(backend, fit_base)
    selected = selected_margin(backend, fit_base, [row["base_answer_id"] for row in fit_rows], [row["base_foil_id"] for row in fit_rows])
    exact_selected_head_error = max(abs(float(selected[i]) - float(full_head[i, row["base_answer_id"]] - full_head[i, row["base_foil_id"]])) for i, row in enumerate(fit_rows))
    forward_calls, evaluations = 3, 24

    panel_specs = (
        ("lexical_A1_heldout", old["A1"][8:], "A", "base"),
        ("lexical_A2", old["A2"], "A", "base"),
        ("fresh_A1", new["A1"], "A", "base"),
        ("fresh_A2", new["A2"], "A", "base"),
        ("lexical_P_noun_shifted_start", old["P"], "P", "donor"),
        ("fresh_P_noun_shifted_start", new["P"], "P", "donor"),
        ("lexical_C", old["C"], "C", "base"),
        ("fresh_C", new["C"], "C", "base"),
    )
    records = []
    capability = True
    target_scale = float(rank1["score"]["families"]["target_scale"])
    for panel, rows, kind, source_side in panel_specs:
        for chunk in producer._chunks(rows, 8):
            base, donor, _ = das.capture_site(backend, chunk, SITE)
            forward_calls += 2
            evaluations += 2 * len(chunk)
            if kind == "A":
                source = base
                target_ids = [row["donor_answer_id"] for row in chunk]
                foil_ids = [row["donor_foil_id"] for row in chunk]
                donor_margin = selected_margin(backend, donor, target_ids, foil_ids)
                base_native_margin = selected_margin(backend, base, [row["base_answer_id"] for row in chunk], [row["base_foil_id"] for row in chunk])
                donor_native_margin = donor_margin
                capability = capability and bool((base_native_margin > 0.0).all()) and bool((donor_native_margin > 0.0).all())
            elif kind == "P":
                source = donor
                target_ids = [row["base_foil_id"] for row in chunk]
                foil_ids = [row["base_answer_id"] for row in chunk]
                original_margin = selected_margin(backend, source, foil_ids, target_ids)
                capability = capability and bool((original_margin > 0.0).all())
                donor_margin = None
            else:
                source = base
                target_ids = [row["base_foil_id"] for row in chunk]
                foil_ids = [row["base_answer_id"] for row in chunk]
                original_margin = selected_margin(backend, source, foil_ids, target_ids)
                capability = capability and bool((original_margin > 0.0).all())
                donor_margin = None

            base_margin, patched_margin, alpha, grid_index = actuate(backend, source, q, target_ids, foil_ids, budget)
            for i, row in enumerate(chunk):
                reflection_denominator = -2.0 * float(base_margin[i])
                reflection = (float(patched_margin[i]) - float(base_margin[i])) / reflection_denominator if abs(reflection_denominator) > 1.0e-6 else float("nan")
                record = {
                    "panel": panel,
                    "kind": kind,
                    "row_id": str(row["row_id"]),
                    "source_side": source_side,
                    "base_target_margin": float(base_margin[i]),
                    "patched_target_margin": float(patched_margin[i]),
                    "alpha": float(alpha[i]),
                    "grid_index": int(grid_index[i]),
                    "margin_reflection_fraction": reflection,
                    "confirmation_donor_used_by_actuator": False,
                }
                if kind == "A":
                    denominator = float(donor_margin[i]) - float(base_margin[i])
                    record["donor_reference_margin"] = float(donor_margin[i])
                    record["recovery"] = (float(patched_margin[i]) - float(base_margin[i])) / denominator if abs(denominator) > 1.0e-6 else float("nan")
                elif kind == "C":
                    record["normalized_unrelated_effect"] = abs(float(patched_margin[i]) - float(base_margin[i])) / target_scale
                records.append(record)

    by_panel = {panel: [record for record in records if record["panel"] == panel] for panel, *_ in panel_specs}
    summaries = {}
    for panel, _rows, kind, _side in panel_specs:
        key = "recovery" if kind == "A" else ("margin_reflection_fraction" if kind == "P" else "normalized_unrelated_effect")
        summaries[panel] = summarize(by_panel[panel], key)

    pred_a = capability and head_ok and native_head_error <= 1.0e-3 and exact_selected_head_error <= 1.0e-4
    pred_b = (
        math.isfinite(budget) and budget > 0.0 and GRID_POINTS == 257
        and all(record["confirmation_donor_used_by_actuator"] is False for record in records)
        and all(math.isfinite(record["alpha"]) and abs(record["alpha"]) <= budget * (1.0 + 1.0e-6) for record in records)
    )
    pred_c = summaries["lexical_A1_heldout"]["mean_recovery"] >= 0.50 and summaries["lexical_A1_heldout"]["direction_fraction"] >= 0.75
    pred_d = all(summaries[panel]["mean_recovery"] >= 0.50 and summaries[panel]["direction_fraction"] >= 0.75 for panel in ("lexical_A2", "fresh_A1", "fresh_A2"))
    pred_e = all(summaries[panel]["mean_margin_reflection_fraction"] >= 0.50 and summaries[panel]["direction_fraction"] >= 0.75 for panel in ("lexical_P_noun_shifted_start", "fresh_P_noun_shifted_start"))
    pred_f = all(summaries[panel]["mean_normalized_unrelated_effect"] <= 0.20 for panel in ("lexical_C", "fresh_C"))
    pred_g = len(records) == 120 and len({(record["panel"], record["row_id"]) for record in records}) == 120 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX and GRID_POINTS * len(records) == HEAD_GRID_EVALUATIONS
    predictions = {
        "pred_a_authority_capability_and_exact_head": pred_a,
        "pred_b_budget_and_actuator_well_formed": pred_b,
        "pred_c_heldout_lexical_actuation": pred_c,
        "pred_d_cross_construction_actuation": pred_d,
        "pred_e_noun_shift_generalization": pred_e,
        "pred_f_unrelated_output_selectivity": pred_f,
        "pred_g_exact_coverage_and_price": pred_g,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_g else "invalid")
    reason = {"screen": "rank1_carrier_supports_donor_free_selective_margin_actuation", "null": "donor_free_actuation_fails_transfer_noun_shift_or_selectivity", "invalid": "authority_capability_head_actuator_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_rank1_donor_free_margin_reflection_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "basis_sha256": rank1["basis"]["sha256"],
        "actuator": {"site": SITE, "grid_points": GRID_POINTS, "budget": budget, "fit_coefficients": [float(value) for value in fit_coefficients], "confirmation_donor_activation_used": False, "calibrated_scalar_count": 1},
        "head_control": {"native_max_abs_difference": native_head_error, "selected_token_vs_full_head_max_abs_difference": exact_selected_head_error},
        "score": {"panels": summaries, "target_scale": target_scale, "forward_calls": forward_calls, "example_evaluations": evaluations, "two_token_head_grid_evaluations": GRID_POINTS * len(records), "record_count": len(records), "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 1},
        "predictions": predictions,
        "intervention_records": records,
        "terminal": terminal,
        "reason": reason,
        "next_action": "compile and test the donor-free actuator as program v10" if terminal == "screen" else "retain donor-dependent carrier projection and test earlier operational interchange",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "budget": budget, "panels": summaries, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
