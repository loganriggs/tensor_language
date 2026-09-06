#!/usr/bin/env python3
"""Donor-free additive actuation through the selective is/was writer."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_exact_head pred_b_budget_and_donor_free_actuator pred_c_heldout_and_cross_lexicon_A pred_d_answer_preserving_generalization pred_e_unrelated_output_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_candidate_aspectual_different_readout_is_was_v2 as v2
import circuit_candidate_aspectual_different_readout_is_was_v3 as v3
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_rank1_donor_free_margin_reflection_v1 as actuator


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_rank1_donor_free_margin_reflection_v1.json"
Q_IS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
V2_CAP = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v2_capability_result.json"
V3_CAP = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v3_capability_result.json"
FACTOR = ROOT / "circuits/followups/aspectual_tense_projected_response_factor_audit_v1_result.json"
V2_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v2.py"
V3_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v3.py"
ACTUATOR_LIBRARY = ROOT / "ops/run_aspectual_anchor_rank1_donor_free_margin_reflection_v1.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_rank1_donor_free_margin_reflection_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.rank1_donor_free_margin_reflection_v1"
SITE = "resid:18"
GRID_POINTS = 257
MODEL_FORWARDS_MAX = 33
EXAMPLE_EVALUATIONS_MAX = 264
HEAD_GRID_EVALUATIONS = 30_840
EXPECTED_PRIOR_SHA256 = "6f8e2b18dccd52230c56001379a90525cd818e660f0eaafac20504b15846232a"
EXPECTED = {
    Q_IS: "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    V2_CAP: "f76fbcd6174cc9e8e3f77352ee5461815156cdae421ecc43a0e0c3576b63af7e",
    V3_CAP: "744d2fd3c8200ca00005357961df3d435a7a13dbdbd7c3a51a487daee76acec3",
    FACTOR: "28a9a49db15e2676d7c2a0a1c6b3ca039dc359fb91fdb47994cda2eb6a85bfc0",
    V2_BUILDER: "4ed62d06e2ffe5c471efc70eeaa35c7524a66923f19c64c803faef7200d4a62f",
    V3_BUILDER: "ac240f13478168948814f59c0425270dd345ba03053332b72b34857e9a022638",
    ACTUATOR_LIBRARY: "47c2a56163381e7a22e7f2d9c84189ff45fbb3c8cc147b9e4bc6a3b0e6a9b310",
}


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped(rows):
    return {family: [row for row in rows if row["family"] == family] for family in ("A1", "A2", "P", "C")}


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    qi = json.loads(Q_IS.read_text())
    caps = [json.loads(V2_CAP.read_text()), json.loads(V3_CAP.read_text())]
    factor = json.loads(FACTOR.read_text())
    rows2, rows3 = v2.build_rows(), v3.build_rows()
    ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and prior["frozen_design"]["grid_points"] == GRID_POINTS
        and actuator.GRID_POINTS == GRID_POINTS
        and qi.get("terminal") == "screen"
        and qi["basis"]["shape"] == [1152, 1]
        and qi["basis"]["sha256"] == "e83ca8d0a89b170edcd334123bd6b25a8f18c39b1e441e4321f2fa96c29d5e1b"
        and all(result.get("terminal") == "screen" and all(cell["passed"] for cell in result["capability_cells"]) for result in caps)
        and factor.get("terminal") == "screen"
        and v2.validate_rows(rows2) == "fef8174cd57a1382d87ae47ddb06ddebd8059e96e4f71472525a066a3048f911"
        and v3.validate_rows(rows3) == "35dc18a4e95764bc2126fc18d672d2f7666e4bc1c84a7bcc06d1299b820d7ad2"
    )
    if not ok:
        raise ExperimentError("candidate, grid, basis, capability, factor, or rows changed")
    return rows2, rows3, qi


def summarize(records, key):
    values = [record[key] for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError(f"missing/nonfinite {key}")
    return {
        "count": len(values),
        f"mean_{key}": statistics.fmean(values),
        f"mean_absolute_{key}": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
        "endpoint_fraction": sum(record["grid_index"] in (0, GRID_POINTS - 1) for record in records) / len(records),
    }


def main():
    rows2, rows3, qi_result = validate_static()
    plan = {
        "schema": "tense_auxiliary_is_was_rank1_donor_free_margin_reflection_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "site": SITE, "rank": 1, "grid_points": GRID_POINTS,
        "execution_policy": "managed_queue_only", "gpu_accessed": False, "model_loaded": False,
        "queue_touched": False, "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "calibration_rows": 8, "confirmation_rows": 120,
        "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "selected_head_grid_evaluations": HEAD_GRID_EVALUATIONS,
        "transformer_backwards": 0, "model_updates": 0, "calibrated_scalars": 1,
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
    q = torch.as_tensor(qi_result["basis"]["values_column_major"], device=backend.device, dtype=torch.float32)
    if q.shape != (1152,) or abs(float(q.norm()) - 1.0) > 1.0e-4 or hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != qi_result["basis"]["sha256"]:
        raise ExperimentError("basis reconstruction failed")
    by2, by3 = grouped(rows2), grouped(rows3)
    fit_rows = by2["A1"][:8]
    head_ok, head_error = das.verify_head(backend, fit_rows, SITE)
    fit_base, fit_donor, _ = das.capture_site(backend, fit_rows, SITE)
    coefficients = (fit_donor - fit_base) @ q
    budget = 1.25 * float(coefficients.abs().max())
    if not math.isfinite(budget) or budget <= 0.0:
        raise ExperimentError("nonpositive/nonfinite budget")
    full_head = das.head_logits(backend, fit_base)
    selected = actuator.selected_margin(backend, fit_base, [row["base_answer_id"] for row in fit_rows], [row["base_foil_id"] for row in fit_rows])
    selected_head_error = max(abs(float(selected[i]) - float(full_head[i, row["base_answer_id"]] - full_head[i, row["base_foil_id"]])) for i, row in enumerate(fit_rows))
    forward_calls, evaluations = 3, 24
    target_scale = float(qi_result["score"]["families"]["target_scale"])

    panels = (
        ("v2_A1_heldout", by2["A1"][8:], "A"),
        ("v2_A2", by2["A2"], "A"),
        ("v3_A1", by3["A1"], "A"),
        ("v3_A2", by3["A2"], "A"),
        ("v2_P", by2["P"], "P"),
        ("v3_P", by3["P"], "P"),
        ("v2_C", by2["C"], "C"),
        ("v3_C", by3["C"], "C"),
    )
    records = []
    capability = True
    for panel, rows, kind in panels:
        for chunk in producer._chunks(rows, 8):
            base, donor, _ = das.capture_site(backend, chunk, SITE)
            forward_calls += 2
            evaluations += 2 * len(chunk)
            if kind == "A":
                source = base
                target_ids = [row["donor_answer_id"] for row in chunk]
                foil_ids = [row["donor_foil_id"] for row in chunk]
                donor_margin = actuator.selected_margin(backend, donor, target_ids, foil_ids)
                capability = capability and bool((actuator.selected_margin(backend, base, [row["base_answer_id"] for row in chunk], [row["base_foil_id"] for row in chunk]) > 0.0).all()) and bool((donor_margin > 0.0).all())
            elif kind == "P":
                source = donor
                target_ids = [row["base_foil_id"] for row in chunk]
                foil_ids = [row["base_answer_id"] for row in chunk]
                donor_margin = None
                capability = capability and bool((actuator.selected_margin(backend, source, foil_ids, target_ids) > 0.0).all())
            else:
                source = base
                target_ids = [row["base_foil_id"] for row in chunk]
                foil_ids = [row["base_answer_id"] for row in chunk]
                donor_margin = None
                capability = capability and bool((actuator.selected_margin(backend, source, foil_ids, target_ids) > 0.0).all())
            base_margin, patched_margin, alpha, grid_index = actuator.actuate(backend, source, q, target_ids, foil_ids, budget)
            for i, row in enumerate(chunk):
                record = {
                    "panel": panel, "kind": kind, "row_id": str(row["row_id"]),
                    "base_target_margin": float(base_margin[i]), "patched_target_margin": float(patched_margin[i]),
                    "alpha": float(alpha[i]), "grid_index": int(grid_index[i]),
                    "confirmation_donor_activation_used_by_actuator": False,
                    "confirmation_donor_margin_used_by_actuator": False,
                    "row_outcome_used_by_actuator": False,
                }
                if kind == "A":
                    denominator = float(donor_margin[i]) - float(base_margin[i])
                    record["donor_reference_margin"] = float(donor_margin[i])
                    record["recovery"] = (float(patched_margin[i]) - float(base_margin[i])) / denominator if abs(denominator) > 1.0e-6 else float("nan")
                elif kind == "P":
                    record["margin_reflection_fraction"] = (float(patched_margin[i]) - float(base_margin[i])) / (-2.0 * float(base_margin[i]))
                else:
                    record["normalized_unrelated_effect"] = abs(float(patched_margin[i]) - float(base_margin[i])) / target_scale
                records.append(record)

    by_panel = {name: [record for record in records if record["panel"] == name] for name, _, _ in panels}
    summaries = {}
    for name, _rows, kind in panels:
        key = "recovery" if kind == "A" else ("margin_reflection_fraction" if kind == "P" else "normalized_unrelated_effect")
        summaries[name] = summarize(by_panel[name], key)
    pred_a = capability and head_ok and head_error <= 1.0e-3 and selected_head_error <= 1.0e-4
    pred_b = (
        math.isfinite(budget) and budget > 0.0
        and all(math.isfinite(record["alpha"]) and abs(record["alpha"]) <= budget * (1.0 + 1.0e-6) for record in records)
        and all(not record["confirmation_donor_activation_used_by_actuator"] and not record["confirmation_donor_margin_used_by_actuator"] and not record["row_outcome_used_by_actuator"] for record in records)
    )
    pred_c = all(summaries[name]["mean_recovery"] >= 0.75 and summaries[name]["direction_fraction"] >= 0.75 for name in ("v2_A1_heldout", "v2_A2", "v3_A1", "v3_A2"))
    pred_d = all(summaries[name]["mean_margin_reflection_fraction"] >= 0.75 and summaries[name]["direction_fraction"] >= 0.75 for name in ("v2_P", "v3_P"))
    pred_e = all(summaries[name]["mean_normalized_unrelated_effect"] <= 0.20 for name in ("v2_C", "v3_C"))
    pred_f = len(records) == 120 and len({(record["panel"], record["row_id"]) for record in records}) == 120 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX and GRID_POINTS * len(records) == HEAD_GRID_EVALUATIONS
    predictions = {
        "pred_a_authority_capability_and_exact_head": pred_a,
        "pred_b_budget_and_donor_free_actuator": pred_b,
        "pred_c_heldout_and_cross_lexicon_A": pred_c,
        "pred_d_answer_preserving_generalization": pred_d,
        "pred_e_unrelated_output_selectivity": pred_e,
        "pred_f_exact_coverage_and_price": pred_f,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_f else "invalid")
    reason = {"screen": "q_is_supports_donor_free_selective_additive_actuation", "null": "additive_actuator_fails_A_P_transfer_or_C_selectivity", "invalid": "authority_capability_head_basis_actuator_or_coverage_invalid"}[terminal]
    result = {
        "schema": "tense_auxiliary_is_was_rank1_donor_free_margin_reflection_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "basis_sha256": qi_result["basis"]["sha256"],
        "actuator": {"site": SITE, "grid_points": GRID_POINTS, "budget": budget, "fit_projection_coefficients": [float(value) for value in coefficients], "confirmation_donor_activation_used": False, "calibrated_scalar_count": 1},
        "head_control": {"native_max_abs_difference": head_error, "selected_token_vs_full_head_max_abs_difference": selected_head_error},
        "score": {"panels": summaries, "target_scale": target_scale, "forward_calls": forward_calls, "example_evaluations": evaluations, "selected_head_grid_evaluations": GRID_POINTS * len(records), "record_count": len(records), "transformer_backwards": 0, "model_updates": 0, "calibrated_scalars": 1},
        "predictions": predictions, "intervention_records": records,
        "terminal": terminal, "reason": reason,
        "next_action": "fit a fixed local resid10 is/was read to target-guided alpha, then test on a capability-qualified fresh lexicon" if terminal == "screen" else "retain projected q_is only and do not raise rank",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "budget": budget, "panels": summaries, "price": result["score"], "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
