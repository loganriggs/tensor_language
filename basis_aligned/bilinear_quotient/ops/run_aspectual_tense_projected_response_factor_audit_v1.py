#!/usr/bin/env python3
"""Factor projected writer response into displacement and observability terms."""

# BQGATE: EXPERIMENT pred_a_authority_head_basis_and_finite_route pred_b_local_first_order_adequacy pred_c_q_has_cross_bottleneck pred_d_q_is_cross_bottleneck pred_e_exact_coverage
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
import circuit_candidate_aspectual_lexical_holdout_v5 as has_had
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_projected_response_factor_audit_v1.json"
MATRIX = ROOT / "circuits/followups/aspectual_tense_matched_projected_writer_response_matrix_v1_result.json"
Q_HAS = ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json"
Q_IS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
HAS_BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
V2_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v2.py"
V3_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v3.py"
DAS_LIBRARY = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/aspectual_tense_projected_response_factor_audit_v1_result.json"
CANDIDATE_ID = "aspectual_tense.projected_response_factor_audit_v1"
SITE = "resid:18"
EXPECTED_PRIOR_SHA256 = "ca5fe4aa1d048ed1a61aa6d05155d5602cc79f3bce4fbfc2241f6e9466241eba"
EXPECTED = {
    MATRIX: "b78b287698d1365a3e8b2f8e3266907eb11dfe8f4510101048fb1d59ab71e9e3",
    Q_HAS: "58b83a2714ae8d53cc799d5e6ae96c61cc476f22e09019e6e1f620581ff9a278",
    Q_IS: "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    HAS_BUILDER: "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859",
    V2_BUILDER: "4ed62d06e2ffe5c471efc70eeaa35c7524a66923f19c64c803faef7200d4a62f",
    V3_BUILDER: "ac240f13478168948814f59c0425270dd345ba03053332b72b34857e9a022638",
    DAS_LIBRARY: "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
}
MODEL_FORWARDS_EXACT = 8
EXAMPLE_EVALUATIONS_EXACT = 400
FINAL_HEAD_GRADIENT_BATCHES_EXACT = 3
ROUTE_TOLERANCE = 1.0e-5


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
    matrix = json.loads(MATRIX.read_text())
    qh, qi = json.loads(Q_HAS.read_text()), json.loads(Q_IS.read_text())
    rows_h, rows2, rows3 = has_had.build_rows(), v2.build_rows(), v3.build_rows()
    ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and prior["frozen_design"]["no_fit"] is True
        and matrix.get("terminal") == "null"
        and matrix["predictions"]["pred_a_authority_head_basis_and_route_agreement"] is True
        and matrix["predictions"]["pred_e_exact_coverage"] is True
        and qh["basis"]["shape"] == qi["basis"]["shape"] == [1152, 1]
        and has_had.validate_rows(rows_h) == "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
        and v2.validate_rows(rows2) == "fef8174cd57a1382d87ae47ddb06ddebd8059e96e4f71472525a066a3048f911"
        and v3.validate_rows(rows3) == "35dc18a4e95764bc2126fc18d672d2f7666e4bc1c84a7bcc06d1299b820d7ad2"
    )
    if not ok:
        raise ExperimentError("candidate, matrix, basis, terminal, or rows changed")
    return rows_h, rows2, rows3, qh, qi, matrix


def capture_population(backend, rows):
    families = grouped(rows)
    ordered, spans = [], {}
    for family in ("A1", "A2", "P", "C"):
        start = len(ordered)
        ordered.extend(families[family])
        spans[family] = slice(start, len(ordered))
    base, donor, _ = das.capture_site(backend, ordered, SITE)
    return families, ordered, base, donor, spans


def margin(backend, x, rows):
    torch = backend.torch
    index = torch.arange(len(rows), device=backend.device)
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device, dtype=torch.long)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device, dtype=torch.long)
    logits = das.head_logits(backend, x)
    return logits[index, answer] - logits[index, foil]


def median_abs(values):
    return float(statistics.median([abs(value) for value in values.detach().cpu().tolist()]))


def pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    return float((x @ y / denom).item()) if float(denom) > 1.0e-12 else 0.0


def factor_population(backend, families, base, donor, spans, writers, scale):
    torch = backend.torch
    a_rows = families["A1"] + families["A2"]
    a_start, a_stop = spans["A1"].start, spans["A2"].stop
    x = base[a_start:a_stop].detach().clone().requires_grad_(True)
    d = donor[a_start:a_stop].detach()
    base_margin = margin(backend, x, a_rows)
    gradient = torch.autograd.grad(base_margin.sum(), x, create_graph=False)[0].detach()
    base_margin = base_margin.detach()
    donor_margin = margin(backend, d, a_rows).detach()
    delta = d - x.detach()
    result = {}
    for writer_name, q in writers.items():
        writer_report = {}
        offset = 0
        for family in ("A1", "A2"):
            count = len(families[family])
            sl = slice(offset, offset + count)
            beta = (delta[sl] @ q).reshape(-1)
            g = (gradient[sl] @ q).reshape(-1)
            linear = beta * g
            patched = x.detach()[sl] + beta[:, None] * q.T
            finite = margin(backend, patched, families[family]) - base_margin[sl]
            native = donor_margin[sl] - base_margin[sl]
            keep = native.abs() > 1.0e-6
            recovery = (finite[keep] / native[keep]).abs().mean()
            relative_l2 = float((linear - finite).norm() / finite.norm().clamp_min(1.0e-12))
            writer_report[family] = {
                "rows": count,
                "median_abs_beta": median_abs(beta),
                "median_abs_g": median_abs(g),
                "median_abs_g_normalized": median_abs(g) / scale,
                "mean_abs_finite_response": float(finite.abs().mean()),
                "mean_abs_first_order_response": float(linear.abs().mean()),
                "finite_mean_absolute_recovery": float(recovery),
                "first_order_finite_pearson": pearson(linear, finite),
                "first_order_relative_l2": relative_l2,
            }
            offset += count
        result[writer_name] = writer_report
    return result, 1


def aggregate_metrics(report, writer, populations):
    beta, g = [], []
    for population in populations:
        for family in ("A1", "A2"):
            beta.append(report[population][writer][family]["median_abs_beta"])
            g.append(report[population][writer][family]["median_abs_g_normalized"])
    return {"median_abs_beta": statistics.mean(beta), "median_abs_g_normalized": statistics.mean(g)}


def classify(beta_ratio, g_ratio):
    beta = "weak" if beta_ratio <= 0.50 else ("retained" if beta_ratio >= 0.75 else "intermediate")
    g = "weak" if g_ratio <= 0.50 else ("retained" if g_ratio >= 0.75 else "intermediate")
    if beta == "weak" and g == "retained":
        label = "donor_displacement_controllability"
    elif beta == "retained" and g == "weak":
        label = "output_observability"
    elif beta == "weak" and g == "weak":
        label = "mixed"
    elif beta == "retained" and g == "retained":
        label = "finite_dose_or_interaction"
    else:
        label = "intermediate"
    return {"beta_ratio": beta_ratio, "g_ratio": g_ratio, "beta_state": beta, "g_state": g, "classification": label}


def main():
    rows_h, rows2, rows3, qh_result, qi_result, matrix = validate_static()
    plan = {
        "schema": "aspectual_tense_projected_response_factor_audit_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "site": SITE, "execution_policy": "managed_queue_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "model_forwards_exact": MODEL_FORWARDS_EXACT,
        "example_evaluations_exact": EXAMPLE_EVALUATIONS_EXACT,
        "final_head_gradient_batches_exact": FINAL_HEAD_GRADIENT_BATCHES_EXACT,
        "fit_steps": 0, "fit_parameters": 0, "transformer_backwards": 0,
        "model_updates": 0, "parameter_grid_points": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    writers = {
        "q_has": backend.torch.as_tensor(qh_result["basis"]["values_column_major"], device=backend.device, dtype=backend.torch.float32).reshape(1152, 1),
        "q_is": backend.torch.as_tensor(qi_result["basis"]["values_column_major"], device=backend.device, dtype=backend.torch.float32).reshape(1152, 1),
    }
    head_h_ok, head_h_error = das.verify_head(backend, grouped(rows_h)["A1"][:8], SITE)
    head_i_ok, head_i_error = das.verify_head(backend, grouped(rows2)["A1"][:8], SITE)
    forward_calls, evaluations = 2, 16
    gradient_batches = 0
    factors, scales = {}, {}
    for name, rows in (("has_had_v5", rows_h), ("is_was_v2", rows2), ("is_was_v3", rows3)):
        families, ordered, base, donor, spans = capture_population(backend, rows)
        forward_calls += 2
        evaluations += 2 * len(ordered)
        a1_rows = families["A1"][:8]
        a1_span = slice(spans["A1"].start, spans["A1"].start + 8)
        scale = das.target_scale(
            backend, base[a1_span], donor[a1_span],
            [row["donor_answer_id"] for row in a1_rows], [row["donor_foil_id"] for row in a1_rows],
        )
        scales[name] = scale
        factors[name], batches = factor_population(backend, families, base, donor, spans, writers, scale)
        gradient_batches += batches

    route_diffs = {}
    for writer in ("q_has", "q_is"):
        for population in ("has_had_v5", "is_was_v2", "is_was_v3"):
            for family in ("A1", "A2"):
                key = f"{writer}:{population}:{family}"
                route_diffs[key] = abs(
                    factors[population][writer][family]["finite_mean_absolute_recovery"]
                    - matrix["response_matrix"][writer][population][family]["mean_absolute_recovery"]
                )
    max_route_difference = max(route_diffs.values())
    qh_native = aggregate_metrics(factors, "q_has", ["has_had_v5"])
    qh_cross = aggregate_metrics(factors, "q_has", ["is_was_v2", "is_was_v3"])
    qi_native = aggregate_metrics(factors, "q_is", ["is_was_v2", "is_was_v3"])
    qi_cross = aggregate_metrics(factors, "q_is", ["has_had_v5"])
    qh_class = classify(qh_cross["median_abs_beta"] / qh_native["median_abs_beta"], qh_cross["median_abs_g_normalized"] / qh_native["median_abs_g_normalized"])
    qi_class = classify(qi_cross["median_abs_beta"] / qi_native["median_abs_beta"], qi_cross["median_abs_g_normalized"] / qi_native["median_abs_g_normalized"])
    classifications = {"q_has_native_to_is_was": qh_class, "q_is_native_to_has_had": qi_class}

    all_cells = [factors[p][q][f] for p in factors for q in writers for f in ("A1", "A2")]
    pred_a = head_h_ok and head_i_ok and max(head_h_error, head_i_error) <= 1.0e-3 and max_route_difference <= ROUTE_TOLERANCE and all(q.shape == (1152, 1) for q in writers.values())
    pred_b = all(cell["first_order_finite_pearson"] >= 0.90 and cell["first_order_relative_l2"] <= 0.35 for cell in all_cells)
    pred_c = qh_class["beta_ratio"] <= 0.50 and qh_class["g_ratio"] >= 0.75
    pred_d = qi_class["beta_ratio"] <= 0.50 and qi_class["g_ratio"] >= 0.75
    pred_e = forward_calls == MODEL_FORWARDS_EXACT and evaluations == EXAMPLE_EVALUATIONS_EXACT and gradient_batches == FINAL_HEAD_GRADIENT_BATCHES_EXACT
    predictions = {
        "pred_a_authority_head_basis_and_finite_route": pred_a,
        "pred_b_local_first_order_adequacy": pred_b,
        "pred_c_q_has_cross_bottleneck": pred_c,
        "pred_d_q_is_cross_bottleneck": pred_d,
        "pred_e_exact_coverage": pred_e,
    }
    terminal = "screen" if pred_a and pred_e else "invalid"
    reason = "projected_response_factors_validly_classified" if terminal == "screen" else "authority_head_route_or_coverage_invalid"
    result = {
        "schema": "aspectual_tense_projected_response_factor_audit_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "site": SITE, "identity": "finite_response=m(x+beta*q)-m(x), beta=delta_dot_q, first_order=beta*grad_m_dot_q",
        "head_verification": {"has_had": {"passed": head_h_ok, "max_abs_difference": head_h_error}, "is_was": {"passed": head_i_ok, "max_abs_difference": head_i_error}},
        "population_scales": scales, "factors": factors,
        "aggregates": {"q_has_native": qh_native, "q_has_cross": qh_cross, "q_is_native": qi_native, "q_is_cross": qi_cross},
        "classifications": classifications,
        "finite_route_agreement": {"max_abs_difference": max_route_difference, "tolerance": ROUTE_TOLERANCE, "cell_differences": route_diffs},
        "price": {"model_forwards": forward_calls, "example_evaluations": evaluations, "final_head_gradient_batches": gradient_batches, "fit_steps": 0, "fit_parameters": 0, "transformer_backwards": 0, "model_updates": 0, "parameter_grid_points": 0},
        "predictions": predictions, "terminal": terminal, "reason": reason,
        "next_action": "update the circuit interface using the measured controllability/observability classification",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "classifications": classifications, "first_order_worst": {"minimum_pearson": min(cell["first_order_finite_pearson"] for cell in all_cells), "maximum_relative_l2": max(cell["first_order_relative_l2"] for cell in all_cells)}, "finite_route_max_abs_difference": max_route_difference, "price": result["price"], "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
