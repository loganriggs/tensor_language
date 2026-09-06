#!/usr/bin/env python3
"""Matched projected-interchange response matrix for q_has and q_is."""

# BQGATE: EXPERIMENT pred_a_authority_head_basis_and_route_agreement pred_b_q_has_projected_transfer_v2 pred_c_q_has_projected_transfer_v3 pred_d_matched_operator_asymmetry pred_e_exact_coverage
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import circuit_candidate_aspectual_different_readout_is_was_v2 as v2
import circuit_candidate_aspectual_different_readout_is_was_v3 as v3
import circuit_candidate_aspectual_lexical_holdout_v5 as has_had
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_matched_projected_writer_response_matrix_v1.json"
Q_HAS_RESULT = ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json"
Q_IS_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
HAS_BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
V2_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v2.py"
V3_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v3.py"
DAS_LIBRARY = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/aspectual_tense_matched_projected_writer_response_matrix_v1_result.json"
CANDIDATE_ID = "aspectual_tense.matched_projected_writer_response_matrix_v1"
SITE = "resid:18"
EXPECTED_PRIOR_SHA256 = "d921f44018c4b445a949d4fad3665234b9a5130311b81e1959e98431a5bfe253"
EXPECTED = {
    Q_HAS_RESULT: "58b83a2714ae8d53cc799d5e6ae96c61cc476f22e09019e6e1f620581ff9a278",
    Q_IS_RESULT: "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    HAS_BUILDER: "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859",
    V2_BUILDER: "4ed62d06e2ffe5c471efc70eeaa35c7524a66923f19c64c803faef7200d4a62f",
    V3_BUILDER: "ac240f13478168948814f59c0425270dd345ba03053332b72b34857e9a022638",
    DAS_LIBRARY: "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
}
MODEL_FORWARDS_EXACT = 8
EXAMPLE_EVALUATIONS_EXACT = 400
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
    qh = json.loads(Q_HAS_RESULT.read_text())
    qi = json.loads(Q_IS_RESULT.read_text())
    rows_h, rows2, rows3 = has_had.build_rows(), v2.build_rows(), v3.build_rows()
    ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and prior["frozen_design"]["no_fit"] is True
        and qh.get("terminal") == "screen"
        and qi.get("terminal") == "screen"
        and qh["basis"]["shape"] == qi["basis"]["shape"] == [1152, 1]
        and qh["basis"]["sha256"] == "123c6e098fcccf68bd9b881bb81c6b95858a258baa688b79a947a3043bb61e39"
        and qi["basis"]["sha256"] == "e83ca8d0a89b170edcd334123bd6b25a8f18c39b1e441e4321f2fa96c29d5e1b"
        and has_had.validate_rows(rows_h) == "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
        and v2.validate_rows(rows2) == "fef8174cd57a1382d87ae47ddb06ddebd8059e96e4f71472525a066a3048f911"
        and v3.validate_rows(rows3) == "35dc18a4e95764bc2126fc18d672d2f7666e4bc1c84a7bcc06d1299b820d7ad2"
    )
    if not ok:
        raise ExperimentError("candidate, no-fit authority, basis, terminal, or rows changed")
    return rows_h, rows2, rows3, qh, qi


def capture_population(backend, rows):
    families = grouped(rows)
    ordered = []
    spans = {}
    for family in ("A1", "A2", "P", "C"):
        start = len(ordered)
        ordered.extend(families[family])
        spans[family] = slice(start, len(ordered))
    base, donor, _ = das.capture_site(backend, ordered, SITE)
    return families, ordered, base, donor, spans


def population_scale(backend, families, base, donor, spans):
    rows = families["A1"][:8]
    span = slice(spans["A1"].start, spans["A1"].start + 8)
    return das.target_scale(
        backend, base[span], donor[span],
        [row["donor_answer_id"] for row in rows], [row["donor_foil_id"] for row in rows],
    )


def score_population(backend, families, base, donor, spans, q, scale):
    report = {}
    for family in ("A1", "A2"):
        rows, span = families[family], spans[family]
        mean, absolute, count = das.subspace_recovery(
            backend, base[span], donor[span], q,
            [row["donor_answer_id"] for row in rows], [row["donor_foil_id"] for row in rows],
        )
        report[family] = {"mean_recovery": mean, "mean_absolute_recovery": absolute, "rows": count}
    for family in ("P", "C"):
        rows, span = families[family], spans[family]
        effect, count = das.subspace_same_answer_effect(
            backend, base[span], donor[span], q,
            [row["donor_answer_id"] for row in rows], [row["donor_foil_id"] for row in rows], scale,
        )
        report[family] = {"same_answer_effect": effect, "rows": count}
    report["target_scale"] = scale
    return report


def route_score(backend, families, base, donor, spans, q, scale, selections):
    report = {}
    for family, sub in selections.items():
        all_rows = families[family]
        rows = all_rows[sub]
        start = spans[family].start + (sub.start or 0)
        stop = spans[family].start + (sub.stop if sub.stop is not None else len(all_rows))
        span = slice(start, stop)
        if family in ("A1", "A2"):
            mean, absolute, count = das.subspace_recovery(
                backend, base[span], donor[span], q,
                [row["donor_answer_id"] for row in rows], [row["donor_foil_id"] for row in rows],
            )
            report[family] = {"mean_recovery": mean, "mean_absolute_recovery": absolute, "rows": count}
        else:
            effect, count = das.subspace_same_answer_effect(
                backend, base[span], donor[span], q,
                [row["donor_answer_id"] for row in rows], [row["donor_foil_id"] for row in rows], scale,
            )
            report[family] = {"same_answer_effect": effect, "rows": count}
    return report


def max_route_difference(actual, expected, mapping):
    differences = []
    for actual_family, expected_name in mapping.items():
        value_key = "mean_absolute_recovery" if actual_family in ("A1", "A2") else "same_answer_effect"
        differences.append(abs(actual[actual_family][value_key] - expected[expected_name][value_key]))
    return max(differences)


def transfer_pass(report):
    return (
        report["A1"]["mean_absolute_recovery"] >= 0.25
        and report["A2"]["mean_absolute_recovery"] >= 0.25
        and report["P"]["same_answer_effect"] <= 0.20
        and report["C"]["same_answer_effect"] <= 0.20
    )


def main():
    rows_h, rows2, rows3, qh_result, qi_result = validate_static()
    plan = {
        "schema": "aspectual_tense_matched_projected_writer_response_matrix_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "site": SITE, "operator": "rank1_projected_interchange",
        "execution_policy": "managed_queue_only", "gpu_accessed": False, "model_loaded": False,
        "queue_touched": False, "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "model_forwards_exact": MODEL_FORWARDS_EXACT, "example_evaluations_exact": EXAMPLE_EVALUATIONS_EXACT,
        "fit_parameters": 0, "gradient_steps": 0, "transformer_backwards": 0,
        "model_updates": 0, "rank_grid_points": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    qh = backend.torch.as_tensor(qh_result["basis"]["values_column_major"], device=backend.device, dtype=backend.torch.float32).reshape(1152, 1)
    qi = backend.torch.as_tensor(qi_result["basis"]["values_column_major"], device=backend.device, dtype=backend.torch.float32).reshape(1152, 1)
    head_h_ok, head_h_error = das.verify_head(backend, grouped(rows_h)["A1"][:8], SITE)
    head_i_ok, head_i_error = das.verify_head(backend, grouped(rows2)["A1"][:8], SITE)
    forward_calls, evaluations = 2, 16

    populations = {}
    for name, rows in (("has_had_v5", rows_h), ("is_was_v2", rows2), ("is_was_v3", rows3)):
        families, ordered, base, donor, spans = capture_population(backend, rows)
        populations[name] = (families, base, donor, spans)
        forward_calls += 2
        evaluations += 2 * len(ordered)

    matrix = {"q_has": {}, "q_is": {}}
    scales = {}
    for population, (families, base, donor, spans) in populations.items():
        scale = population_scale(backend, families, base, donor, spans)
        scales[population] = scale
        matrix["q_has"][population] = score_population(backend, families, base, donor, spans, qh, scale)
        matrix["q_is"][population] = score_population(backend, families, base, donor, spans, qi, scale)

    has_families, has_base, has_donor, has_spans = populations["has_had_v5"]
    v2_families, v2_base, v2_donor, v2_spans = populations["is_was_v2"]
    v3_families, v3_base, v3_donor, v3_spans = populations["is_was_v3"]
    qh_route = route_score(backend, has_families, has_base, has_donor, has_spans, qh, qh_result["score"]["families"]["target_scale"], {"A1": slice(8, None), "A2": slice(None), "P": slice(None), "C": slice(None)})
    qi_v2_route = route_score(backend, v2_families, v2_base, v2_donor, v2_spans, qi, qi_result["score"]["families"]["target_scale"], {"A1": slice(8, None), "A2": slice(None), "P": slice(8, None), "C": slice(8, None)})
    qi_v3_route = route_score(backend, v3_families, v3_base, v3_donor, v3_spans, qi, qi_result["score"]["families"]["target_scale"], {"A1": slice(None), "A2": slice(None), "P": slice(None), "C": slice(None)})
    qi_has_route = route_score(backend, has_families, has_base, has_donor, has_spans, qi, qi_result["score"]["families"]["target_scale"], {"A1": slice(None), "A2": slice(None), "P": slice(None), "C": slice(None)})
    qh_diff = max_route_difference(qh_route, qh_result["score"]["families"], {"A1": "lexical_A1_heldout", "A2": "lexical_A2", "P": "lexical_P", "C": "lexical_C"})
    qi_v2_diff = max_route_difference(qi_v2_route, qi_result["score"]["families"], {"A1": "v2_A1_heldout", "A2": "v2_A2", "P": "v2_P_heldout", "C": "v2_C_heldout"})
    qi_v3_diff = max_route_difference(qi_v3_route, qi_result["score"]["families"], {"A1": "v3_A1", "A2": "v3_A2", "P": "v3_P", "C": "v3_C"})
    qi_has_diff = max_route_difference(qi_has_route, qi_result["reciprocal_has_had"], {"A1": "has_had_v5_A1", "A2": "has_had_v5_A2", "P": "has_had_v5_P", "C": "has_had_v5_C"})
    route = {"q_has_native_max_abs_difference": qh_diff, "q_is_v2_max_abs_difference": qi_v2_diff, "q_is_v3_max_abs_difference": qi_v3_diff, "q_is_has_had_max_abs_difference": qi_has_diff, "tolerance": ROUTE_TOLERANCE}

    max_route_difference_value = max(qh_diff, qi_v2_diff, qi_v3_diff, qi_has_diff)
    pred_a = head_h_ok and head_i_ok and max(head_h_error, head_i_error) <= 1.0e-3 and max_route_difference_value <= ROUTE_TOLERANCE and qh.shape == qi.shape == (1152, 1)
    pred_b = transfer_pass(matrix["q_has"]["is_was_v2"])
    pred_c = transfer_pass(matrix["q_has"]["is_was_v3"])
    pred_d = pred_b and pred_c and matrix["q_is"]["has_had_v5"]["A1"]["mean_absolute_recovery"] <= 0.20 and matrix["q_is"]["has_had_v5"]["A2"]["mean_absolute_recovery"] <= 0.20
    pred_e = forward_calls == MODEL_FORWARDS_EXACT and evaluations == EXAMPLE_EVALUATIONS_EXACT
    predictions = {
        "pred_a_authority_head_basis_and_route_agreement": pred_a,
        "pred_b_q_has_projected_transfer_v2": pred_b,
        "pred_c_q_has_projected_transfer_v3": pred_c,
        "pred_d_matched_operator_asymmetry": pred_d,
        "pred_e_exact_coverage": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_e else "invalid")
    reason = {"screen": "asymmetric_projected_writer_reuse", "null": "q_has_projected_cross_readout_transfer_misses_or_is_intermediate", "invalid": "authority_head_route_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_tense_matched_projected_writer_response_matrix_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "site": SITE, "operator": "rank1_projected_interchange", "population_scales": scales,
        "head_verification": {"has_had": {"passed": head_h_ok, "max_abs_difference": head_h_error}, "is_was": {"passed": head_i_ok, "max_abs_difference": head_i_error}},
        "basis": {"q_has_sha256": qh_result["basis"]["sha256"], "q_is_sha256": qi_result["basis"]["sha256"], "shape_each": [1152, 1]},
        "route_agreement": route, "response_matrix": matrix,
        "price": {"model_forwards": forward_calls, "example_evaluations": evaluations, "fit_parameters": 0, "gradient_steps": 0, "transformer_backwards": 0, "model_updates": 0, "rank_grid_points": 0},
        "predictions": predictions, "terminal": terminal, "reason": reason,
        "next_action": "separate output sensitivity from donor-displacement controllability" if terminal == "null" else "promote asymmetric projected writer reuse",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "q_has_cross_v2": matrix["q_has"]["is_was_v2"], "q_has_cross_v3": matrix["q_has"]["is_was_v3"], "q_is_cross_has": matrix["q_is"]["has_had_v5"], "route_agreement": route, "price": result["price"], "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
