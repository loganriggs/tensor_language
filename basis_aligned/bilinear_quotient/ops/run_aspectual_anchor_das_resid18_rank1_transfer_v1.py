#!/usr/bin/env python3
"""Rank-one DAS carrier test for aspectual anchor at resid18."""

# BQGATE: EXPERIMENT pred_a_authority_head_and_rank pred_b_heldout_lexical_a1 pred_c_cross_construction_lexical_a2 pred_d_prospective_construction_transfer pred_e_same_answer_selectivity pred_f_exact_coverage
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import circuit_candidate_aspectual_fresh_construction_v2 as fresh
import circuit_candidate_aspectual_lexical_holdout_v5 as lexical
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_das_resid18_rank1_transfer_v1.json"
LEXICAL_RESULT = ROOT / "circuits/followups/aspectual_anchor_explicit_path_lexical_holdout_v2_result.json"
FRESH_RESULT = ROOT / "circuits/followups/aspectual_anchor_program_v7_fresh_construction_transfer_v1_result.json"
LEXICAL_BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
FRESH_BUILDER = ROOT / "ops/circuit_candidate_aspectual_fresh_construction_v2.py"
DAS_LIBRARY = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.das_resid18_rank1_transfer_v1"
EXPECTED_PRIOR_SHA256 = "64b7f44abee69d3ceebd45896e7595199218773976e4f9579457155e6c099db3"
EXPECTED = {
    LEXICAL_RESULT: "fd1b4ae15e1d327001c8b172bcbecb0f15609d6da01bec8c8dddbf8de107549e",
    FRESH_RESULT: "8e1ea19c94ef269d2e9c7c0577568a0e1fe2e8bc6640e016377270df2dc68129",
    LEXICAL_BUILDER: "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859",
    FRESH_BUILDER: "848332a12c22bf523573e015b6f8f0a38b5865db8b77434dcbe6a176d98370ac",
    DAS_LIBRARY: "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
}
SITE = "resid:18"
RANK = 1
STEPS = 300
MODEL_FORWARDS_MAX = 19
EXAMPLE_EVALUATIONS_MAX = 264


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
    lexical_result = json.loads(LEXICAL_RESULT.read_text())
    fresh_result = json.loads(FRESH_RESULT.read_text())
    lexical_rows, fresh_rows = lexical.build_rows(), fresh.build_rows()
    if (
        prior.get("candidate_id") != CANDIDATE_ID or prior["frozen_design"]["rank"] != RANK
        or lexical_result.get("terminal") != fresh_result.get("terminal") or lexical_result.get("terminal") != "screen"
        or lexical.validate_rows(lexical_rows) != "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
        or fresh.validate_rows(fresh_rows) != "3c30019fdcc087c0e7410cd82d02458307bc6987ff9d23349dcf97d076f797d7"
    ):
        raise ExperimentError("authority, terminal, row hash, or rank changed")
    return lexical_rows, fresh_rows


def main() -> None:
    lexical_rows, fresh_rows = validate_static()
    plan = {
        "schema": "aspectual_anchor_das_resid18_rank1_transfer_dryrun_v1", "candidate_id": CANDIDATE_ID,
        "site": SITE, "rank": RANK, "steps": STEPS, "execution_policy": "managed_queue_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "transformer_backwards": 0, "model_updates": 0, "final_head_gradient_steps": STEPS, "fit_parameters": 1152,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    old = {family: [row for row in lexical_rows if row["family"] == family] for family in ("A1", "A2", "P", "C")}
    new = {family: [row for row in fresh_rows if row["family"] == family] for family in ("A1", "A2", "P", "C")}
    fit_rows, held_rows = old["A1"][:8], old["A1"][8:]

    head_ok, head_error = das.verify_head(backend, fit_rows, SITE)
    forward_calls, evaluations = 1, len(fit_rows)
    if not head_ok:
        raise ExperimentError(f"exact head verification failed: {head_error}")
    base_fit, donor_fit, _site = das.capture_site(backend, fit_rows, SITE)
    forward_calls += 2
    evaluations += 2 * len(fit_rows)
    q = das.fit_subspace(
        backend, base_fit, donor_fit,
        [row["donor_answer_id"] for row in fit_rows], [row["donor_foil_id"] for row in fit_rows],
        rank=RANK, steps=STEPS,
    )

    base_held, donor_held, _ = das.capture_site(backend, held_rows, SITE)
    forward_calls += 2
    evaluations += 2 * len(held_rows)
    scale = das.target_scale(
        backend, base_held, donor_held,
        [row["donor_answer_id"] for row in held_rows], [row["donor_foil_id"] for row in held_rows],
    )
    report = {}
    answer_tests = (
        ("lexical_A1_heldout", held_rows, base_held, donor_held),
        ("lexical_A2", old["A2"], None, None),
        ("fresh_A1", new["A1"], None, None),
        ("fresh_A2", new["A2"], None, None),
    )
    for name, rows, captured_base, captured_donor in answer_tests:
        if captured_base is None:
            captured_base, captured_donor, _ = das.capture_site(backend, rows, SITE)
            forward_calls += 2
            evaluations += 2 * len(rows)
        mean, absolute, count = das.subspace_recovery(
            backend, captured_base, captured_donor, q,
            [row["donor_answer_id"] for row in rows], [row["donor_foil_id"] for row in rows],
        )
        report[name] = {"mean_recovery": mean, "mean_absolute_recovery": absolute, "rows": count}
    for panel_name, families in (("lexical", old), ("fresh", new)):
        for family in ("P", "C"):
            rows = families[family]
            base, donor, _ = das.capture_site(backend, rows, SITE)
            forward_calls += 2
            evaluations += 2 * len(rows)
            effect, count = das.subspace_same_answer_effect(
                backend, base, donor, q,
                [row["donor_answer_id"] for row in rows], [row["donor_foil_id"] for row in rows], scale,
            )
            report[f"{panel_name}_{family}"] = {"same_answer_effect": effect, "rows": count}
    report["target_scale"] = scale

    basis = q.detach().cpu().reshape(-1)
    basis_values = [float(value) for value in basis]
    basis_sha256 = hashlib.sha256(basis.numpy().tobytes()).hexdigest()
    pred_a = head_ok and head_error <= 1.0e-3 and RANK == 1 and q.shape == (1152, 1)
    pred_b = report["lexical_A1_heldout"]["mean_absolute_recovery"] >= 0.50
    pred_c = report["lexical_A2"]["mean_absolute_recovery"] >= 0.50
    pred_d = report["fresh_A1"]["mean_absolute_recovery"] >= 0.50 and report["fresh_A2"]["mean_absolute_recovery"] >= 0.50
    pred_e = all(report[f"{panel}_{family}"]["same_answer_effect"] <= 0.20 for panel in ("lexical", "fresh") for family in ("P", "C"))
    pred_f = forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX and len(basis_values) == 1152
    predictions = {"pred_a_authority_head_and_rank": pred_a, "pred_b_heldout_lexical_a1": pred_b, "pred_c_cross_construction_lexical_a2": pred_c, "pred_d_prospective_construction_transfer": pred_d, "pred_e_same_answer_selectivity": pred_e, "pred_f_exact_coverage": pred_f}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_f else "invalid")
    reason = {"screen": "rank1_resid18_direction_transfers_selectively", "null": "rank1_direction_fails_transfer_or_selectivity", "invalid": "authority_head_rank_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_das_resid18_rank1_transfer_result_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "site": SITE, "rank": RANK, "steps": STEPS,
        "head_verification": {"passed": head_ok, "max_abs_difference": head_error},
        "basis": {"shape": [1152, 1], "dtype": "float32", "sha256": basis_sha256, "values_column_major": basis_values},
        "score": {"families": report, "forward_calls": forward_calls, "example_evaluations": evaluations, "transformer_backwards": 0, "model_updates": 0, "final_head_gradient_steps": STEPS, "fit_parameters": 1152},
        "predictions": predictions, "terminal": terminal, "reason": reason,
        "next_action": "test whether v8 displacement lies primarily in the released rank1 carrier" if terminal == "screen" else "retain full resid18 state and do not raise rank",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "families": report, "basis_sha256": basis_sha256, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
