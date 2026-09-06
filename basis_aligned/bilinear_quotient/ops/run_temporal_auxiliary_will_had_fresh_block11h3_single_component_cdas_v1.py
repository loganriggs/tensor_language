#!/usr/bin/env python3
"""Closure-valid constrained DAS inside fresh temporal block11H3."""

# BQGATE: EXPERIMENT pred_a_authority_and_identity_closure pred_b_exact_h3_is_material pred_c_optimizer_dominates_dim_on_fit_objective pred_d_cdas_axis_and_complement_transfer pred_e_selective_controls pred_f_restart_stability
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_fast_screen_candidate_polarity_state as polarity
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_unit_greedy as g
import single_component_das_eval as single

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_block11h3_single_component_cdas_v1.json"
H3_AUTHORITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_block11_h3_response_v1_result.json"
REDTEAM = ROOT / "circuits/followups/unit_subspace_trust_v4_redteam_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
UNIT_LIB = ROOT / "ops/circuit_unit_greedy.py"
SINGLE_LIB = ROOT / "ops/single_component_das_eval.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_block11h3_single_component_cdas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_block11h3_single_component_cdas_v1"
UNIT = ("attn:11:head:03",)
EXPECTED = {
    "prior": "f8f5aa0cfdb85959bd5295f52ae85af4372253f781544bb07c1958d8131af196",
    "h3": "ee95aef443d63ce936f011ce2d551b8a0b220aa701507ecad30a72383475405a",
    "redteam": "3eb2c07f2dcc37f91a127cbc959805dbde61e1515b472b5b076f5e8304e2becd",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "unit_lib": "530ce8a5c0dba6a9b7700f5fe1bd716bc7b6a052a5f02311cb0955d74b55eba7",
    "single_lib": "363569c1b1cf20e4f31a4569d2467b4c86a6563405a49766af968037e12028b8",
}
RANK, STEPS, LR, SEEDS, COMPLEMENT_WEIGHT = 1, 150, 0.03, (1, 2), 1.0
MODEL_FORWARDS, EXAMPLE_EVALUATIONS = 948, 15568
MODEL_BACKWARD_FORWARDS, MODEL_UPDATES = 900, 450


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "h3": H3_AUTHORITY, "redteam": REDTEAM, "builder": BUILDER,
             "unit_lib": UNIT_LIB, "single_lib": SINGLE_LIB}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    prior = json.loads(PRIOR.read_text())
    h3 = json.loads(H3_AUTHORITY.read_text())
    redteam = json.loads(REDTEAM.read_text())
    rows = candidate.build_rows()
    counts = {family: sum(row["transform_id"] == family for row in rows)
              for family in ("A1", "A2", "P", "C")}
    if (prior.get("candidate_id") != CANDIDATE_ID or h3.get("terminal") != "screen"
            or not all(h3.get("predictions", {}).values())
            or redteam.get("terminal") != "v4_partition_invalid"
            or counts != {"A1": 32, "A2": 32, "P": 32, "C": 32}
            or single.validate_units(UNIT) != (11, "head")):
        raise ExperimentError("population, external H3 authority, or red-team terminal changed")
    return rows


def dryrun_receipt():
    return {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "unit": list(UNIT), "rank": RANK,
        "steps_per_start": STEPS, "starts": ["dim", *[f"random_seed_{s}" for s in SEEDS]],
        "model_forwards": MODEL_FORWARDS, "example_evaluations": EXAMPLE_EVALUATIONS,
        "transformer_backward_forwards": MODEL_BACKWARD_FORWARDS,
        "model_updates": MODEL_UPDATES, "fit_parameters_max": 16,
    }


def axes(backend, prep, q=None, complement=False):
    return g.patched_axis(backend, prep, UNIT, q=q, complement=complement)


def direction_fraction(prep, values):
    recoveries = [
        kernel.signed_pairwise_donor_recovery(base, donor, value)
        for base, donor, value in zip(prep.base_axis, prep.donor_axis, values)
    ]
    return sum(value > 0 for value in recoveries) / len(recoveries)


def report_direction(backend, prep, q):
    exact_axis = axes(backend, prep)
    sub_axis = axes(backend, prep, q=q)
    comp_axis = axes(backend, prep, q=q, complement=True)
    exact = g.recovery(prep, exact_axis)
    sub = g.recovery(prep, sub_axis)
    comp = g.recovery(prep, comp_axis)
    return {
        "exact_recovery": exact,
        "subspace_recovery": sub,
        "subspace_fraction": sub / exact if abs(exact) > 1e-8 else None,
        "subspace_direction_fraction": direction_fraction(prep, sub_axis),
        "complement_recovery": comp,
        "complement_fraction": comp / exact if abs(exact) > 1e-8 else None,
        "complement_direction_fraction": direction_fraction(prep, comp_axis),
    }


def main():
    rows = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    instrument = g.verify_against_producer(
        backend, g.rows_of(polarity, "A1"), layer=11, heads=(3,), mlp_layer=4)
    family_rows = {family: [row for row in rows if row["transform_id"] == family]
                   for family in ("A1", "A2", "P", "C")}
    preps = {
        "fit": g.prepare(backend, family_rows["A1"][0::2]),
        "heldout": g.prepare(backend, family_rows["A1"][1::2]),
        "a2": g.prepare(backend, family_rows["A2"]),
        "p": g.prepare(backend, family_rows["P"]),
        "c": g.prepare(backend, family_rows["C"]),
    }
    closures = {key: single.identity_closure(backend, preps[key], UNIT)
                for key in ("fit", "heldout", "a2")}
    fitted = single.fit(
        backend, preps["fit"], UNIT, rank=RANK, steps=STEPS, lr=LR,
        random_seeds=SEEDS, complement_weight=COMPLEMENT_WEIGHT)
    q_dim = g.diff_in_means_direction(backend, preps["fit"], UNIT)
    reports = {
        "dim": {key: report_direction(backend, preps[key], q_dim)
                for key in ("heldout", "a2")},
        "cdas": {key: report_direction(backend, preps[key], fitted.q)
                 for key in ("heldout", "a2")},
    }
    scale = g.target_scale(preps["fit"])
    controls = {}
    for name, q in (("dim", q_dim), ("cdas", fitted.q)):
        controls[name] = {
            "p_effect": g.same_answer_effect(preps["p"], axes(backend, preps["p"], q=q), scale),
            "c_effect": g.same_answer_effect(preps["c"], axes(backend, preps["c"], q=q), scale),
        }
    cosine_to_dim = float((fitted.q[:, 0] @ q_dim[:, 0]).abs())
    pred_a = bool(instrument["passed"] and all(
        closure["max_abs_logit_error"] <= 1e-4 for closure in closures.values()))
    pred_b = bool(
        reports["cdas"]["heldout"]["exact_recovery"] >= 0.03
        and reports["cdas"]["a2"]["exact_recovery"] >= 0.02
        and all(reports["cdas"][key]["subspace_direction_fraction"] >= 0.75
                for key in ("heldout", "a2")))
    pred_c = fitted.best_objective["joint"] <= fitted.dim_objective["joint"] + 1e-8
    pred_d = all(
        0.50 <= reports["cdas"][key]["subspace_fraction"] <= 1.20
        and reports["cdas"][key]["subspace_direction_fraction"] >= 0.75
        and abs(reports["cdas"][key]["complement_fraction"]) <= 0.30
        for key in ("heldout", "a2"))
    pred_e = controls["cdas"]["p_effect"] <= 0.20 and controls["cdas"]["c_effect"] <= 0.35
    pred_f = fitted.restart_min_pairwise_cosine >= 0.80
    predictions = {
        "pred_a_authority_and_identity_closure": bool(pred_a),
        "pred_b_exact_h3_is_material": bool(pred_b),
        "pred_c_optimizer_dominates_dim_on_fit_objective": bool(pred_c),
        "pred_d_cdas_axis_and_complement_transfer": bool(pred_d),
        "pred_e_selective_controls": bool(pred_e),
        "pred_f_restart_stability": bool(pred_f),
    }
    if not pred_a:
        terminal = "invalid"
    elif all(predictions.values()):
        terminal = "screen"
    elif pred_a and pred_b and pred_c and (not pred_d or not pred_e):
        terminal = "objective_mismatch"
    elif all((pred_a, pred_b, pred_c, pred_d, pred_e)) and not pred_f:
        terminal = "nonidentifiable"
    else:
        terminal = "null"
    fit_report = asdict(fitted)
    fit_report.pop("q")
    result = {
        "schema": "temporal_auxiliary_fresh_block11h3_single_component_cdas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": instrument, "identity_closure": closures,
        "fit": fit_report, "selected_cosine_to_dim": cosine_to_dim,
        "reports": reports, "controls": controls, "predictions": predictions,
        "terminal": terminal,
        "price": {
            "model_forwards": MODEL_FORWARDS,
            "example_evaluations": EXAMPLE_EVALUATIONS,
            "transformer_backward_forwards": MODEL_BACKWARD_FORWARDS,
            "model_updates": MODEL_UPDATES,
            "fit_parameters": fitted.span_rank,
        },
    }
    if not all(math.isfinite(value) for value in (
        fitted.best_objective["joint"], fitted.dim_objective["joint"], cosine_to_dim)):
        raise ExperimentError("nonfinite fit summary")
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal,
                      "predictions": predictions, "identity_closure": closures,
                      "fit": fit_report, "reports": reports, "controls": controls,
                      "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
