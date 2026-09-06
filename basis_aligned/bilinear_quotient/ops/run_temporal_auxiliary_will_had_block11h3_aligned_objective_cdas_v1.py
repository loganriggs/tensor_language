#!/usr/bin/env python3
"""Optimize cDAS against the full causal-effect metric it is meant to satisfy."""

# BQGATE: EXPERIMENT pred_a_authority_closure_reproduction_and_price pred_b_interior_regularization_selected pred_c_aligned_objective_beats_dim_on_sealed_a2 pred_d_selected_axis_preserves_scalar_usefulness pred_e_optimization_improves_its_fit_target
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_unit_greedy as g
import run_temporal_auxiliary_will_had_block11h3_regularized_cdas_v1 as parent
import single_component_das_eval as single


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_block11h3_aligned_objective_cdas_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_regularized_cdas_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_block11h3_regularized_cdas_v1.py"
AXIS_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_block11h3_cdas_axis_necessity_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
UNIT_LIB = ROOT / "ops/circuit_unit_greedy.py"
SINGLE_LIB = ROOT / "ops/single_component_das_eval.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_aligned_objective_cdas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.block11h3_aligned_objective_cdas_v1"
EXPECTED = {
    "prior": "5f26205b8de1954cca778bd48fdce1496f369f4e8a4eda34b25fe367ffeda034",
    "parent_result": "f7d53dd6530dbdbebba7610236adc862b3c595bd83fb6c1b24d8fd4365543163",
    "parent_runner": "966fc3b4bafba272ca5702a934635f6ae033abc8c1575cefd1390fda2b1cdc11",
    "axis_result": "4f345907c41222ebeec33c3a860052a0a8f39166655da0354f69e79a1c577fc5",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "unit_lib": "2a8c01fcf0c3830ec4581c6b78013323ae297b95d41e680683b99d83398a0782",
    "single_lib": "363569c1b1cf20e4f31a4569d2467b4c86a6563405a49766af968037e12028b8",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
STEPS, LR = 100, 0.03
FORWARDS, EVALUATIONS, BACKWARDS, UPDATES = 1373, 22176, 1200, 600


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "parent_result": PARENT_RESULT, "parent_runner": PARENT_RUNNER,
             "axis_result": AXIS_RESULT, "builder": BUILDER, "unit_lib": UNIT_LIB,
             "single_lib": SINGLE_LIB, "producer": PRODUCER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority hash changed")
    authority = json.loads(PARENT_RESULT.read_text())
    if authority.get("terminal") != "null" or authority.get("candidate_id") != parent.CANDIDATE_ID:
        raise ExperimentError("parent result changed")
    rows = candidate.build_rows()
    axis_result = json.loads(AXIS_RESULT.read_text())
    old_coordinates = axis_result["axis_artifact"]["coordinates"]
    if len(rows) != 128 or len(old_coordinates) != 128:
        raise ExperimentError("population or baseline axis changed")
    return rows, authority, old_coordinates


def centered(tensor):
    return tensor - tensor.mean(1, keepdim=True)


def aligned_targets(backend, prep):
    targets = parent.fit_targets(backend, prep)
    base, exact = targets["base"][1], targets["exact"][1]
    exact_delta = centered(exact - base)
    scale = exact_delta.square().mean(1).sqrt()
    if bool((scale <= 1e-8).any()):
        raise ExperimentError("exact full-vocabulary effect is zero")
    targets.update({"base_centered": centered(base).detach(),
                    "exact_centered": centered(exact).detach(),
                    "vector_scale": scale.detach()})
    return targets


def objective(backend, prep, targets, q, weight):
    sub = parent.full_forward(backend, prep, q=q, grad=True)
    comp = parent.full_forward(backend, prep, q=q, complement=True, grad=True)
    match = (((parent.scalar_axis(sub[0]) - targets["exact_axis"])
              / targets["denominator"]) ** 2).mean()
    inert = (((parent.scalar_axis(comp[0]) - targets["base_axis"])
              / targets["denominator"]) ** 2).mean()
    vector_match = (((centered(sub[1]) - targets["exact_centered"])
                     / targets["vector_scale"][:, None]) ** 2).mean()
    vector_inert = (((centered(comp[1]) - targets["base_centered"])
                     / targets["vector_scale"][:, None]) ** 2).mean()
    scalar, vector = match + inert, vector_match + vector_inert
    return scalar + weight * vector, {"scalar": scalar, "vector": vector,
                                      "scalar_match": match, "scalar_inert": inert,
                                      "vector_match": vector_match, "vector_inert": vector_inert}


def fit_weight(backend, prep, targets, span, q_dim, weight):
    torch = backend.torch
    raw = (span @ q_dim).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=LR)
    trace, best = [], None

    def checkpoint(step):
        nonlocal best
        with torch.no_grad():
            coordinate = raw / raw.norm().clamp_min(1e-30)
            q = span.T @ coordinate
            loss, pieces = objective(backend, prep, targets, q, weight)
        report = {"step": step, "joint": float(loss),
                  **{key: float(value) for key, value in pieces.items()}}
        trace.append(report)
        if best is None or report["joint"] < best[0]["joint"]:
            best = (report, q.detach().clone())

    checkpoint(0)
    for update in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        coordinate = raw / raw.norm().clamp_min(1e-30)
        loss = objective(backend, prep, targets, span.T @ coordinate, weight)[0]
        loss.backward()
        optimizer.step()
        if (update + 1) % 10 == 0:
            checkpoint(update + 1)
    return {"weight": weight, "q": best[1], "best": best[0], "trace": trace,
            "coordinates": best[1][:, 0].detach().cpu().tolist()}


def main():
    rows, authority, old_coordinates = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "lambda_grid": LAMBDAS,
              "starts_per_lambda": 1, "updates_per_lambda": STEPS,
              "model_forwards": FORWARDS, "example_evaluations": EVALUATIONS,
              "transformer_backward_forwards": BACKWARDS, "model_updates": UPDATES,
              "fit_parameters_max": 16}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    family = {name: [row for row in rows if row["transform_id"] == name]
              for name in ("A1", "A2")}
    preps = {"fit": g.prepare(backend, family["A1"][0::2]),
             "validation": g.prepare(backend, family["A1"][1::2]),
             "a2": g.prepare(backend, family["A2"])}
    targets = aligned_targets(backend, preps["fit"])
    delta = single.cached_delta_matrix(backend, preps["fit"], parent.UNIT)
    span, singular, span_rank = single.empirical_span(delta)
    q_dim = g.diff_in_means_direction(backend, preps["fit"], parent.UNIT)
    fits = [fit_weight(backend, preps["fit"], targets, span, q_dim, weight)
            for weight in LAMBDAS]
    q_old = backend.torch.tensor(old_coordinates, device=backend.device).float().unsqueeze(1)
    q_kl = backend.torch.tensor(authority["fits"]["kl"]["coordinates"],
                               device=backend.device).float().unsqueeze(1)
    validation_axes = {"dim": q_dim, "unregularized": q_old, "kl": q_kl,
                       **{f"aligned_{fit['weight']:g}": fit["q"] for fit in fits}}
    validation, validation_closure = parent.evaluate_axes(
        backend, preps["validation"], validation_axes)
    selected = min(fits, key=lambda fit: validation[f"aligned_{fit['weight']:g}"]
                   ["full_vocabulary"]["joint_squared_objective"])
    selected_name = f"aligned_{selected['weight']:g}"
    a2_axes = {"dim": q_dim, "unregularized": q_old, "kl": q_kl,
               selected_name: selected["q"]}
    a2, a2_closure = parent.evaluate_axes(backend, preps["a2"], a2_axes)
    reproduction = max(
        abs(validation[name][scope]["joint_squared_objective"]
            - authority["reports"]["heldout"][old_name][scope]["joint_squared_objective"])
        for name, old_name in (("dim", "dim"), ("unregularized", "unregularized"), ("kl", "kl"))
        for scope in ("scalar", "full_vocabulary"))
    finite = [value for fit in fits for point in fit["trace"] for value in point.values()
              if isinstance(value, (int, float))]
    price = {"model_forwards": FORWARDS, "example_evaluations": EVALUATIONS,
             "transformer_backward_forwards": BACKWARDS, "model_updates": UPDATES,
             "fit_parameters": span_rank}
    closure_values = [*targets["closure"].values(), *a2_closure.values(),
                      *validation_closure.values()]
    pred_a = bool(max(closure_values) <= 1e-4
                  and reproduction <= 0.01 and all(math.isfinite(value) for value in finite)
                  and span_rank == 16)
    pred_b = selected["weight"] not in (LAMBDAS[0], LAMBDAS[-1])
    pred_c = (a2[selected_name]["full_vocabulary"]["joint_squared_objective"]
              < a2["dim"]["full_vocabulary"]["joint_squared_objective"])
    pred_d = (a2[selected_name]["scalar"]["joint_squared_objective"]
              <= 2 * a2["dim"]["scalar"]["joint_squared_objective"]
              and a2[selected_name]["scalar"]["joint_squared_objective"]
              < a2["unregularized"]["scalar"]["joint_squared_objective"])
    pred_e = all(fit["best"]["joint"] < fit["trace"][0]["joint"] for fit in fits)
    predictions = {
        "pred_a_authority_closure_reproduction_and_price": pred_a,
        "pred_b_interior_regularization_selected": pred_b,
        "pred_c_aligned_objective_beats_dim_on_sealed_a2": pred_c,
        "pred_d_selected_axis_preserves_scalar_usefulness": pred_d,
        "pred_e_optimization_improves_its_fit_target": pred_e,
    }
    terminal = ("invalid" if not pred_a else "screen" if all(predictions.values())
                else "target_repaired" if all((pred_a, pred_c, pred_d, pred_e))
                else "null")
    result = {"schema": "temporal_auxiliary_block11h3_aligned_objective_cdas_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started,
              "authority_sha256": EXPECTED,
              "empirical_span": {"rank": span_rank,
                  "singular_values": [float(value) for value in singular.detach().cpu()]},
              "closure": {"fit": targets["closure"], "validation": validation_closure,
                          "a2": a2_closure},
              "baseline_reproduction_max_abs_error": reproduction,
              "fits": [{key: value for key, value in fit.items() if key != "q"} for fit in fits],
              "selection": {"panel": "heldout_A1_validation", "weight": selected["weight"],
                            "axis_name": selected_name},
              "reports": {"validation": validation, "a2_sealed": a2},
              "predictions": predictions, "price": price, "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "selection", "reports",
          "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
