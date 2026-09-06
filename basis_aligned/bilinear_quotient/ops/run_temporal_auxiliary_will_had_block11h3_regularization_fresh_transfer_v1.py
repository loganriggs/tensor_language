#!/usr/bin/env python3
"""No-refit OOD test of frozen regularized block11H3 cDAS axes."""

# BQGATE: EXPERIMENT pred_a_authority_capability_closure_and_price pred_b_regularization_reduces_unregularized_overfit pred_c_aligned_beats_dim_out_of_task pred_d_aligned_preserves_behavioral_usefulness pred_e_regularization_ranking_is_construction_stable
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as discovery
import circuit_candidate_temporal_auxiliary_fresh_cues_v2 as candidate
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_unit_greedy as g
import run_temporal_auxiliary_will_had_block11h3_regularized_cdas_v1 as evaluator


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_block11h3_regularization_fresh_transfer_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v2.py"
DISCOVERY_BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
ALIGNED = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_aligned_objective_cdas_v1_result.json"
REGULARIZED = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_regularized_cdas_v1_result.json"
SCALAR_AXIS = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_block11h3_cdas_axis_necessity_v2_result.json"
EVALUATOR = ROOT / "ops/run_temporal_auxiliary_will_had_block11h3_regularized_cdas_v1.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
UNIT_LIB = ROOT / "ops/circuit_unit_greedy.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_regularization_fresh_transfer_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.block11h3_regularization_fresh_transfer_v1"
EXPECTED = {
    "prior": "a415fa5f6adb143528c436a80a1e84024ef9a2ec4b4aabea18dec58494821740",
    "builder": "adbfaf91ed2889cc42da85255edf9f5074f1002e9ad93dc1d4ff706de66d1144",
    "discovery_builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "aligned": "3aea84323bae1c2e46a430ef5f08b838826504693e6b1ba8a05027ca065b379d",
    "regularized": "f7d53dd6530dbdbebba7610236adc862b3c595bd83fb6c1b24d8fd4365543163",
    "scalar_axis": "4f345907c41222ebeec33c3a860052a0a8f39166655da0354f69e79a1c577fc5",
    "evaluator": "966fc3b4bafba272ca5702a934635f6ae033abc8c1575cefd1390fda2b1cdc11",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
    "unit_lib": "302094521f5f5abe26a00301460a80cccd74059f0bb864996dac36a0d35ac2ab",
}
MAX_FORWARDS, MAX_EVALUATIONS = 32, 992


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "builder": BUILDER, "discovery_builder": DISCOVERY_BUILDER,
             "aligned": ALIGNED, "regularized": REGULARIZED, "scalar_axis": SCALAR_AXIS,
             "evaluator": EVALUATOR, "producer": PRODUCER, "unit_lib": UNIT_LIB}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    prior = json.loads(PRIOR.read_text())
    aligned = json.loads(ALIGNED.read_text())
    regularized = json.loads(REGULARIZED.read_text())
    scalar_axis = json.loads(SCALAR_AXIS.read_text())
    rows = candidate.build_rows()
    discovery_rows = discovery.build_rows()
    if (prior.get("candidate_id") != CANDIDATE_ID
            or aligned.get("selection", {}).get("weight") != 0.3
            or regularized.get("terminal") != "null"
            or scalar_axis.get("terminal") != "screen"
            or len(rows) != 128 or len(discovery_rows) != 128):
        raise ExperimentError("frozen selection, result terminal, or population changed")
    return rows, discovery_rows, aligned, regularized, scalar_axis


def coordinates(torch, device, values):
    q = torch.tensor(values, device=device).float().unsqueeze(1)
    if q.shape != (128, 1) or abs(float(q.norm()) - 1.0) > 1e-4:
        raise ExperimentError("frozen axis is not a unit 128-vector")
    return q


def capability(prep):
    return {
        "n": len(prep.rows),
        "base_correct": sum(value < 0 for value in prep.base_axis),
        "donor_correct": sum(value > 0 for value in prep.donor_axis),
        "joint_correct": sum(base < 0 and donor > 0
                             for base, donor in zip(prep.base_axis, prep.donor_axis)),
    }


def all_finite(value):
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def main():
    rows, discovery_rows, aligned_result, regularized_result, scalar_result = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False,
              "axes": ["dim", "unregularized", "noise", "kl", "aligned"],
              "fit_updates": 0, "model_updates": 0,
              "model_forwards_max": MAX_FORWARDS,
              "example_evaluations_max": MAX_EVALUATIONS}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    discovery_a1 = [row for row in discovery_rows if row["transform_id"] == "A1"][0::2]
    discovery_prep = g.prepare(backend, discovery_a1)
    q_dim = g.diff_in_means_direction(backend, discovery_prep, evaluator.UNIT)
    aligned_fit = next(fit for fit in aligned_result["fits"] if fit["weight"] == 0.3)
    regularized_fits = {fit["name"]: fit for fit in regularized_result["fits"]}
    axes = {
        "dim": q_dim,
        "unregularized": coordinates(torch, backend.device,
                                      scalar_result["axis_artifact"]["coordinates"]),
        "noise": coordinates(torch, backend.device, regularized_fits["noise"]["coordinates"]),
        "kl": coordinates(torch, backend.device, regularized_fits["kl"]["coordinates"]),
        "aligned": coordinates(torch, backend.device, aligned_fit["coordinates"]),
    }
    family = {name: [row for row in rows if row["transform_id"] == name]
              for name in ("A1", "A2")}
    preps = {name: g.prepare(backend, family[name]) for name in ("A1", "A2")}
    evaluated = {name: evaluator.evaluate_axes(backend, prep, axes)
                 for name, prep in preps.items()}
    reports = {name: result[0] for name, result in evaluated.items()}
    closure = {name: result[1] for name, result in evaluated.items()}
    capabilities = {name: capability(prep) for name, prep in preps.items()}
    full = lambda panel, method: reports[panel][method]["full_vocabulary"]["joint_squared_objective"]
    scalar = lambda panel, method: reports[panel][method]["scalar"]["joint_squared_objective"]
    regularized_names = ("noise", "kl", "aligned")
    best_regularized = {panel: min(regularized_names, key=lambda name: full(panel, name))
                        for panel in ("A1", "A2")}
    pred_a = bool(
        all(cell["base_correct"] >= 24 and cell["donor_correct"] >= 24
            for cell in capabilities.values())
        and max(value for cell in closure.values() for value in cell.values()) <= 1e-4
        and all_finite(reports)
    )
    pred_b = any(all(full(panel, method) < full(panel, "unregularized")
                         for panel in ("A1", "A2")) for method in regularized_names)
    pred_c = all(full(panel, "aligned") < full(panel, "dim") for panel in ("A1", "A2"))
    pred_d = all(scalar(panel, "aligned") <= 1.25 * scalar(panel, "dim")
                 and scalar(panel, "aligned") < scalar(panel, "unregularized")
                 for panel in ("A1", "A2"))
    pred_e = best_regularized["A1"] == best_regularized["A2"]
    predictions = {
        "pred_a_authority_capability_closure_and_price": pred_a,
        "pred_b_regularization_reduces_unregularized_overfit": pred_b,
        "pred_c_aligned_beats_dim_out_of_task": pred_c,
        "pred_d_aligned_preserves_behavioral_usefulness": pred_d,
        "pred_e_regularization_ranking_is_construction_stable": pred_e,
    }
    if not pred_a:
        terminal = "invalid"
    elif all(predictions.values()):
        terminal = "screen"
    elif pred_b and pred_d and not pred_c:
        terminal = "regularization_helpful_dim_best"
    elif pred_b:
        terminal = "regularization_unstable"
    else:
        terminal = "overfit_not_repaired"
    result = {
        "schema": "temporal_auxiliary_block11h3_regularization_fresh_transfer_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "capability": capabilities, "identity_closure": closure,
        "best_regularized_method": best_regularized,
        "axis_cosines": {left: {right: float((axes[left][:, 0] @ axes[right][:, 0]).abs())
                                for right in axes} for left in axes},
        "reports": reports, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": MAX_FORWARDS,
                  "example_evaluations": (2 * len(discovery_a1)
                      + sum((2 + 3 + 2 * len(axes)) * len(family[name])
                            for name in ("A1", "A2"))),
                  "fit_updates": 0, "model_updates": 0},
    }
    if result["price"]["example_evaluations"] > MAX_EVALUATIONS:
        raise ExperimentError("price exceeded")
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability",
          "best_regularized_method", "reports", "predictions", "terminal", "price")},
          sort_keys=True))


if __name__ == "__main__":
    main()
