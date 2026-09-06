#!/usr/bin/env python3
"""Fit a multi-cue H3 axis and test rank-one versus task-union rank two."""

# BQGATE: EXPERIMENT pred_a_authority_capability_closure_and_price pred_b_task_augmentation_beats_single_task_axis pred_c_optimization_beats_pooled_dim pred_d_rank2_union_identifies_shared_subspace pred_e_rank2_improvement_is_behavioral
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as cue_v1
import circuit_candidate_temporal_auxiliary_fresh_cues_v2 as cue_v2
import circuit_candidate_temporal_auxiliary_fresh_cues_v3 as candidate
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_unit_greedy as g
import run_temporal_auxiliary_will_had_block11h3_aligned_objective_cdas_v1 as aligned
import run_temporal_auxiliary_will_had_block11h3_regularized_cdas_v1 as evaluator
import single_component_das_eval as single


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_block11h3_multicue_subspace_v1.json"
V1 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
V2 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v2.py"
V3 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v3.py"
SCALAR_AXIS = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_block11h3_cdas_axis_necessity_v2_result.json"
REDTEAM = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_regularization_fresh_transfer_v1_result.json"
ALIGNED_LIB = ROOT / "ops/run_temporal_auxiliary_will_had_block11h3_aligned_objective_cdas_v1.py"
EVALUATOR_LIB = ROOT / "ops/run_temporal_auxiliary_will_had_block11h3_regularized_cdas_v1.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
UNIT_LIB = ROOT / "ops/circuit_unit_greedy.py"
SINGLE_LIB = ROOT / "ops/single_component_das_eval.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.block11h3_multicue_subspace_v1"
EXPECTED = {
    "prior": "ed628427a427f484a6f58340d85a1dabe396844ab85f6ae3df067e67a63fafd7",
    "v1": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "v2": "adbfaf91ed2889cc42da85255edf9f5074f1002e9ad93dc1d4ff706de66d1144",
    "v3": "d434eb9d86aba45eae12b93974113c85f33251d204fb78b632e3116cdce21d6b",
    "scalar_axis": "4f345907c41222ebeec33c3a860052a0a8f39166655da0354f69e79a1c577fc5",
    "redteam": "0802864bae174aa73ffb5bb92d2f2df20ee3ca8baa128437f09504ba11146b92",
    "aligned_lib": "40e1934e06db5c39eb31e11f92df519846948ccceb1d2b1087855557d7f999e9",
    "evaluator_lib": "966fc3b4bafba272ca5702a934635f6ae033abc8c1575cefd1390fda2b1cdc11",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
    "unit_lib": "302094521f5f5abe26a00301460a80cccd74059f0bb864996dac36a0d35ac2ab",
    "single_lib": "363569c1b1cf20e4f31a4569d2467b4c86a6563405a49766af968037e12028b8",
}
STEPS, LR, VECTOR_WEIGHT = 100, 0.03, 0.3
MAX_FORWARDS, MAX_EVALUATIONS, MAX_BACKWARD_FORWARDS, MAX_UPDATES = 254, 29760, 200, 100


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "v1": V1, "v2": V2, "v3": V3,
             "scalar_axis": SCALAR_AXIS, "redteam": REDTEAM,
             "aligned_lib": ALIGNED_LIB, "evaluator_lib": EVALUATOR_LIB,
             "producer": PRODUCER, "unit_lib": UNIT_LIB, "single_lib": SINGLE_LIB}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    prior, scalar_axis, redteam = [json.loads(path.read_text())
                                   for path in (PRIOR, SCALAR_AXIS, REDTEAM)]
    rows = {"v1": cue_v1.build_rows(), "v2": cue_v2.build_rows(),
            "v3": candidate.build_rows()}
    if (prior.get("candidate_id") != CANDIDATE_ID
            or scalar_axis.get("terminal") != "screen"
            or redteam.get("terminal") != "overfit_not_repaired"
            or any(len(value) != 128 for value in rows.values())):
        raise ExperimentError("candidate, parent terminal, or population changed")
    return rows, scalar_axis


def target_rows(rows):
    return [row for row in rows if row["transform_id"] in {"A1", "A2"}]


def fit_rank1(backend, prep, targets, span, q_dim):
    torch = backend.torch
    raw = (span @ q_dim).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=LR)

    def current():
        coordinate = raw / raw.norm().clamp_min(1e-30)
        return span.T @ coordinate

    with torch.no_grad():
        initial_loss, initial_pieces = aligned.objective(
            backend, prep, targets, current(), VECTOR_WEIGHT)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = aligned.objective(backend, prep, targets, current(), VECTOR_WEIGHT)[0]
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        q = current().detach()
        final_loss, final_pieces = aligned.objective(backend, prep, targets, q, VECTOR_WEIGHT)
    return q, {
        "initial": {"joint": float(initial_loss),
                    **{key: float(value) for key, value in initial_pieces.items()}},
        "final": {"joint": float(final_loss),
                  **{key: float(value) for key, value in final_pieces.items()}},
    }


def evaluate(backend, prep, axes):
    torch = backend.torch
    identity = torch.eye(128, device=backend.device)
    outputs = {"base": evaluator.full_forward(backend, prep),
               "exact": evaluator.full_forward(backend, prep, exact=True),
               "identity": evaluator.full_forward(backend, prep, q=identity)}
    for name, q in axes.items():
        outputs[name] = evaluator.full_forward(backend, prep, q=q)
        outputs[name + "_complement"] = evaluator.full_forward(
            backend, prep, q=q, complement=True)
    closure = {
        "answer_foil_max_abs": float((outputs["identity"][0] - outputs["exact"][0]).abs().max()),
        "full_vocabulary_max_abs": float((outputs["identity"][1] - outputs["exact"][1]).abs().max()),
    }
    reports = {name: evaluator.redteam.axis_report(prep, outputs, name) for name in axes}
    union_axis = evaluator.scalar_axis(outputs["two_task_dim_union_rank2"][0])
    direction_fraction = sum(
        (float(value) - base) / (donor - base) > 0
        for base, donor, value in zip(prep.base_axis, prep.donor_axis, union_axis)
    ) / len(prep.rows)
    return reports, closure, direction_fraction


def capability(prep):
    return {"n": len(prep.rows),
            "base_correct": sum(value < 0 for value in prep.base_axis),
            "donor_correct": sum(value > 0 for value in prep.donor_axis)}


def all_finite(value):
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def main():
    rows, scalar_result = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False,
              "fit_task_families": ["v1", "v2"], "sealed_test": "v3",
              "rank1_updates": STEPS, "ranks": [1, 2],
              "model_forwards_max": MAX_FORWARDS,
              "example_evaluations_max": MAX_EVALUATIONS,
              "transformer_backward_forwards_max": MAX_BACKWARD_FORWARDS,
              "model_updates_max": MAX_UPDATES}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    fit_rows = {name: target_rows(rows[name]) for name in ("v1", "v2")}
    task_preps = {name: g.prepare(backend, value) for name, value in fit_rows.items()}
    pooled_rows = fit_rows["v1"] + fit_rows["v2"]
    pooled_prep = g.prepare(backend, pooled_rows)
    q_task = {name: g.diff_in_means_direction(backend, prep, evaluator.UNIT)
              for name, prep in task_preps.items()}
    q_pooled = g.diff_in_means_direction(backend, pooled_prep, evaluator.UNIT)
    raw_union = backend.torch.cat([q_task["v1"], q_task["v2"]], dim=1)
    q_union = backend.torch.linalg.qr(raw_union, mode="reduced").Q
    if q_union.shape != (128, 2):
        raise ExperimentError("two task DIM axes are rank deficient")
    targets = aligned.aligned_targets(backend, pooled_prep)
    delta = single.cached_delta_matrix(backend, pooled_prep, evaluator.UNIT)
    span, singular, span_rank = single.empirical_span(delta)
    q_aligned, fit_trace = fit_rank1(backend, pooled_prep, targets, span, q_pooled)
    q_single = backend.torch.tensor(
        scalar_result["axis_artifact"]["coordinates"], device=backend.device).float().unsqueeze(1)
    axes = {"single_task_unregularized": q_single, "pooled_dim": q_pooled,
            "pooled_aligned_rank1": q_aligned, "two_task_dim_union_rank2": q_union}
    test_rows = {family: [row for row in rows["v3"] if row["transform_id"] == family]
                 for family in ("A1", "A2")}
    test_preps = {family: g.prepare(backend, value) for family, value in test_rows.items()}
    evaluated = {family: evaluate(backend, prep, axes)
                 for family, prep in test_preps.items()}
    reports = {family: value[0] for family, value in evaluated.items()}
    closure = {family: value[1] for family, value in evaluated.items()}
    direction_fraction = {family: value[2] for family, value in evaluated.items()}
    capabilities = {family: capability(prep) for family, prep in test_preps.items()}
    full = lambda panel, method: reports[panel][method]["full_vocabulary"]["joint_squared_objective"]
    scalar = lambda panel, method: reports[panel][method]["scalar"]["joint_squared_objective"]
    rank1 = ("single_task_unregularized", "pooled_dim", "pooled_aligned_rank1")
    pred_a = bool(all(cell["base_correct"] >= 24 and cell["donor_correct"] >= 24
                      for cell in capabilities.values())
                  and max(value for cell in closure.values() for value in cell.values()) <= 1e-4
                  and all_finite(reports) and fit_trace["final"]["joint"] < fit_trace["initial"]["joint"])
    pred_b = all(full(panel, "pooled_aligned_rank1") < full(panel, "single_task_unregularized")
                 for panel in ("A1", "A2"))
    pred_c = all(full(panel, "pooled_aligned_rank1") < full(panel, "pooled_dim")
                 for panel in ("A1", "A2"))
    pred_d = all(full(panel, "two_task_dim_union_rank2") < min(full(panel, name) for name in rank1)
                 and abs(reports[panel]["two_task_dim_union_rank2"]["scalar"]
                         ["signed_complement_fraction"]) <= 0.30 for panel in ("A1", "A2"))
    pred_e = all(scalar(panel, "two_task_dim_union_rank2") < min(scalar(panel, name) for name in rank1)
                 and direction_fraction[panel] >= 0.90 for panel in ("A1", "A2"))
    predictions = {
        "pred_a_authority_capability_closure_and_price": pred_a,
        "pred_b_task_augmentation_beats_single_task_axis": pred_b,
        "pred_c_optimization_beats_pooled_dim": pred_c,
        "pred_d_rank2_union_identifies_shared_subspace": pred_d,
        "pred_e_rank2_improvement_is_behavioral": pred_e,
    }
    rank1_no_worse = all(full(panel, "pooled_aligned_rank1")
                         <= full(panel, "two_task_dim_union_rank2") for panel in ("A1", "A2"))
    terminal = ("invalid" if not pred_a else "shared_rank2" if pred_d and pred_e
                else "shared_rank1" if pred_b and pred_c and rank1_no_worse
                else "task_conditioned")
    price = {"model_forwards": 239, "example_evaluations": 27840,
             "transformer_backward_forwards": 200, "model_updates": 100}
    if (price["model_forwards"] > MAX_FORWARDS
            or price["example_evaluations"] > MAX_EVALUATIONS):
        raise ExperimentError("price exceeded")
    result = {
        "schema": "temporal_auxiliary_block11h3_multicue_subspace_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "empirical_span": {"rank": span_rank,
                           "singular_values": [float(value) for value in singular.detach().cpu()]},
        "task_dim_cosine": float((q_task["v1"][:, 0] @ q_task["v2"][:, 0]).abs()),
        "pooled_aligned_cosines": {name: float((q_aligned[:, 0] @ q[:, 0]).abs())
                                    for name, q in q_task.items()},
        "fit_trace": fit_trace, "capability": capabilities,
        "identity_closure": closure, "union_direction_fraction": direction_fraction,
        "reports": reports, "predictions": predictions,
        "rank1_no_worse_than_rank2": rank1_no_worse,
        "terminal": terminal, "price": price,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "task_dim_cosine",
          "pooled_aligned_cosines", "fit_trace", "capability", "union_direction_fraction",
          "reports", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
