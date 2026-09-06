#!/usr/bin/env python3
"""Capability-repaired confirmation of the multi-cue H3 subspace test."""

# BQGATE: EXPERIMENT pred_a_authority_capability_closure_and_price pred_b_task_augmentation_beats_single_task_axis pred_c_optimization_beats_pooled_dim pred_d_rank2_union_identifies_shared_subspace pred_e_rank2_improvement_is_behavioral
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import circuit_candidate_temporal_auxiliary_fresh_cues_v4 as candidate
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_unit_greedy as g
import run_temporal_auxiliary_will_had_block11h3_multicue_subspace_v1 as parent
import run_temporal_auxiliary_will_had_block11h3_regularized_cdas_v1 as evaluator
import run_temporal_auxiliary_will_had_block11h3_aligned_objective_cdas_v1 as aligned
import single_component_das_eval as single

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2.json"
PARENT = ROOT / "ops/run_temporal_auxiliary_will_had_block11h3_multicue_subspace_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v4.py"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v4_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.block11h3_multicue_subspace_v2"
EXPECTED = {
    "prior": "222047bf8d8779de0921b02f9df56af196f5f13b769adecb8bc4b5021d093548",
    "parent": "fb339ee67bbab60531f5f3d5e27823bf5fa28fa23c26e5510f01d5736de2cd50",
    "builder": "31e40a5e8a8b285ce7afdb6327276c0aa28b4759083586d0310b0857c8b86764",
    "capability": "63b69e3bc57a0a8a9afcffa252737614f9a8a41b6732b9fff655d9da128ef8b2",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    if {"prior": sha(PRIOR), "parent": sha(PARENT), "builder": sha(BUILDER),
            "capability": sha(CAPABILITY)} != EXPECTED:
        raise RuntimeError("authority hash changed")
    prior, capability = [json.loads(path.read_text()) for path in (PRIOR, CAPABILITY)]
    _parent_rows, scalar_result = parent.validate_static()
    if (prior.get("candidate_id") != CANDIDATE_ID or capability.get("terminal") != "manifest"
            or not all(capability.get("predictions", {}).values())):
        raise RuntimeError("candidate or capability authority changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "rank1_updates": parent.STEPS,
              "ranks": [1, 2], "model_forwards_max": parent.MAX_FORWARDS,
              "example_evaluations_max": parent.MAX_EVALUATIONS,
              "transformer_backward_forwards_max": parent.MAX_BACKWARD_FORWARDS,
              "model_updates_max": parent.MAX_UPDATES}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    fit_rows = {"v1": parent.target_rows(parent.cue_v1.build_rows()),
                "v2": parent.target_rows(parent.cue_v2.build_rows())}
    task_preps = {name: g.prepare(backend, rows) for name, rows in fit_rows.items()}
    pooled_prep = g.prepare(backend, fit_rows["v1"] + fit_rows["v2"])
    q_task = {name: g.diff_in_means_direction(backend, prep, evaluator.UNIT)
              for name, prep in task_preps.items()}
    q_pooled = g.diff_in_means_direction(backend, pooled_prep, evaluator.UNIT)
    q_union = backend.torch.linalg.qr(
        backend.torch.cat([q_task["v1"], q_task["v2"]], dim=1), mode="reduced").Q
    targets = aligned.aligned_targets(backend, pooled_prep)
    delta = single.cached_delta_matrix(backend, pooled_prep, evaluator.UNIT)
    span, singular, span_rank = single.empirical_span(delta)
    q_aligned, fit_trace = parent.fit_rank1(backend, pooled_prep, targets, span, q_pooled)
    q_single = backend.torch.tensor(scalar_result["axis_artifact"]["coordinates"],
                                    device=backend.device).float().unsqueeze(1)
    axes = {"single_task_unregularized": q_single, "pooled_dim": q_pooled,
            "pooled_aligned_rank1": q_aligned, "two_task_dim_union_rank2": q_union}
    test_rows = {family: [row for row in candidate.build_rows()
                          if row["transform_id"] == family] for family in ("A1", "A2")}
    preps = {family: g.prepare(backend, rows) for family, rows in test_rows.items()}
    evaluated = {family: parent.evaluate(backend, prep, axes) for family, prep in preps.items()}
    reports = {family: value[0] for family, value in evaluated.items()}
    closure = {family: value[1] for family, value in evaluated.items()}
    direction = {family: value[2] for family, value in evaluated.items()}
    full = lambda panel, method: reports[panel][method]["full_vocabulary"]["joint_squared_objective"]
    scalar = lambda panel, method: reports[panel][method]["scalar"]["joint_squared_objective"]
    rank1 = ("single_task_unregularized", "pooled_dim", "pooled_aligned_rank1")
    pred_a = bool(max(value for cell in closure.values() for value in cell.values()) <= 1e-4
                  and parent.all_finite(reports)
                  and fit_trace["final"]["joint"] < fit_trace["initial"]["joint"])
    pred_b = all(full(p, "pooled_aligned_rank1") < full(p, "single_task_unregularized")
                 for p in ("A1", "A2"))
    pred_c = all(full(p, "pooled_aligned_rank1") < full(p, "pooled_dim")
                 for p in ("A1", "A2"))
    pred_d = all(full(p, "two_task_dim_union_rank2") < min(full(p, m) for m in rank1)
                 and abs(reports[p]["two_task_dim_union_rank2"]["scalar"]
                         ["signed_complement_fraction"]) <= 0.30 for p in ("A1", "A2"))
    pred_e = all(scalar(p, "two_task_dim_union_rank2") < min(scalar(p, m) for m in rank1)
                 and direction[p] >= 0.90 for p in ("A1", "A2"))
    predictions = {"pred_a_authority_capability_closure_and_price": pred_a,
        "pred_b_task_augmentation_beats_single_task_axis": pred_b,
        "pred_c_optimization_beats_pooled_dim": pred_c,
        "pred_d_rank2_union_identifies_shared_subspace": pred_d,
        "pred_e_rank2_improvement_is_behavioral": pred_e}
    rank1_no_worse = all(full(p, "pooled_aligned_rank1")
                         <= full(p, "two_task_dim_union_rank2") for p in ("A1", "A2"))
    terminal = ("invalid" if not pred_a else "shared_rank2" if pred_d and pred_e
                else "shared_rank1" if pred_b and pred_c and rank1_no_worse
                else "task_conditioned")
    result = {"schema": "temporal_auxiliary_block11h3_multicue_subspace_result_v2",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "capability_authority": {"counts": capability["counts"],
            "joint_correct": capability["joint_correct"]},
        "empirical_span": {"rank": span_rank,
            "singular_values": [float(value) for value in singular.detach().cpu()]},
        "task_dim_cosine": float((q_task["v1"][:, 0] @ q_task["v2"][:, 0]).abs()),
        "axis_artifacts": {"pooled_aligned_rank1": q_aligned[:, 0].detach().cpu().tolist(),
                           "two_task_dim_union_rank2": q_union.detach().cpu().tolist()},
        "fit_trace": fit_trace, "identity_closure": closure,
        "union_direction_fraction": direction, "reports": reports,
        "predictions": predictions, "rank1_no_worse_than_rank2": rank1_no_worse,
        "terminal": terminal, "price": {"model_forwards": 239,
            "example_evaluations": 27840, "transformer_backward_forwards": 200,
            "model_updates": 100}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "task_dim_cosine",
          "capability_authority", "fit_trace", "union_direction_fraction", "reports",
          "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
