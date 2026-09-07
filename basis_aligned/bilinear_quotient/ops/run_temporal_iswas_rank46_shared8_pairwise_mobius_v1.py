#!/usr/bin/env python3
"""Exact degree-two causal Möbius pilot on the shared eight-MLP writer bank."""
# BQGATE: EXPERIMENT pred_a_authority_and_full_intersection_replay pred_b_singleton_nonadditivity_reproduces pred_c_nontrivial_pair_interaction_both_tasks pred_d_degree_two_reduces_error_both_tasks pred_e_pair_interactions_are_sparse
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json
import math
import os
import time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import pooled_response_projector as pooled
import run_temporal_iswas_rank46_task_mode_complete_rank_ladder_v1 as ladderrun
import run_temporal_iswas_rank46_task_typed_mode_causal_factorial_v1 as factorial
import run_temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v1 as edgeimpl
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_iswas_rank46_task_rank4_joint_source_factorial_v1 as joint

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank46_shared8_pairwise_mobius_v1.json"
JOINT_RESULT = ROOT / "circuits/followups/temporal_iswas_rank46_task_rank4_joint_source_factorial_v1_result.json"
JOINT_IMPL = ROOT / "ops/run_temporal_iswas_rank46_task_rank4_joint_source_factorial_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_rank46_shared8_pairwise_mobius_v1_result.json"
EXPECTED = {
    "prior": "dfb3600bfdc529200ae41e4e803131d62ef8bb12e89c843867d55af5d63dc187",
    "joint_result": "3f4a1a1260d5b1493bc14cca74c03f1dd03523e773b6fb8b8c197da7a6c31e5f",
    "joint_impl": "c42b5e0870c0e903a4db83ae0076d2c8c117b5ba72154088cf7f26fe1114303b",
}
SHARED = ("MLP0", "MLP1", "MLP2", "MLP3", "MLP4", "MLP6", "MLP7", "MLP8")
TASKS = ("temporal", "iswas")
MAX_FORWARDS = 48


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def response_vectors(backend, fresh, base, changed, qs, modes):
    torch = backend.torch
    width = int(backend.model.config.n_embd // backend.model.config.n_head)
    result = {}
    for task in TASKS:
        ids = [i for i, row in enumerate(fresh["rows"]) if klfit.task_name(row) == task]
        pieces = []
        for site in edgeimpl.RESPONSE_SITES:
            rb = qs[site] @ modes[site][task]
            b = edgeimpl.valid_rows(torch, edgeimpl.raw_response(base, site, width), fresh["batch"], ids)
            x = edgeimpl.valid_rows(torch, edgeimpl.raw_response(changed, site, width), fresh["batch"], ids)
            pieces.append(((x - b) @ rb).reshape(-1).float())
        result[task] = torch.cat(pieces)
    return result


def vector_metrics(torch, value, target):
    denominator = target.square().sum().clamp_min(1e-30)
    return {
        "signed_projection": float((value * target).sum() / denominator),
        "relative_squared_error": float((value - target).square().sum() / denominator),
        "norm_ratio": float(value.square().sum().sqrt() / denominator.sqrt()),
    }


def main():
    observed = {
        "prior": sha(PRIOR),
        "joint_result": sha(JOINT_RESULT),
        "joint_impl": sha(JOINT_IMPL),
    }
    authority = json.loads(JOINT_RESULT.read_text())
    frozen_shared = tuple(authority["arms"]["intersection"])
    subsets = [()] + [(site,) for site in SHARED] + list(combinations(SHARED, 2)) + [SHARED]
    dry = {
        "candidate_id": "temporal_auxiliary.iswas_rank46_shared8_pairwise_mobius_v1",
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "shared_sources": list(SHARED),
        "subset_count": len(subsets),
        "model_forwards_max": MAX_FORWARDS,
        "fit_updates": 0,
        "model_updates": 0,
        "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)

    started, tic = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    fitted = pooled.fit(backend)
    qs = fitted["projector"]
    task_rows, full_modes = ladderrun.fit_full_modes(backend, fitted)
    modes = {
        site: {task: full_modes[site][task][:, :4] for task in TASKS}
        for site in edgeimpl.RESPONSE_SITES
    }
    fresh = oodctx.capture(backend, factorial.TCAP, factorial.ICAP)
    _unused, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"])
    base_output, base = edgeimpl.capture_with_source_patch(backend, fresh["batch"], fresh["base_full"], ())
    donor_output, donor = edgeimpl.capture_with_source_patch(backend, fresh["donor_batch"], donor_full, ())
    base_state = atlasrun.states(torch, backend, base_output, fresh["rows"])
    donor_state = atlasrun.states(torch, backend, donor_output, fresh["rows"])
    target = response_vectors(backend, fresh, base, donor, qs, modes)

    vectors = {}
    behavior = {}
    finite = []
    full_output = None
    for subset in subsets:
        output, changed = edgeimpl.capture_with_source_patch(backend, fresh["batch"], donor_full, subset)
        key = "+".join(subset) if subset else "empty"
        vectors[key] = response_vectors(backend, fresh, base, changed, qs, modes)
        changed_state = atlasrun.states(torch, backend, output, fresh["rows"])
        behavior[key] = joint.behavior_report(backend, fresh["rows"], base_state, donor_state, changed_state)
        finite.extend(behavior[key].values())
        if subset == SHARED:
            full_output = output

    empty = vectors["empty"]
    singleton_sum = {}
    pair2_sum = {}
    singleton_metrics = {}
    pair2_metrics = {}
    pair_reports = {}
    pooled_pair_norms = {}
    max_pair_norm = {task: 0.0 for task in TASKS}
    for task in TASKS:
        singleton_sum[task] = empty[task].clone()
        for site in SHARED:
            singleton_sum[task] += vectors[site][task] - empty[task]
        pair2_sum[task] = singleton_sum[task].clone()
        singleton_metrics[task] = vector_metrics(torch, singleton_sum[task], target[task])

    for left, right in combinations(SHARED, 2):
        name = f"{left}+{right}"
        pair_reports[name] = {}
        pooled_sq = 0.0
        for task in TASKS:
            delta = vectors[name][task] - vectors[left][task] - vectors[right][task] + empty[task]
            pair2_sum[task] += delta
            metrics = vector_metrics(torch, delta, target[task])
            residual = target[task] - singleton_sum[task]
            residual_den = residual.square().sum().clamp_min(1e-30)
            metrics["residual_projection"] = float((delta * residual).sum() / residual_den)
            pair_reports[name][task] = metrics
            max_pair_norm[task] = max(max_pair_norm[task], metrics["norm_ratio"])
            pooled_sq += metrics["norm_ratio"] ** 2
        pooled_pair_norms[name] = math.sqrt(pooled_sq)

    for task in TASKS:
        pair2_metrics[task] = vector_metrics(torch, pair2_sum[task], target[task])
    ordered_pairs = sorted(pooled_pair_norms, key=lambda name: (-pooled_pair_norms[name], name))
    total_mass = sum(pooled_pair_norms.values())
    top8_mass_fraction = sum(pooled_pair_norms[name] for name in ordered_pairs[:8]) / max(1e-30, total_mass)

    full_key = "+".join(SHARED)
    full_metrics = {task: vector_metrics(torch, vectors[full_key][task], target[task]) for task in TASKS}
    _cunused, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    control_output, _ = edgeimpl.capture_with_source_patch(backend, fresh["control_batch"], control_donor_full, SHARED)
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, fresh["control_base_state"]).float()}
    controls = klfit.control_metrics(
        backend,
        control_ctx,
        das.head_logits(backend, atlasrun.states(torch, backend, control_output, fresh["controls"])).float(),
    )
    finite += [value for report in full_metrics.values() for value in report.values()]
    finite += [value for report in singleton_metrics.values() for value in report.values()]
    finite += [value for report in pair2_metrics.values() for value in report.values()]
    finite += list(max_pair_norm.values()) + [top8_mass_fraction]

    fit_ids = {row["row_id"] for rows in task_rows.values() for row in rows} | {
        row["row_id"] for row in fitted["control_rows"]
    }
    eval_ids = {row["row_id"] for row in fresh["rows"] + fresh["controls"]}
    pa = (
        observed == EXPECTED
        and frozen_shared == SHARED
        and len(subsets) == 38
        and not (fit_ids & eval_ids)
        and all(math.isfinite(float(value)) for value in finite)
        and min(full_metrics[task]["signed_projection"] for task in TASKS) >= 0.8
        and min(behavior[full_key][task] for task in TASKS) >= 0.75
    )
    pb = all(
        singleton_metrics[task]["signed_projection"] >= 1.5
        and singleton_metrics[task]["relative_squared_error"] >= 1.0
        for task in TASKS
    )
    pc = min(max_pair_norm.values()) >= 0.05
    improvements = {
        task: singleton_metrics[task]["relative_squared_error"] - pair2_metrics[task]["relative_squared_error"]
        for task in TASKS
    }
    pd = min(improvements.values()) >= 0.10
    pe = top8_mass_fraction >= 0.50
    predictions = {
        "pred_a_authority_and_full_intersection_replay": bool(pa),
        "pred_b_singleton_nonadditivity_reproduces": bool(pb),
        "pred_c_nontrivial_pair_interaction_both_tasks": bool(pc),
        "pred_d_degree_two_reduces_error_both_tasks": bool(pd),
        "pred_e_pair_interactions_are_sparse": bool(pe),
    }
    terminal = (
        "invalid"
        if not pa
        else "sparse_degree_two_interaction_map"
        if all(predictions.values())
        else "higher_order_or_dense_interaction_boundary"
    )
    result = {
        "schema": "temporal_iswas_rank46_shared8_pairwise_mobius_result_v1",
        "started_utc": started,
        "finished_utc": now(),
        "serial_seconds": time.perf_counter() - tic,
        "authority_sha256": EXPECTED,
        "shared_sources": list(SHARED),
        "subset_count": len(subsets),
        "full_intersection": {"response": full_metrics, "behavior_signed_projection": behavior[full_key]},
        "controls": controls,
        "singleton_sum": singleton_metrics,
        "pairwise_sum": pair2_metrics,
        "pairwise_rse_improvement": improvements,
        "max_pair_norm_ratio": max_pair_norm,
        "top8_pair_mass_fraction": top8_mass_fraction,
        "top_pairs": [{"pair": name, "pooled_norm": pooled_pair_norms[name]} for name in ordered_pairs],
        "pair_reports": pair_reports,
        "predictions": predictions,
        "terminal": terminal,
        "price": {
            "model_forwards_max": MAX_FORWARDS,
            "fit_updates": 0,
            "model_updates": 0,
            "transformer_backwards": 0,
        },
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "full_intersection", "singleton_sum", "pairwise_sum", "pairwise_rse_improvement",
        "max_pair_norm_ratio", "top8_pair_mass_fraction", "top_pairs", "predictions", "terminal", "price"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
