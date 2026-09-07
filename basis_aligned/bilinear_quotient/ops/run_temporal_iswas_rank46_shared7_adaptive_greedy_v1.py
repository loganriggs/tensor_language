#!/usr/bin/env python3
"""Adaptive backward deletion of the shared two-task physical writer bank."""
# BQGATE: EXPERIMENT pred_a_authority_initial_replay_finiteness_and_price pred_b_terminal_support_has_at_most_five_sources pred_c_greedy_path_reaches_selectivity pred_d_terminal_retains_a_late_behavior_module pred_e_control_kl_improves_by_point02
from datetime import datetime, timezone
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
import run_temporal_iswas_rank46_shared8_pairwise_mobius_v1 as pairimpl

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank46_shared7_adaptive_greedy_v1.json"
ORDERED = ROOT / "circuits/followups/temporal_iswas_rank46_shared8_ordered_conditional_deletion_v1_result.json"
PAIR_IMPL = ROOT / "ops/run_temporal_iswas_rank46_shared8_pairwise_mobius_v1.py"
JOINT_IMPL = ROOT / "ops/run_temporal_iswas_rank46_task_rank4_joint_source_factorial_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_rank46_shared7_adaptive_greedy_v1_result.json"
EXPECTED = {
    "prior": "0cb8c8b949503ddb95d856c99a87efaa8b0860f8b339cccbbd74dbfc52342fe6",
    "ordered": "5803747e3891aadbc021f9e661b268c8fd70f989315d8f4e5c9dd02e742b87c5",
    "pair_impl": "bded542af69309ed27c076de1da0a0aa533a9a2aa3d90d434167041e63c17b55",
    "joint_impl": "c42b5e0870c0e903a4db83ae0076d2c8c117b5ba72154088cf7f26fe1114303b",
}
INITIAL = ("MLP0", "MLP1", "MLP2", "MLP3", "MLP4", "MLP6", "MLP7")
CANONICAL = INITIAL
TASKS = ("temporal", "iswas")
MAX_FORWARDS = 72


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def functional(report):
    return all(
        report["response"][task]["signed_projection"] >= 0.8
        and report["response"][task]["relative_squared_error"] <= 0.2
        and report["behavior_signed_projection"][task] >= 0.75
        for task in TASKS
    )


def selective(report):
    return functional(report) and report["control"]["median_kl"] <= 0.02 and report["control"]["top1_flip_fraction"] == 0.0


def main():
    observed = {"prior": sha(PRIOR), "ordered": sha(ORDERED), "pair_impl": sha(PAIR_IMPL), "joint_impl": sha(JOINT_IMPL)}
    authority = json.loads(ORDERED.read_text())
    dry = {
        "candidate_id": "temporal_auxiliary.iswas_rank46_shared7_adaptive_greedy_v1",
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "initial_support": list(INITIAL),
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
    modes = {site: {task: full_modes[site][task][:, :4] for task in TASKS} for site in edgeimpl.RESPONSE_SITES}
    fresh = oodctx.capture(backend, factorial.TCAP, factorial.ICAP)
    _unused, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"])
    _cunused, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    base_output, base = edgeimpl.capture_with_source_patch(backend, fresh["batch"], fresh["base_full"], ())
    donor_output, donor = edgeimpl.capture_with_source_patch(backend, fresh["donor_batch"], donor_full, ())
    base_state = atlasrun.states(torch, backend, base_output, fresh["rows"])
    donor_state = atlasrun.states(torch, backend, donor_output, fresh["rows"])
    target = pairimpl.response_vectors(backend, fresh, base, donor, qs, modes)
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, fresh["control_base_state"]).float()}
    cache = {}
    finite = []

    def evaluate(support):
        support = tuple(support)
        if support in cache:
            return cache[support]
        output, changed = edgeimpl.capture_with_source_patch(backend, fresh["batch"], donor_full, support)
        vectors = pairimpl.response_vectors(backend, fresh, base, changed, qs, modes)
        response = {task: pairimpl.vector_metrics(torch, vectors[task], target[task]) for task in TASKS}
        behavior = joint.behavior_report(
            backend, fresh["rows"], base_state, donor_state,
            atlasrun.states(torch, backend, output, fresh["rows"]),
        )
        control_output, _ = edgeimpl.capture_with_source_patch(backend, fresh["control_batch"], control_donor_full, support)
        control = klfit.control_metrics(
            backend, control_ctx,
            das.head_logits(backend, atlasrun.states(torch, backend, control_output, fresh["controls"])).float(),
        )
        report = {"support": list(support), "response": response, "behavior_signed_projection": behavior, "control": control}
        cache[support] = report
        finite.extend(list(behavior.values()))
        finite.extend(value for task in TASKS for value in response[task].values())
        finite.extend(list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]])
        return report

    current = INITIAL
    current_report = evaluate(current)
    path = [{"support": list(current), "removed": None, "report": current_report, "selective": selective(current_report)}]
    stages = []
    while len(current) > 1:
        current_is_selective = selective(current_report)
        candidates = []
        for removed in current:
            remaining = tuple(site for site in current if site != removed)
            report = evaluate(remaining)
            eligible = selective(report) if current_is_selective else (
                functional(report) and report["control"]["median_kl"] <= current_report["control"]["median_kl"] + 1e-8
            )
            minimum_behavior = min(report["behavior_signed_projection"].values())
            minimum_response = min(report["response"][task]["signed_projection"] for task in TASKS)
            candidates.append({
                "removed": removed,
                "remaining": list(remaining),
                "eligible": eligible,
                "functional": functional(report),
                "selective": selective(report),
                "minimum_behavior_projection": minimum_behavior,
                "minimum_response_projection": minimum_response,
                "report": report,
            })
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        stages.append({"from_support": list(current), "current_selective": current_is_selective, "candidates": candidates})
        if not eligible:
            break
        chosen = min(
            eligible,
            key=lambda candidate: (
                candidate["report"]["control"]["median_kl"],
                candidate["report"]["control"]["top1_flip_fraction"],
                -candidate["minimum_behavior_projection"],
                -candidate["minimum_response_projection"],
                CANONICAL.index(candidate["removed"]),
            ),
        )
        current = tuple(chosen["remaining"])
        current_report = chosen["report"]
        path.append({"support": list(current), "removed": chosen["removed"], "report": current_report, "selective": selective(current_report)})

    fit_ids = {row["row_id"] for rows in task_rows.values() for row in rows} | {row["row_id"] for row in fitted["control_rows"]}
    eval_ids = {row["row_id"] for row in fresh["rows"] + fresh["controls"]}
    authority_initial = tuple(authority["selected_remaining_sources"])
    pa = (
        observed == EXPECTED and authority_initial == INITIAL and not (fit_ids & eval_ids)
        and functional(path[0]["report"]) and all(math.isfinite(float(value)) for value in finite)
        and len(cache) * 2 <= MAX_FORWARDS
    )
    pb = len(current) <= 5
    pc = any(step["selective"] for step in path)
    pd = bool({"MLP6", "MLP7"} & set(current))
    original_eight_kl = 0.05278966575860977
    kl_improvement = original_eight_kl - current_report["control"]["median_kl"]
    pe = kl_improvement >= 0.02
    predictions = {
        "pred_a_authority_initial_replay_finiteness_and_price": bool(pa),
        "pred_b_terminal_support_has_at_most_five_sources": bool(pb),
        "pred_c_greedy_path_reaches_selectivity": bool(pc),
        "pred_d_terminal_retains_a_late_behavior_module": bool(pd),
        "pred_e_control_kl_improves_by_point02": bool(pe),
    }
    terminal = "invalid" if not pa else "greedy_selective_shared_writer_program" if all(predictions.values()) else "greedy_shared_writer_boundary"
    result = {
        "schema": "temporal_iswas_rank46_shared7_adaptive_greedy_result_v1",
        "started_utc": started,
        "finished_utc": now(),
        "serial_seconds": time.perf_counter() - tic,
        "authority_sha256": EXPECTED,
        "initial_support": list(INITIAL),
        "path": path,
        "stages": stages,
        "terminal_support": list(current),
        "terminal_report": current_report,
        "terminal_selective": selective(current_report),
        "terminal_local_boundary": bool(stages and not any(candidate["eligible"] for candidate in stages[-1]["candidates"])),
        "kl_improvement_from_original_eight": kl_improvement,
        "evaluated_support_count": len(cache),
        "predictions": predictions,
        "terminal": terminal,
        "price": {"model_forwards": len(cache) * 2, "model_forwards_max": MAX_FORWARDS, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "path", "terminal_support", "terminal_report", "terminal_selective", "terminal_local_boundary",
        "kl_improvement_from_original_eight", "evaluated_support_count", "predictions", "terminal", "price"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
