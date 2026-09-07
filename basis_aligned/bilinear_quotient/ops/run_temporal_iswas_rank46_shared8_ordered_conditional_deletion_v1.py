#!/usr/bin/env python3
"""Ordered and full-context conditional atlas with first greedy deletion."""
# BQGATE: EXPERIMENT pred_a_authority_full_replay_finiteness_and_price pred_b_at_least_one_leave_one_out_is_functional pred_c_at_least_one_leave_one_out_is_selective pred_d_early_mlp0_4_prefix_is_functional pred_e_context_dependence_is_widespread
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
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank46_shared8_ordered_conditional_deletion_v1.json"
PAIR_V2 = ROOT / "circuits/followups/temporal_iswas_rank46_shared8_pairwise_mobius_v2_result.json"
JOINT_IMPL = ROOT / "ops/run_temporal_iswas_rank46_task_rank4_joint_source_factorial_v1.py"
PAIR_IMPL = ROOT / "ops/run_temporal_iswas_rank46_shared8_pairwise_mobius_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_rank46_shared8_ordered_conditional_deletion_v1_result.json"
EXPECTED = {
    "prior": "69f3ce615d4c8c899b35052fe00239527d6f643b63ac25f104fcfe4e097546bf",
    "pair_v2": "a86c42cedbba40740bcc8fb7479b57bb95ee710c435c0585ad65b58eea479b30",
    "joint_impl": "c42b5e0870c0e903a4db83ae0076d2c8c117b5ba72154088cf7f26fe1114303b",
    "pair_impl": "bded542af69309ed27c076de1da0a0aa533a9a2aa3d90d434167041e63c17b55",
}
SHARED = ("MLP0", "MLP1", "MLP2", "MLP3", "MLP4", "MLP6", "MLP7", "MLP8")
TASKS = ("temporal", "iswas")
MAX_FORWARDS = 40


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def subset_name(subset):
    return "+".join(subset) if subset else "empty"


def functional(report):
    return all(
        report["response"][task]["signed_projection"] >= 0.8
        and report["response"][task]["relative_squared_error"] <= 0.2
        and report["behavior_signed_projection"][task] >= 0.75
        for task in TASKS
    )


def main():
    observed = {"prior": sha(PRIOR), "pair_v2": sha(PAIR_V2), "joint_impl": sha(JOINT_IMPL), "pair_impl": sha(PAIR_IMPL)}
    prefixes = {site: SHARED[: index + 1] for index, site in enumerate(SHARED)}
    leave_one_out = {site: tuple(value for value in SHARED if value != site) for site in SHARED}
    requested = [()] + [(site,) for site in SHARED] + list(prefixes.values()) + list(leave_one_out.values()) + [SHARED]
    subsets = []
    for subset in requested:
        if subset not in subsets:
            subsets.append(subset)
    dry = {
        "candidate_id": "temporal_auxiliary.iswas_rank46_shared8_ordered_conditional_deletion_v1",
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "shared_sources": list(SHARED),
        "unique_response_arms": len(subsets),
        "control_arms": 9,
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
    base_output, base = edgeimpl.capture_with_source_patch(backend, fresh["batch"], fresh["base_full"], ())
    donor_output, donor = edgeimpl.capture_with_source_patch(backend, fresh["donor_batch"], donor_full, ())
    base_state = atlasrun.states(torch, backend, base_output, fresh["rows"])
    donor_state = atlasrun.states(torch, backend, donor_output, fresh["rows"])
    target = pairimpl.response_vectors(backend, fresh, base, donor, qs, modes)

    vectors = {}
    reports = {}
    finite = []
    for subset in subsets:
        output, changed = edgeimpl.capture_with_source_patch(backend, fresh["batch"], donor_full, subset)
        key = subset_name(subset)
        vectors[key] = pairimpl.response_vectors(backend, fresh, base, changed, qs, modes)
        behavior = joint.behavior_report(
            backend, fresh["rows"], base_state, donor_state,
            atlasrun.states(torch, backend, output, fresh["rows"]),
        )
        response = {task: pairimpl.vector_metrics(torch, vectors[key][task], target[task]) for task in TASKS}
        reports[key] = {"sources": list(subset), "response": response, "behavior_signed_projection": behavior}
        finite += list(behavior.values()) + [value for task in TASKS for value in response[task].values()]

    _cunused, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, fresh["control_base_state"]).float()}
    control_reports = {}
    for removed, subset in [("none", SHARED)] + list(leave_one_out.items()):
        output, _changed = edgeimpl.capture_with_source_patch(backend, fresh["control_batch"], control_donor_full, subset)
        metrics = klfit.control_metrics(
            backend, control_ctx,
            das.head_logits(backend, atlasrun.states(torch, backend, output, fresh["controls"])).float(),
        )
        control_reports[removed] = metrics
        finite += list(metrics["margin_rms_fraction"].values()) + [metrics["median_kl"], metrics["max_kl"], metrics["top1_flip_fraction"]]

    empty = vectors["empty"]
    full = vectors[subset_name(SHARED)]
    conditional = {}
    context_counts = {task: 0 for task in TASKS}
    previous = ()
    for site in SHARED:
        singleton_key = site
        prefix = prefixes[site]
        prefix_key = subset_name(prefix)
        previous_key = subset_name(previous)
        loo_key = subset_name(leave_one_out[site])
        conditional[site] = {}
        for task in TASKS:
            singleton = vectors[singleton_key][task] - empty[task]
            prefix_increment = vectors[prefix_key][task] - vectors[previous_key][task]
            full_increment = full[task] - vectors[loo_key][task]
            sm = pairimpl.vector_metrics(torch, singleton, target[task])
            pm = pairimpl.vector_metrics(torch, prefix_increment, target[task])
            fm = pairimpl.vector_metrics(torch, full_increment, target[task])
            max_shift = max(abs(pm["signed_projection"] - sm["signed_projection"]), abs(fm["signed_projection"] - sm["signed_projection"]))
            conditional[site][task] = {"singleton": sm, "ordered_prefix_increment": pm, "full_context_increment": fm, "max_projection_shift_from_singleton": max_shift}
            if max_shift >= 0.10:
                context_counts[task] += 1
            finite += list(sm.values()) + list(pm.values()) + list(fm.values()) + [max_shift]
        previous = prefix

    loo_reports = {}
    functional_sites = []
    selective_sites = []
    for site in SHARED:
        report = reports[subset_name(leave_one_out[site])]
        is_functional = functional(report)
        control = control_reports[site]
        is_selective = is_functional and control["median_kl"] <= 0.02 and control["top1_flip_fraction"] == 0.0
        min_response = min(report["response"][task]["signed_projection"] for task in TASKS)
        loo_reports[site] = {
            "remaining_sources": list(leave_one_out[site]),
            "response": report["response"],
            "behavior_signed_projection": report["behavior_signed_projection"],
            "control": control,
            "functional": is_functional,
            "selective": is_selective,
            "minimum_response_projection": min_response,
        }
        if is_functional:
            functional_sites.append(site)
        if is_selective:
            selective_sites.append(site)

    pool = selective_sites if selective_sites else functional_sites
    selected_removed = min(
        pool,
        key=lambda site: (control_reports[site]["median_kl"], -loo_reports[site]["minimum_response_projection"], SHARED.index(site)),
    ) if pool else None
    selected_status = "selective" if selected_removed in selective_sites else "functional_frontier" if selected_removed else None
    full_report = reports[subset_name(SHARED)]
    early_report = reports[subset_name(SHARED[:5])]
    fit_ids = {row["row_id"] for rows in task_rows.values() for row in rows} | {row["row_id"] for row in fitted["control_rows"]}
    eval_ids = {row["row_id"] for row in fresh["rows"] + fresh["controls"]}
    pa = (
        observed == EXPECTED and len(subsets) == 23 and not (fit_ids & eval_ids)
        and all(math.isfinite(float(value)) for value in finite) and functional(full_report)
    )
    pb = bool(functional_sites)
    pc = bool(selective_sites)
    pd = functional(early_report)
    pe = min(context_counts.values()) >= 5
    predictions = {
        "pred_a_authority_full_replay_finiteness_and_price": bool(pa),
        "pred_b_at_least_one_leave_one_out_is_functional": bool(pb),
        "pred_c_at_least_one_leave_one_out_is_selective": bool(pc),
        "pred_d_early_mlp0_4_prefix_is_functional": bool(pd),
        "pred_e_context_dependence_is_widespread": bool(pe),
    }
    terminal = "invalid" if not pa else "ordered_conditional_selective_deletion" if all(predictions.values()) else "ordered_conditional_deletion_frontier"
    result = {
        "schema": "temporal_iswas_rank46_shared8_ordered_conditional_deletion_result_v1",
        "started_utc": started,
        "finished_utc": now(),
        "serial_seconds": time.perf_counter() - tic,
        "authority_sha256": EXPECTED,
        "shared_sources": list(SHARED),
        "unique_response_arms": len(subsets),
        "full_report": full_report,
        "early_mlp0_4_report": early_report,
        "full_control": control_reports["none"],
        "conditional_increments": conditional,
        "context_dependent_count": context_counts,
        "leave_one_out": loo_reports,
        "selected_removed": selected_removed,
        "selected_status": selected_status,
        "selected_remaining_sources": list(leave_one_out[selected_removed]) if selected_removed else None,
        "predictions": predictions,
        "terminal": terminal,
        "price": {"model_forwards_max": MAX_FORWARDS, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "full_report", "early_mlp0_4_report", "context_dependent_count", "leave_one_out",
        "selected_removed", "selected_status", "selected_remaining_sources", "predictions", "terminal", "price"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
