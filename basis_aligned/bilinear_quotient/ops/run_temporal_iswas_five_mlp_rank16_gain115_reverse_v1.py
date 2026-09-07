#!/usr/bin/env python3
"""Reverse OOD execution of the frozen five-MLP rank-16 gain-1.15 program."""
# BQGATE: EXPERIMENT pred_a_authority_basis_endpoint_finiteness_and_price pred_b_full_five_source_reverse_is_functional pred_c_projected_gain115_reverse_is_functional pred_d_projected_reverse_is_selective pred_e_forward_reverse_behavior_is_stable
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
import run_temporal_iswas_five_mlp_source_dim_subspace_v1 as sourcev1
import run_temporal_iswas_five_mlp_position_svd_ladder_v1 as position

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_five_mlp_rank16_gain115_reverse_v1.json"
GAIN_RESULT = ROOT / "circuits/followups/temporal_iswas_five_mlp_rank16_gain_curve_v1_result.json"
GAIN_IMPL = ROOT / "ops/run_temporal_iswas_five_mlp_rank16_gain_curve_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_five_mlp_rank16_gain115_reverse_v1_result.json"
EXPECTED = {
    "prior": "1c420ca13e37fc82e2ecbf1b8edc523d27018558d75d20d5aad2852c8225309b",
    "gain_result": "958f58cf6ac05973b65a38cd258093926999351016f1aa1bc2ec608c4844b75d",
    "gain_impl": "f9646695d72ba891a98d36357a7d5da3c1b25be81136fc65b7a374120782230a",
}
SUPPORT, TASKS, GAIN = sourcev1.SUPPORT, sourcev1.TASKS, 1.15
MAX_FORWARDS = 28


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def make_report(backend, fresh, reverse_fresh, baseline_response, target_response, changed_response,
                baseline_state, target_state, changed_state, qs, modes):
    torch = backend.torch
    target = pairimpl.response_vectors(backend, reverse_fresh, baseline_response, target_response, qs, modes)
    vectors = pairimpl.response_vectors(backend, reverse_fresh, baseline_response, changed_response, qs, modes)
    response = {task: pairimpl.vector_metrics(torch, vectors[task], target[task]) for task in TASKS}
    behavior = joint.behavior_report(backend, fresh["rows"], baseline_state, target_state, changed_state)
    return {"response": response, "behavior_signed_projection": behavior}


def main():
    observed = {"prior": sha(PRIOR), "gain_result": sha(GAIN_RESULT), "gain_impl": sha(GAIN_IMPL)}
    gain_authority = json.loads(GAIN_RESULT.read_text())
    dry = {"candidate_id": "temporal_auxiliary.iswas_five_mlp_rank16_gain115_reverse_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "gain": GAIN,
           "rank": 16, "support": list(SUPPORT), "model_forwards_max": MAX_FORWARDS,
           "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    fitted = pooled.fit(backend); qs = fitted["projector"]; _task_rows, full_modes = ladderrun.fit_full_modes(backend, fitted)
    modes = {site: {task: full_modes[site][task][:, :4] for task in TASKS} for site in edgeimpl.RESPONSE_SITES}
    bases, _spectra, _energy, position_counts = position.fit_position_bases(backend, fitted["target_rows"])
    scaled = {site: {"scaled": bases[site]["rank16"] * math.sqrt(GAIN)} for site in SUPPORT}
    fresh = oodctx.capture(backend, factorial.TCAP, factorial.ICAP)
    base_output, base = edgeimpl.capture_with_source_patch(backend, fresh["batch"], fresh["base_full"], ())
    donor_output, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"])
    _donor_observe_output, donor = edgeimpl.capture_with_source_patch(backend, fresh["donor_batch"], donor_full, ())
    control_donor_output, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    base_state = atlasrun.states(torch, backend, base_output, fresh["rows"]); donor_state = atlasrun.states(torch, backend, donor_output, fresh["rows"])
    reverse_fresh = {"rows": fresh["rows"], "batch": fresh["donor_batch"]}

    full_output, full_response = edgeimpl.capture_with_source_patch(backend, fresh["donor_batch"], fresh["base_full"], SUPPORT)
    projected_output, projected_response = sourcev1.capture_projected(
        backend, fresh["donor_batch"], donor_full, fresh["base_full"], scaled, "scaled")
    full_report = make_report(backend, fresh, reverse_fresh, donor, base, full_response, donor_state, base_state,
                              atlasrun.states(torch, backend, full_output, fresh["rows"]), qs, modes)
    projected_report = make_report(backend, fresh, reverse_fresh, donor, base, projected_response, donor_state, base_state,
                                   atlasrun.states(torch, backend, projected_output, fresh["rows"]), qs, modes)
    control_output, _ = sourcev1.capture_projected(
        backend, fresh["control_donor_batch"], control_donor_full, fresh["control_base_full"], scaled, "scaled")
    control_donor_state = atlasrun.states(torch, backend, control_donor_output, fresh["controls"])
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, control_donor_state).float()}
    control = klfit.control_metrics(backend, control_ctx,
        das.head_logits(backend, atlasrun.states(torch, backend, control_output, fresh["controls"])).float())
    projected_report["control"] = control
    finite = [value for report in (full_report, projected_report) for section in ("response", "behavior_signed_projection")
              for value in ([v for task in TASKS for v in report[section][task].values()] if section == "response" else report[section].values())]
    finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]
    pa = observed == EXPECTED and gain_authority["selected_gain"] == GAIN and min(position_counts.values()) >= 128 and all(math.isfinite(float(v)) for v in finite)
    pb = sourcev1.functional(full_report)
    pc = sourcev1.functional(projected_report)
    pd = pc and control["median_kl"] <= .02 and control["top1_flip_fraction"] == 0.0
    forward_behavior = gain_authority["reports"]["1.15"]["behavior_signed_projection"]
    behavior_difference = {task: abs(projected_report["behavior_signed_projection"][task] - forward_behavior[task]) for task in TASKS}
    pe = max(behavior_difference.values()) <= .10
    predictions = {"pred_a_authority_basis_endpoint_finiteness_and_price": bool(pa),
                   "pred_b_full_five_source_reverse_is_functional": bool(pb),
                   "pred_c_projected_gain115_reverse_is_functional": bool(pc),
                   "pred_d_projected_reverse_is_selective": bool(pd),
                   "pred_e_forward_reverse_behavior_is_stable": bool(pe)}
    terminal = "invalid" if not pa else "bidirectional_selective_five_mlp_rank16_program" if all(predictions.values()) else "reverse_source_program_boundary"
    result = {"schema": "temporal_iswas_five_mlp_rank16_gain115_reverse_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "support": list(SUPPORT), "rank": 16, "gain": GAIN,
              "full_reverse": full_report, "projected_reverse": projected_report, "forward_reverse_behavior_abs_difference": behavior_difference,
              "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards_max": MAX_FORWARDS, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("full_reverse", "projected_reverse", "forward_reverse_behavior_abs_difference", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
