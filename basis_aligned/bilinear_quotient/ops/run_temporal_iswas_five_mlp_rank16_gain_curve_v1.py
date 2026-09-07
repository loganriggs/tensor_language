#!/usr/bin/env python3
"""Fixed scalar gain curve on the selective all-position rank-16 source program."""
# BQGATE: EXPERIMENT pred_a_authority_basis_replay_finiteness_and_price pred_b_a_registered_gain_is_functional pred_c_selected_gain_is_selective pred_d_selected_gain_is_at_most_1p25 pred_e_selected_response_does_not_overshoot
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
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_five_mlp_rank16_gain_curve_v1.json"
LADDER_RESULT = ROOT / "circuits/followups/temporal_iswas_five_mlp_position_svd_ladder_v1_result.json"
LADDER_IMPL = ROOT / "ops/run_temporal_iswas_five_mlp_position_svd_ladder_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_five_mlp_rank16_gain_curve_v1_result.json"
EXPECTED = {
    "prior": "0171b555be641c258a89efdf8144e21b7d0c9735d9446d7573074077a0377f5c",
    "ladder_result": "b6f4562bcc6e19dc808a0debaf35d32b0d6c1dbd6c6a6e7846a92253b89dc014",
    "ladder_impl": "cc2733a518e015b3cccde4d109e46125d69e90fa60555a593c921f6f0e860236",
}
GAINS = (1.00, 1.10, 1.15, 1.20, 1.25, 1.30)
SUPPORT, TASKS = sourcev1.SUPPORT, sourcev1.TASKS
MAX_FORWARDS = 32


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    observed = {"prior": sha(PRIOR), "ladder_result": sha(LADDER_RESULT), "ladder_impl": sha(LADDER_IMPL)}
    authority = json.loads(LADDER_RESULT.read_text())
    dry = {"candidate_id": "temporal_auxiliary.iswas_five_mlp_rank16_gain_curve_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "gains": list(GAINS),
           "rank": 16, "support": list(SUPPORT), "model_forwards_max": MAX_FORWARDS,
           "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    fitted = pooled.fit(backend); qs = fitted["projector"]; _task_rows, full_modes = ladderrun.fit_full_modes(backend, fitted)
    modes = {site: {task: full_modes[site][task][:, :4] for task in TASKS} for site in edgeimpl.RESPONSE_SITES}
    bases, _spectra, _energy, position_counts = position.fit_position_bases(backend, fitted["target_rows"])
    fresh = oodctx.capture(backend, factorial.TCAP, factorial.ICAP)
    _unused, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"])
    _cunused, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    base_output, base = edgeimpl.capture_with_source_patch(backend, fresh["batch"], fresh["base_full"], ())
    donor_output, donor = edgeimpl.capture_with_source_patch(backend, fresh["donor_batch"], donor_full, ())
    base_state = atlasrun.states(torch, backend, base_output, fresh["rows"]); donor_state = atlasrun.states(torch, backend, donor_output, fresh["rows"])
    target = pairimpl.response_vectors(backend, fresh, base, donor, qs, modes)
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, fresh["control_base_state"]).float()}
    reports, finite = {}, []
    for gain in GAINS:
        scaled = {site: {"scaled": bases[site]["rank16"] * math.sqrt(gain)} for site in SUPPORT}
        output, changed = sourcev1.capture_projected(backend, fresh["batch"], fresh["base_full"], donor_full, scaled, "scaled")
        control_output, _ = sourcev1.capture_projected(backend, fresh["control_batch"], fresh["control_base_full"], control_donor_full, scaled, "scaled")
        vectors = pairimpl.response_vectors(backend, fresh, base, changed, qs, modes)
        response = {task: pairimpl.vector_metrics(torch, vectors[task], target[task]) for task in TASKS}
        behavior = joint.behavior_report(backend, fresh["rows"], base_state, donor_state, atlasrun.states(torch, backend, output, fresh["rows"]))
        control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, control_output, fresh["controls"])).float())
        reports[f"{gain:.2f}"] = {"response": response, "behavior_signed_projection": behavior, "control": control,
                                    "functional": sourcev1.functional({"response": response, "behavior_signed_projection": behavior})}
        finite += list(behavior.values()) + [value for task in TASKS for value in response[task].values()]
        finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]
    replay = authority["reports"]["rank16"]
    replay_error = max(
        max(abs(reports["1.00"]["response"][task][metric] - replay["response"][task][metric]) for task in TASKS for metric in ("signed_projection", "relative_squared_error", "norm_ratio")),
        max(abs(reports["1.00"]["behavior_signed_projection"][task] - replay["behavior_signed_projection"][task]) for task in TASKS),
        abs(reports["1.00"]["control"]["median_kl"] - replay["control"]["median_kl"]),
        abs(reports["1.00"]["control"]["top1_flip_fraction"] - replay["control"]["top1_flip_fraction"]),
    )
    functional_gains = [gain for gain in GAINS if reports[f"{gain:.2f}"]["functional"]]
    selected = min(functional_gains) if functional_gains else None
    selected_report = reports[f"{selected:.2f}"] if selected is not None else None
    pa = observed == EXPECTED and authority["atomic_instrument"]["published_measurements_finite"] and min(position_counts.values()) >= 128 and replay_error <= 1e-5 and all(math.isfinite(float(v)) for v in finite)
    pb = selected is not None
    pc = pb and selected_report["control"]["median_kl"] <= .02 and selected_report["control"]["top1_flip_fraction"] == 0.0
    pd = pb and selected <= 1.25
    pe = pb and max(selected_report["response"][task]["signed_projection"] for task in TASKS) <= 1.20
    predictions = {"pred_a_authority_basis_replay_finiteness_and_price": bool(pa), "pred_b_a_registered_gain_is_functional": bool(pb),
                   "pred_c_selected_gain_is_selective": bool(pc), "pred_d_selected_gain_is_at_most_1p25": bool(pd),
                   "pred_e_selected_response_does_not_overshoot": bool(pe)}
    terminal = "invalid" if not pa else "selective_gain_calibrated_source_program" if all(predictions.values()) else "gain_calibration_boundary"
    result = {"schema": "temporal_iswas_five_mlp_rank16_gain_curve_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "rank16_replay_max_abs_error": replay_error,
              "gains": list(GAINS), "reports": reports, "functional_gains": functional_gains, "selected_gain": selected,
              "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards_max": MAX_FORWARDS, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("rank16_replay_max_abs_error", "reports", "functional_gains", "selected_gain", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
