#!/usr/bin/env python3
"""All-position source-output SVD ladder at the five-MLP circuit boundary."""
# BQGATE: EXPERIMENT pred_a_atomic_authority_split_full_replay_and_finiteness pred_b_a_position_svd_rank_is_functional pred_c_selected_rank_is_selective pred_d_selected_rank_is_at_most_64 pred_e_rank64_complement_is_insufficient
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
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_iswas_rank46_task_rank4_joint_source_factorial_v1 as joint
import run_temporal_iswas_rank46_shared8_pairwise_mobius_v1 as pairimpl
import run_temporal_iswas_five_mlp_source_dim_subspace_v1 as sourcev1

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_five_mlp_position_svd_ladder_v1.json"
SOURCE_V2 = ROOT / "circuits/followups/temporal_iswas_five_mlp_source_dim_subspace_v2_result.json"
SOURCE_IMPL = ROOT / "ops/run_temporal_iswas_five_mlp_source_dim_subspace_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_five_mlp_position_svd_ladder_v1_result.json"
EXPECTED = {
    "prior": "4f6deabf8ed3cf21915341f0149fc9c660cb399a5a73e877f9ac1b4f64aba763",
    "source_v2": "80fa5ff1ad7463260ad10aacefac2fb3c8a8cd36fac1ed4381e5a11b103dba2a",
    "source_impl": "f12b51dee33ec5f553f084d77749d14ffd7fdd3d041d7567d55739db9b1e530d",
}
SUPPORT = sourcev1.SUPPORT
TASKS = sourcev1.TASKS
RANKS = (8, 16, 32, 64, 128)
ARMS = ("full",) + tuple(f"rank{rank}" for rank in RANKS) + ("rank64_complement",)
MAX_FORWARDS = 36


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def fit_position_bases(backend, rows):
    torch = backend.torch
    batch, _bo, base, _do, donor, _bi, _di = interface.cap_inputs(backend, rows)
    bases, spectra, energy, row_counts = {}, {}, {}, {}
    for site in SUPPORT:
        _kind, layer, _head = atlasrun.site_parts(site)
        matrix = interface.valid_delta(batch, base["mlp"][layer], donor["mlp"][layer]).to(backend.device).float()
        _u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        bases[site] = {f"rank{rank}": vh[:rank].T.contiguous() for rank in RANKS}
        denominator = singular.square().sum().clamp_min(1e-30)
        spectra[site] = [float(value) for value in singular[:140]]
        energy[site] = {f"rank{rank}": float(singular[:rank].square().sum() / denominator) for rank in RANKS}
        row_counts[site] = int(matrix.shape[0])
    return bases, spectra, energy, row_counts


def main():
    observed = {"prior": sha(PRIOR), "source_v2": sha(SOURCE_V2), "source_impl": sha(SOURCE_IMPL)}
    source_v2 = json.loads(SOURCE_V2.read_text())
    dry = {"candidate_id": "temporal_auxiliary.iswas_five_mlp_position_svd_ladder_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "support": list(SUPPORT),
           "ranks": list(RANKS), "arms": list(ARMS), "model_forwards_max": MAX_FORWARDS,
           "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)

    started, tic = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    fitted = pooled.fit(backend)
    qs = fitted["projector"]
    _task_rows, full_modes = ladderrun.fit_full_modes(backend, fitted)
    modes = {site: {task: full_modes[site][task][:, :4] for task in TASKS} for site in edgeimpl.RESPONSE_SITES}
    bases, spectra, fit_energy, position_counts = fit_position_bases(backend, fitted["target_rows"])
    fresh = oodctx.capture(backend, factorial.TCAP, factorial.ICAP)
    _unused, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"])
    _cunused, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    base_output, base = edgeimpl.capture_with_source_patch(backend, fresh["batch"], fresh["base_full"], ())
    donor_output, donor = edgeimpl.capture_with_source_patch(backend, fresh["donor_batch"], donor_full, ())
    base_state = atlasrun.states(torch, backend, base_output, fresh["rows"])
    donor_state = atlasrun.states(torch, backend, donor_output, fresh["rows"])
    target = pairimpl.response_vectors(backend, fresh, base, donor, qs, modes)
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, fresh["control_base_state"]).float()}
    reports, finite = {}, []
    for arm in ARMS:
        if arm == "full":
            output, changed = edgeimpl.capture_with_source_patch(backend, fresh["batch"], donor_full, SUPPORT)
            control_output, _ = edgeimpl.capture_with_source_patch(backend, fresh["control_batch"], control_donor_full, SUPPORT)
        else:
            basis_arm = "rank64" if arm == "rank64_complement" else arm
            output, changed = sourcev1.capture_projected(backend, fresh["batch"], fresh["base_full"], donor_full, bases, basis_arm,
                                                        complement=arm == "rank64_complement")
            control_output, _ = sourcev1.capture_projected(backend, fresh["control_batch"], fresh["control_base_full"], control_donor_full,
                                                           bases, basis_arm, complement=arm == "rank64_complement")
        vectors = pairimpl.response_vectors(backend, fresh, base, changed, qs, modes)
        response = {task: pairimpl.vector_metrics(torch, vectors[task], target[task]) for task in TASKS}
        behavior = joint.behavior_report(backend, fresh["rows"], base_state, donor_state,
                                         atlasrun.states(torch, backend, output, fresh["rows"]))
        control = klfit.control_metrics(backend, control_ctx,
                                        das.head_logits(backend, atlasrun.states(torch, backend, control_output, fresh["controls"])).float())
        reports[arm] = {"response": response, "behavior_signed_projection": behavior, "control": control}
        finite += list(behavior.values()) + [value for task in TASKS for value in response[task].values()]
        finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]

    functional_ranks = [rank for rank in RANKS if sourcev1.functional(reports[f"rank{rank}"])]
    selected_rank = min(functional_ranks) if functional_ranks else None
    atomic_a = {
        "authority_hashes": observed == EXPECTED,
        "source_v2_valid_null": source_v2["terminal"] == "source_subspace_transfer_null" and all(source_v2["atomic_instrument"].values()),
        "support_exact": list(SUPPORT) == ["MLP0", "MLP1", "MLP2", "MLP3", "MLP6"],
        "full_arm_functional": sourcev1.functional(reports["full"]),
        "published_measurements_finite": all(math.isfinite(float(value)) for value in finite)
            and all(math.isfinite(value) for values in spectra.values() for value in values),
        "rank_availability": min(position_counts.values()) >= max(RANKS),
    }
    pa = all(atomic_a.values())
    pb = selected_rank is not None
    pc = pb and reports[f"rank{selected_rank}"]["control"]["median_kl"] <= 0.02 and reports[f"rank{selected_rank}"]["control"]["top1_flip_fraction"] == 0.0
    pd = pb and selected_rank <= 64
    complement = reports["rank64_complement"]
    pe = all(abs(complement["response"][task]["signed_projection"]) <= 0.25
             and abs(complement["behavior_signed_projection"][task]) <= 0.25 for task in TASKS)
    predictions = {"pred_a_atomic_authority_split_full_replay_and_finiteness": bool(pa),
                   "pred_b_a_position_svd_rank_is_functional": bool(pb), "pred_c_selected_rank_is_selective": bool(pc),
                   "pred_d_selected_rank_is_at_most_64": bool(pd), "pred_e_rank64_complement_is_insufficient": bool(pe)}
    terminal = "invalid" if not pa else "selective_position_source_program" if all(predictions.values()) else "position_source_rank_boundary"
    result = {"schema": "temporal_iswas_five_mlp_position_svd_ladder_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter() - tic, "authority_sha256": EXPECTED, "atomic_instrument": atomic_a,
              "support": list(SUPPORT), "position_counts": position_counts, "fit_energy": fit_energy, "spectra": spectra,
              "reports": reports, "functional_ranks": functional_ranks, "selected_rank": selected_rank,
              "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards_max": MAX_FORWARDS, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("atomic_instrument", "position_counts", "fit_energy", "reports", "functional_ranks", "selected_rank", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
