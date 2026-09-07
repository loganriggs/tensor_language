#!/usr/bin/env python3
"""DIM and empirical low-rank source-output patches at the five-MLP boundary."""
# BQGATE: EXPERIMENT pred_a_authority_split_full_replay_finiteness_and_price pred_b_dim2_is_functional pred_c_dim2_is_selective pred_d_dim2_complement_is_insufficient pred_e_dim2_is_lowest_kl_functional_projected_arm
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

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_five_mlp_source_dim_subspace_v1.json"
GREEDY = ROOT / "circuits/followups/temporal_iswas_rank46_shared7_adaptive_greedy_v1_result.json"
EDGE_IMPL = ROOT / "ops/run_temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_five_mlp_source_dim_subspace_v1_result.json"
EXPECTED = {
    "prior": "7e2aaa9b07db8936f3010fadbf02656a2eae285f6818c0a5932992d30a23b5ad",
    "greedy": "43ae9fc88e3ac64058e92fcedc95555fed31dd75bb5d546590b4a998316f9183",
    "edge_impl": "341fcf3d9b0fe1df029f2da6a3ddcb90e57cfc46ea21893a085c1883dfa095b",
}
SUPPORT = ("MLP0", "MLP1", "MLP2", "MLP3", "MLP6")
TASKS = ("temporal", "iswas")
ARMS = ("full", "dim1", "dim2", "svd4", "svd8", "dim2_complement")
MAX_FORWARDS = 32


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def fit_source_bases(backend, rows):
    torch = backend.torch
    batch, _bo, base, _do, donor, _bi, _di = interface.cap_inputs(backend, rows)
    bases, spectra, energy = {}, {}, {}
    for site in SUPPORT:
        _kind, layer, _head = atlasrun.site_parts(site)
        per_row = []
        by_task = {task: [] for task in TASKS}
        for index, row in enumerate(rows):
            stop = int(batch.semantic_positions[index]) + 1
            vector = (donor["mlp"][layer][index, :stop] - base["mlp"][layer][index, :stop]).float().mean(0).to(backend.device)
            per_row.append(vector)
            by_task[klfit.task_name(row)].append(vector)
        matrix = torch.stack(per_row)
        pooled_mean = matrix.mean(0)
        dim1 = pooled_mean[:, None] / pooled_mean.norm().clamp_min(1e-30)
        task_means = torch.stack([torch.stack(by_task[task]).mean(0) for task in TASKS], dim=1)
        dim2 = torch.linalg.qr(task_means, mode="reduced").Q[:, :2].contiguous()
        _u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        bases[site] = {"dim1": dim1, "dim2": dim2, "svd4": vh[:4].T.contiguous(), "svd8": vh[:8].T.contiguous()}
        spectra[site] = [float(value) for value in singular[:12]]
        denominator = matrix.square().sum().clamp_min(1e-30)
        energy[site] = {
            name: float(((matrix @ q) @ q.T).square().sum() / denominator)
            for name, q in bases[site].items()
        }
    return bases, spectra, energy


def capture_projected(backend, batch, base_cache, donor_cache, bases, arm, *, complement=False):
    handles = []
    response = {"attention": {}, "mlp": {}, "mlp_input": {}}
    width = int(backend.model.config.n_embd // backend.model.config.n_head)
    for site in SUPPORT:
        _kind, layer, _head = atlasrun.site_parts(site)
        base = base_cache["mlp"][layer]
        donor = donor_cache["mlp"][layer]
        q = bases[site][arm]
        def patch(_module, _arguments, output, base=base, donor=donor, q=q, complement=complement):
            changed = output.clone()
            for index, position in enumerate(batch.semantic_positions):
                stop = int(position) + 1
                delta = (donor[index, :stop] - base[index, :stop]).to(changed).float()
                projected = (delta @ q) @ q.T
                if complement:
                    projected = delta - projected
                changed[index, :stop] = base[index, :stop].to(changed) + projected.to(changed)
            return changed
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(patch))
    for site in edgeimpl.RESPONSE_SITES:
        kind, layer, _head = atlasrun.site_parts(site)
        if kind == "attn":
            def save_attention(_module, arguments, layer=layer):
                response["attention"][layer] = arguments[0].detach().float().clone()
            handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(save_attention))
        else:
            def save_input(_module, arguments, layer=layer):
                response["mlp_input"][layer] = arguments[0].detach().float().clone()
            def save_mlp(_module, _arguments, output, layer=layer):
                response["mlp"][layer] = output.detach().float().clone()
            handles.append(backend.model.transformer.h[layer].mlp.register_forward_pre_hook(save_input))
            handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(save_mlp))
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(response["attention"]) != {8, 9, 11} or set(response["mlp"]) != {1, 3, 4, 6}:
        raise RuntimeError("incomplete projected-source response capture")
    return output, response


def functional(report):
    return all(
        report["response"][task]["signed_projection"] >= 0.8
        and report["response"][task]["relative_squared_error"] <= 0.2
        and report["behavior_signed_projection"][task] >= 0.75
        for task in TASKS
    )


def main():
    observed = {"prior": sha(PRIOR), "greedy": sha(GREEDY), "edge_impl": sha(EDGE_IMPL)}
    authority = json.loads(GREEDY.read_text())
    dry = {
        "candidate_id": "temporal_auxiliary.iswas_five_mlp_source_dim_subspace_v1",
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "support": list(SUPPORT),
        "arms": list(ARMS),
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
    source_bases, spectra, fit_energy = fit_source_bases(backend, fitted["target_rows"])
    fresh = oodctx.capture(backend, factorial.TCAP, factorial.ICAP)
    _unused, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"])
    _cunused, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    base_output, base = edgeimpl.capture_with_source_patch(backend, fresh["batch"], fresh["base_full"], ())
    donor_output, donor = edgeimpl.capture_with_source_patch(backend, fresh["donor_batch"], donor_full, ())
    base_state = atlasrun.states(torch, backend, base_output, fresh["rows"])
    donor_state = atlasrun.states(torch, backend, donor_output, fresh["rows"])
    target = pairimpl.response_vectors(backend, fresh, base, donor, qs, modes)
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, fresh["control_base_state"]).float()}
    reports = {}
    finite = []
    for arm in ARMS:
        if arm == "full":
            output, changed = edgeimpl.capture_with_source_patch(backend, fresh["batch"], donor_full, SUPPORT)
            control_output, _ = edgeimpl.capture_with_source_patch(backend, fresh["control_batch"], control_donor_full, SUPPORT)
        else:
            basis_arm = "dim2" if arm == "dim2_complement" else arm
            output, changed = capture_projected(
                backend, fresh["batch"], fresh["base_full"], donor_full, source_bases, basis_arm,
                complement=arm == "dim2_complement",
            )
            control_output, _ = capture_projected(
                backend, fresh["control_batch"], fresh["control_base_full"], control_donor_full, source_bases, basis_arm,
                complement=arm == "dim2_complement",
            )
        vectors = pairimpl.response_vectors(backend, fresh, base, changed, qs, modes)
        response = {task: pairimpl.vector_metrics(torch, vectors[task], target[task]) for task in TASKS}
        behavior = joint.behavior_report(
            backend, fresh["rows"], base_state, donor_state,
            atlasrun.states(torch, backend, output, fresh["rows"]),
        )
        control = klfit.control_metrics(
            backend, control_ctx,
            das.head_logits(backend, atlasrun.states(torch, backend, control_output, fresh["controls"])).float(),
        )
        reports[arm] = {"response": response, "behavior_signed_projection": behavior, "control": control}
        finite += list(behavior.values()) + [value for task in TASKS for value in response[task].values()]
        finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]

    fit_ids = {row["row_id"] for rows in task_rows.values() for row in rows} | {row["row_id"] for row in fitted["control_rows"]}
    eval_ids = {row["row_id"] for row in fresh["rows"] + fresh["controls"]}
    pa = (
        observed == EXPECTED and tuple(authority["terminal_support"]) == SUPPORT and not (fit_ids & eval_ids)
        and functional(reports["full"]) and all(math.isfinite(float(value)) for value in finite)
        and all(math.isfinite(value) for values in spectra.values() for value in values)
    )
    pb = functional(reports["dim2"])
    pc = pb and reports["dim2"]["control"]["median_kl"] <= 0.02 and reports["dim2"]["control"]["top1_flip_fraction"] == 0.0
    pd = all(
        abs(reports["dim2_complement"]["response"][task]["signed_projection"]) <= 0.25
        and abs(reports["dim2_complement"]["behavior_signed_projection"][task]) <= 0.25
        for task in TASKS
    )
    projected = ("dim1", "dim2", "svd4", "svd8")
    functional_projected = [arm for arm in projected if functional(reports[arm])]
    pe = pb and functional_projected and reports["dim2"]["control"]["median_kl"] <= min(
        reports[arm]["control"]["median_kl"] for arm in functional_projected
    ) + 1e-8
    predictions = {
        "pred_a_authority_split_full_replay_finiteness_and_price": bool(pa),
        "pred_b_dim2_is_functional": bool(pb),
        "pred_c_dim2_is_selective": bool(pc),
        "pred_d_dim2_complement_is_insufficient": bool(pd),
        "pred_e_dim2_is_lowest_kl_functional_projected_arm": bool(pe),
    }
    terminal = "invalid" if not pa else "selective_dim_source_program" if all(predictions.values()) else "source_subspace_transfer_or_selectivity_boundary"
    result = {
        "schema": "temporal_iswas_five_mlp_source_dim_subspace_result_v1",
        "started_utc": started,
        "finished_utc": now(),
        "serial_seconds": time.perf_counter() - tic,
        "authority_sha256": EXPECTED,
        "support": list(SUPPORT),
        "source_basis_fit_energy": fit_energy,
        "source_basis_spectra": spectra,
        "reports": reports,
        "functional_projected_arms": functional_projected,
        "predictions": predictions,
        "terminal": terminal,
        "price": {"model_forwards_max": MAX_FORWARDS, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("source_basis_fit_energy", "reports", "functional_projected_arms", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
