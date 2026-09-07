#!/usr/bin/env python3
"""Causal shared/contrast weight groups inside the five rank-16 source interfaces."""
# BQGATE: EXPERIMENT pred_a_authority_replay_closure_finiteness_and_price pred_b_task_pair_blocks_are_crossfit_stable pred_c_own_task_beats_cross_task pred_d_mean_plus_contrast_is_functional_and_full_equivalent pred_e_union_complement_is_insufficient pred_f_weight_grouping_adds_no_parent_collateral
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import pooled_response_projector as pooled
import task_mode_hidden_groups as groups
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlas
import run_temporal_iswas_five_mlp_position_svd_ladder_v1 as svd
import run_temporal_iswas_five_mlp_rank16_hidden_weight_compiler_v1 as compiler
import run_temporal_iswas_five_mlp_source_dim_subspace_v1 as source
import run_temporal_iswas_rank16_activation_conditioned_hidden_groups_v1 as hidden
import run_temporal_iswas_rank46_shared8_pairwise_mobius_v1 as pair
import run_temporal_iswas_rank46_task_mode_complete_rank_ladder_v1 as ladder
import run_temporal_iswas_rank46_task_rank4_joint_source_factorial_v1 as joint
import run_temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v1 as edge
import run_temporal_iswas_rank46_task_typed_mode_causal_factorial_v1 as factorial

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank16_source_task_pair_weight_groups_v1.json"
AUDIT = ROOT / "circuits/followups/temporal_iswas_five_mlp_rank16_hidden_weight_compiler_v3_audit_result.json"
HOLDOUT = ROOT / "circuits/followups/temporal_iswas_rank16_hidden_mask30_construction_holdout_v1_result.json"
ATTRIBUTION = ROOT / "circuits/followups/temporal_iswas_rank16_hidden_mask30_control_attribution_v1_result.json"
HELPER = ROOT / "ops/task_mode_hidden_groups.py"
MATH_REVIEW = ROOT.parent / "polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-07_1426.md"
OUT = ROOT / "circuits/followups/temporal_iswas_rank16_source_task_pair_weight_groups_v1_result.json"
EXPECTED = {
    "prior": "cc21b54414896fdc246aea35be84eef42917ae9e30fdf91104c394bba530af12",
    "weight_compiler_audit": "0ef59ddb33f3d0534e5ecc283844189683eb73b9c2f8ba40f7a7af91f4a14d0b",
    "mask30_holdout": "346402de851dd9d600fd767532a9b0238d90296fa3ed6305344af263bfabe63e",
    "mask30_control_attribution": "7c25c516398c4daa97522c19bfc5b27ff663ce73f225f7b9e1061f9571f82ebf",
    "task_mode_helper": "f0f448113b4862f7b6451b01bd3f9c5df24d211a83a3996bb0e72833457d94e4",
    "mathematical_review": "811d43e42c6257fb7546a11e9626c5145a2f126c8990de9c3dd5fc2ea7f9dd01",
}
SUPPORT = tuple(source.SUPPORT)
TASKS = tuple(source.TASKS)
ARMS = ("full_rank16", "shared_mean", "task_contrast", "mean_plus_contrast", "own_task", "cross_task", "mean_plus_contrast_complement")
GAIN = 1.15
MAX_FORWARDS = 48


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def row_task(row, *, control=False):
    return "temporal" if control else klfit.task_name(row)


def z_matrix(torch, batch, delta_hidden, rows, task, weight_map, parity=None):
    indices = [index for index, row in enumerate(rows) if klfit.task_name(row) == task]
    if parity is not None:
        indices = indices[parity::2]
    pieces = []
    for index in indices:
        stop = int(batch.semantic_positions[index]) + 1
        pieces.append(delta_hidden[index, :stop].float() @ weight_map)
    return torch.cat(pieces, dim=0)


def task_basis(torch, matrix, rank=4):
    _u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    basis = vh[:rank].T.contiguous()
    gap = float(singular[rank - 1] / singular[rank].clamp_min(1e-30)) if len(singular) > rank else math.inf
    return basis, [float(value) for value in singular[:8]], gap


def min_subspace_cosine(torch, left, right):
    if left.shape[1] != right.shape[1] or left.shape[1] == 0:
        return 0.0
    return float(torch.linalg.svdvals(left.T @ right).min())


def effective_width(energy):
    return float(energy.sum().square() / energy.square().sum().clamp_min(1e-30))


def top_fraction(energy, fraction):
    count = max(1, int(math.ceil(len(energy) * fraction)))
    return float(torch_topk(energy, count).sum() / energy.sum().clamp_min(1e-30))


def torch_topk(value, count):
    return value.topk(count).values


def block_for(blocks, site, arm, task):
    entry = blocks[site]
    if arm == "full_rank16":
        return entry["full"]
    if arm == "shared_mean":
        return entry["mean"]
    if arm == "task_contrast":
        return entry["contrast"]
    if arm == "mean_plus_contrast":
        return entry["union"]
    if arm == "mean_plus_contrast_complement":
        return entry["complement"]
    if arm == "own_task":
        return entry[task]
    other = "iswas" if task == "temporal" else "temporal"
    return entry[other]


def run_blocks(backend, batch, rows, base_out, delta_hidden, bases, maps, blocks, arm, *, control=False):
    handles = []
    response = {"attention": {}, "mlp": {}, "mlp_input": {}}
    for site in SUPPORT:
        layer = int(site[3:])
        base_value = base_out["mlp"][layer]
        delta = delta_hidden[site]
        q = bases[site]["rank16"]
        weight_map = maps[site]

        def patch(_module, _arguments, output, site=site, base_value=base_value, delta=delta, q=q, weight_map=weight_map):
            changed = output.clone()
            for index, row in enumerate(rows):
                stop = int(batch.semantic_positions[index]) + 1
                block = block_for(blocks, site, arm, row_task(row, control=control)).to(changed)
                projected = (delta[index, :stop].float() @ weight_map @ block) @ (q @ block).T
                changed[index, :stop] = base_value[index, :stop].to(changed) + (GAIN * projected).to(changed)
            return changed

        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(patch))
    for response_site in edge.RESPONSE_SITES:
        kind, layer, _head = atlas.site_parts(response_site)
        if kind == "attn":
            handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
                lambda _module, arguments, layer=layer: response["attention"].__setitem__(layer, arguments[0].detach().float().clone())))
        else:
            handles.append(backend.model.transformer.h[layer].mlp.register_forward_pre_hook(
                lambda _module, arguments, layer=layer: response["mlp_input"].__setitem__(layer, arguments[0].detach().float().clone())))
            handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(
                lambda _module, _arguments, output, layer=layer: response["mlp"].__setitem__(layer, output.detach().float().clone())))
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()
    return output, response


def main():
    candidate_id = "temporal_auxiliary.iswas_rank16_source_task_pair_weight_groups_v1"
    dry = {
        "candidate_id": candidate_id, "dryrun": True, "gpu_accessed": False, "model_loaded": False,
        "queue_touched": False, "support": SUPPORT, "arms": ARMS, "model_forwards_max": MAX_FORWARDS,
        "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {
        "prior": PRIOR, "weight_compiler_audit": AUDIT, "mask30_holdout": HOLDOUT,
        "mask30_control_attribution": ATTRIBUTION, "task_mode_helper": HELPER, "mathematical_review": MATH_REVIEW,
    }
    observed = {key: sha(path) for key, path in paths.items()}
    started, tic = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    native = backend.native
    forward_count = 0

    def counted(*args, **kwargs):
        nonlocal forward_count
        forward_count += 1
        return native(*args, **kwargs)

    backend.native = counted
    fitted = pooled.fit(backend)
    response_qs = fitted["projector"]
    _task_rows, full_modes = ladder.fit_full_modes(backend, fitted)
    response_modes = {site: {task: full_modes[site][task][:, :4] for task in TASKS} for site in edge.RESPONSE_SITES}
    bases, _spectra, _energy, _counts = svd.fit_position_bases(backend, fitted["target_rows"])
    fit_batch, *_ = interface.cap_inputs(backend, fitted["target_rows"])
    fit_hidden0, _ = compiler.capture(backend, fit_batch)
    fit_donor_batch = das._batch(backend, fitted["target_rows"], side="donor")
    fit_hidden1, _ = compiler.capture(backend, fit_donor_batch)

    maps, blocks, diagnostics = {}, {}, {}
    closure = []
    for site in SUPPORT:
        layer = int(site[3:])
        q = bases[site]["rank16"]
        weight_map = backend.model.transformer.h[layer].mlp.Down.weight.detach().float().T @ q
        maps[site] = weight_map
        delta = fit_hidden1[site] - fit_hidden0[site]
        task_bases, task_spectra, task_gaps = {}, {}, {}
        half_blocks = []
        for parity in (0, 1):
            parity_bases = {}
            for task in TASKS:
                parity_bases[task], _spectrum, _gap = task_basis(torch, z_matrix(torch, fit_batch, delta, fitted["target_rows"], task, weight_map, parity))
            half_blocks.append(groups.canonical_pair_directions(torch, parity_bases["temporal"], parity_bases["iswas"]))
        for task in TASKS:
            task_bases[task], task_spectra[task], task_gaps[task] = task_basis(torch, z_matrix(torch, fit_batch, delta, fitted["target_rows"], task, weight_map))
        pair_blocks = groups.canonical_pair_directions(torch, task_bases["temporal"], task_bases["iswas"])
        union_raw = torch.cat((pair_blocks["mean"], pair_blocks["contrast"]), dim=1)
        union = torch.linalg.qr(union_raw, mode="reduced").Q
        complete = torch.linalg.qr(union, mode="complete").Q
        complement = complete[:, union.shape[1]:]
        blocks[site] = {
            "full": torch.eye(16, device=backend.device), "mean": pair_blocks["mean"],
            "contrast": pair_blocks["contrast"], "union": union, "complement": complement,
            **task_bases,
        }
        flat_delta = hidden.flatten(fit_batch, delta)
        participation = groups.participation_energy(torch, flat_delta, weight_map, pair_blocks)
        diagnostics[site] = {
            "principal_cosines": [float(value) for value in pair_blocks["principal_cosines"]],
            "task_spectra": task_spectra, "task_rank4_gap": task_gaps,
            "crossfit_min_cosine": {
                "mean": min_subspace_cosine(torch, half_blocks[0]["mean"], half_blocks[1]["mean"]),
                "contrast": min_subspace_cosine(torch, half_blocks[0]["contrast"], half_blocks[1]["contrast"]),
            },
            "participation": {
                name: {"effective_hidden_width": effective_width(value), "top10pct_energy_fraction": top_fraction(value, .10)}
                for name, value in participation.items()
            },
            "block_ranks": {name: int(blocks[site][name].shape[1]) for name in ("mean", "contrast", "union", "complement")},
        }
        full_delta = (flat_delta @ weight_map) @ q.T
        union_delta = (flat_delta @ weight_map @ union) @ (q @ union).T
        complement_delta = (flat_delta @ weight_map @ complement) @ (q @ complement).T
        closure.append(float((union_delta + complement_delta - full_delta).square().sum() / full_delta.square().sum().clamp_min(1e-30)))

    fresh = oodctx.capture(backend, factorial.TCAP, factorial.ICAP)
    hidden0, _ = compiler.capture(backend, fresh["batch"])
    hidden1, _ = compiler.capture(backend, fresh["donor_batch"])
    control_hidden0, _ = compiler.capture(backend, fresh["control_batch"])
    control_hidden1, _ = compiler.capture(backend, fresh["control_donor_batch"])
    base_output, base_response = edge.capture_with_source_patch(backend, fresh["batch"], fresh["base_full"], ())
    donor_output, donor_response = edge.capture_with_source_patch(backend, fresh["donor_batch"], fresh["base_full"], ())
    base_state = atlas.states(torch, backend, base_output, fresh["rows"])
    donor_state = atlas.states(torch, backend, donor_output, fresh["rows"])
    target_response = pair.response_vectors(backend, fresh, base_response, donor_response, response_qs, response_modes)
    control_base_logits = das.head_logits(backend, fresh["control_base_state"]).float()
    control_context = {"rows": fresh["controls"], "base_logits": control_base_logits}
    reports, finite = {}, []
    target_delta = {site: hidden1[site] - hidden0[site] for site in SUPPORT}
    control_delta = {site: control_hidden1[site] - control_hidden0[site] for site in SUPPORT}
    for arm in ARMS:
        output, changed = run_blocks(backend, fresh["batch"], fresh["rows"], fresh["base_full"], target_delta, bases, maps, blocks, arm)
        control_output, _ = run_blocks(backend, fresh["control_batch"], fresh["controls"], fresh["control_base_full"], control_delta, bases, maps, blocks, arm, control=True)
        vectors = pair.response_vectors(backend, fresh, base_response, changed, response_qs, response_modes)
        response = {task: pair.vector_metrics(torch, vectors[task], target_response[task]) for task in TASKS}
        behavior = joint.behavior_report(backend, fresh["rows"], base_state, donor_state, atlas.states(torch, backend, output, fresh["rows"]))
        control_logits = das.head_logits(backend, atlas.states(torch, backend, control_output, fresh["controls"])).float()
        control = klfit.control_metrics(backend, control_context, control_logits)
        flipped = [row["row_id"] for index, row in enumerate(fresh["controls"]) if int(control_base_logits[index].argmax()) != int(control_logits[index].argmax())]
        control["flipped_row_ids"] = flipped
        reports[arm] = {"response": response, "behavior_signed_projection": behavior, "control": control}
        finite += list(behavior.values()) + [value for task in TASKS for value in response[task].values()]
        finite += list(control["margin_rms_fraction"].values()) + [control["median_kl"], control["max_kl"], control["top1_flip_fraction"]]

    holdout = json.loads(HOLDOUT.read_text())
    attribution = json.loads(ATTRIBUTION.read_text())
    full_reference = holdout["reports"]["full"]
    replay = max(
        max(abs(reports["full_rank16"]["response"][task][metric] - full_reference["response"][task][metric]) for task in TASKS for metric in ("signed_projection", "relative_squared_error", "norm_ratio")),
        max(abs(reports["full_rank16"]["behavior_signed_projection"][task] - full_reference["behavior_signed_projection"][task]) for task in TASKS),
        abs(reports["full_rank16"]["control"]["median_kl"] - attribution["reports"]["full"]["summary"]["median_kl"]),
    )
    stable_sites = sum(
        diagnostics[site]["crossfit_min_cosine"]["mean"] >= .70 and diagnostics[site]["crossfit_min_cosine"]["contrast"] >= .70
        for site in SUPPORT
    )
    union = reports["mean_plus_contrast"]
    full = reports["full_rank16"]
    complement = reports["mean_plus_contrast_complement"]
    authority_ok = json.loads(AUDIT.read_text())["terminal"] == "distributed_weight_compiled_source_program" and holdout["terminal"] == "null" and attribution["terminal"] == "parent_limited_selectivity"
    pred_a = observed == EXPECTED and authority_ok and replay <= 1e-5 and max(closure) <= 1e-6 and all(math.isfinite(float(value)) for value in finite) and forward_count <= MAX_FORWARDS
    pred_b = stable_sites >= 4
    pred_c = all(reports["own_task"]["behavior_signed_projection"][task] >= reports["cross_task"]["behavior_signed_projection"][task] + .10 for task in TASKS)
    pred_d = source.functional(union) and all(
        abs(union["response"][task]["signed_projection"] - full["response"][task]["signed_projection"]) <= .03
        and abs(union["behavior_signed_projection"][task] - full["behavior_signed_projection"][task]) <= .03
        for task in TASKS
    )
    pred_e = all(abs(complement["response"][task]["signed_projection"]) <= .50 and abs(complement["behavior_signed_projection"][task]) <= .50 for task in TASKS)
    pred_f = set(union["control"]["flipped_row_ids"]).issubset(full["control"]["flipped_row_ids"]) and union["control"]["median_kl"] <= full["control"]["median_kl"] + .002 and all(union["control"]["margin_rms_fraction"][task] <= full["control"]["margin_rms_fraction"][task] + .02 for task in TASKS)
    predictions = {
        "pred_a_authority_replay_closure_finiteness_and_price": bool(pred_a),
        "pred_b_task_pair_blocks_are_crossfit_stable": bool(pred_b),
        "pred_c_own_task_beats_cross_task": bool(pred_c),
        "pred_d_mean_plus_contrast_is_functional_and_full_equivalent": bool(pred_d),
        "pred_e_union_complement_is_insufficient": bool(pred_e),
        "pred_f_weight_grouping_adds_no_parent_collateral": bool(pred_f),
    }
    terminal = "invalid" if not pred_a else "stable_task_pair_source_weight_circuit" if all(predictions.values()) else "functional_unstable_task_blocks" if pred_d and not pred_b else "shared_source_group_only" if pred_b and pred_d and not pred_c else "probe_basis_only"
    result = {
        "schema": "temporal_iswas_rank16_source_task_pair_weight_groups_result_v1",
        "started_utc": started, "finished_utc": now(), "serial_seconds": time.perf_counter() - tic,
        "authority_sha256": EXPECTED, "diagnostics": diagnostics, "stable_site_count": stable_sites,
        "union_complement_closure_rse_by_site": dict(zip(SUPPORT, closure)), "full_replay_max_abs_error": replay,
        "reports": reports, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards_observed": forward_count, "model_forwards_max": MAX_FORWARDS, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("diagnostics", "stable_site_count", "union_complement_closure_rse_by_site", "full_replay_max_abs_error", "reports", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
