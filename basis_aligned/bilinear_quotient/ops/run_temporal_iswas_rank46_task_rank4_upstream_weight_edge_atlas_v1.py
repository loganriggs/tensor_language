#!/usr/bin/env python3
"""Full native-component patch atlas into frozen task-rank-four response modes."""
# BQGATE: EXPERIMENT pred_a_authority_alignment_self_patch_finiteness_and_price pred_b_sparse_causal_incidence pred_c_task_typed_edge_tensor pred_d_promoted_edges_close_through_exact_weights pred_e_additive_atlas_replays_response_order
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import pooled_response_projector as pooled
import run_temporal_iswas_rank46_task_mode_complete_rank_ladder_v1 as ladderrun
import run_temporal_iswas_rank46_task_typed_mode_causal_factorial_v1 as factorial
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v1.json"
RANK46 = ROOT / "circuits/followups/temporal_five_mlp_rank47_pooled_greedy_rank46_deletion_v1_result.json"
CORRECTION = ROOT / "circuits/followups/temporal_five_mlp_rank47_pooled_greedy_rank46_deletion_v1_count_correction.json"
LADDER = ROOT / "circuits/followups/temporal_iswas_rank46_task_mode_complete_rank_ladder_v1_result.json"
LADDER_RUNNER = ROOT / "ops/run_temporal_iswas_rank46_task_mode_complete_rank_ladder_v1.py"
FACTORIAL_RUNNER = ROOT / "ops/run_temporal_iswas_rank46_task_typed_mode_causal_factorial_v1.py"
ATLAS_HELPER = ROOT / "ops/run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v1_result.json"
EXPECTED = {
    "prior": "5dd54a7c3817bd9380ed4d4cab3d49be46edf0d9b9b4cb185a9dba75e7b7197f",
    "rank46": "dc8b66d826acede98bde996babe42420dd9e805981f6b81de458d568f29eea1d",
    "correction": "fc1b3ac442e93a841131996bd73f826a18279c80947089deac4ed7060a988c50",
    "ladder": "345ecd0542890c69181efe067729560375543c694f2183319ff824300028e6d8",
    "ladder_runner": "291dc68eddf827bf97c1fdcae01b43a2b4bd2df953acbbb9efe08caff8867473",
    "factorial_runner": "aab635003748cf2164c8b6b467c10b35e4214ec7cb99dac451e78170b664f940",
    "atlas_helper": "6e28d38ec1446eafb3518c1bfe603a5e3469ceadb6f80266e2c695a274692366",
}
RESPONSE_SITES = ("MLP1", "MLP3", "MLP4", "MLP6", "L8H1", "L9H1", "L9H4", "L11H3")
LIVE_EDGE = .05
MAX_FORWARDS = 96


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def order(site):
    kind, layer, _head = atlasrun.site_parts(site)
    return layer, 0 if kind == "attn" else 1


def compatible(source, target):
    return order(source) < order(target)


def capture_with_source_patch(backend, batch, source_cache, sites):
    """Patch whole source heads/modules, then capture all eight live response outputs."""
    by_attention = {}; mlps = set(); handles = []; response = {"attention": {}, "mlp": {}, "mlp_input": {}}
    width = int(backend.model.config.n_embd // backend.model.config.n_head)
    for site in sites:
        kind, layer, head = atlasrun.site_parts(site)
        if kind == "attn": by_attention.setdefault(layer, []).append(head)
        else: mlps.add(layer)
    # Source hooks are deliberately registered before observation hooks.
    for layer, heads in by_attention.items():
        source = source_cache["attention"][layer]
        def patch_attention(_module, arguments, heads=tuple(heads), source=source):
            changed = arguments[0].clone()
            for i, pos in enumerate(batch.semantic_positions):
                stop = int(pos) + 1
                for head in heads:
                    a, z = head * width, (head + 1) * width
                    changed[i, :stop, a:z] = source[i, :stop, a:z].to(changed)
            return (changed,) + tuple(arguments[1:])
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch_attention))
    for layer in mlps:
        source = source_cache["mlp"][layer]
        def patch_mlp(_module, _arguments, output, source=source):
            changed = output.clone()
            for i, pos in enumerate(batch.semantic_positions):
                stop = int(pos) + 1; changed[i, :stop] = source[i, :stop].to(changed)
            return changed
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(patch_mlp))
    for site in RESPONSE_SITES:
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
    try: output = backend.native(batch, capture=True)
    finally:
        for handle in handles: handle.remove()
    if (set(response["attention"]) != {8, 9, 11} or set(response["mlp"]) != {1, 3, 4, 6}
            or set(response["mlp_input"]) != {1, 3, 4, 6}):
        raise RuntimeError("incomplete response capture")
    return output, response


def raw_response(cache, site, width):
    kind, layer, head = atlasrun.site_parts(site)
    if kind == "mlp": return cache["mlp"][layer]
    return cache["attention"][layer][..., head * width:(head + 1) * width]


def valid_rows(torch, value, batch, ids):
    return torch.cat([value[i, :int(batch.semantic_positions[i])+1] for i in ids])


def exact_mlp_delta(module, base_input, changed_input):
    with __import__('torch').no_grad():
        b = module.Left(base_input).float() * module.Right(base_input).float()
        c = module.Left(changed_input).float() * module.Right(changed_input).float()
        return (c - b) @ module.Down.weight.detach().float().T


def main():
    paths = {"prior": PRIOR, "rank46": RANK46, "correction": CORRECTION, "ladder": LADDER,
             "ladder_runner": LADDER_RUNNER, "factorial_runner": FACTORIAL_RUNNER, "atlas_helper": ATLAS_HELPER}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"weight-edge atlas authority changed: {observed}")
    support = json.loads(RANK46.read_text())["selected_support"]
    dry = {"candidate_id": "temporal_auxiliary.iswas_rank46_task_rank4_upstream_weight_edge_atlas_v1",
           "dryrun": True, "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "source_count": len(support), "response_sites": RESPONSE_SITES, "live_edge_cutoff": LIVE_EDGE,
           "model_forwards_max": MAX_FORWARDS, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    fitted = pooled.fit(backend); qs = fitted["projector"]; task_rows, full_modes = ladderrun.fit_full_modes(backend, fitted)
    modes = {site: {task: full_modes[site][task][:, :4] for task in ("temporal", "iswas")} for site in RESPONSE_SITES}
    fresh = oodctx.capture(backend, factorial.TCAP, factorial.ICAP); batch = fresh["batch"]
    _unused, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"])
    _base_output, base = capture_with_source_patch(backend, batch, fresh["base_full"], ())
    _donor_output, donor = capture_with_source_patch(backend, fresh["donor_batch"], donor_full, ())
    _self_output, self_cache = capture_with_source_patch(backend, batch, fresh["base_full"], support)
    width = int(backend.model.config.n_embd // backend.model.config.n_head)
    ids = {task: [i for i, row in enumerate(fresh["rows"]) if klfit.task_name(row) == task] for task in ("temporal", "iswas")}
    self_error = 0.0
    for site in RESPONSE_SITES:
        self_error = max(self_error, float((raw_response(self_cache, site, width)-raw_response(base, site, width)).abs().max()))
    reports = {}; finite = [self_error]; closure_values = []; promoted_closure = []; task_edges = {task: set() for task in ids}
    additive = {task: {site: None for site in RESPONSE_SITES} for task in ids}
    for source_site in support:
        _output, changed = capture_with_source_patch(backend, batch, donor_full, (source_site,))
        reports[source_site] = {}
        for target in RESPONSE_SITES:
            if not compatible(source_site, target): continue
            reports[source_site][target] = {}
            actual_all = raw_response(changed, target, width) - raw_response(base, target, width)
            ref_all = raw_response(donor, target, width) - raw_response(base, target, width)
            kind, layer, _head = atlasrun.site_parts(target)
            if kind == "mlp":
                module = backend.model.transformer.h[layer].mlp
                predicted_all = exact_mlp_delta(module, base["mlp_input"][layer], changed["mlp_input"][layer])
            for task, task_ids in ids.items():
                rb = qs[target] @ modes[target][task]; other = "iswas" if task == "temporal" else "temporal"
                ob = qs[target] @ modes[target][other]
                actual = valid_rows(torch, actual_all, batch, task_ids); reference = valid_rows(torch, ref_all, batch, task_ids)
                own, ref = (actual @ rb) @ rb.T, (reference @ rb) @ rb.T
                cross = (actual @ ob) @ ob.T
                den, anorm = ref.square().sum().clamp_min(1e-30), actual.square().sum().clamp_min(1e-30)
                signed = float((own * ref).sum() / den); own_energy = float(own.square().sum() / anorm); cross_energy = float(cross.square().sum() / anorm)
                if kind == "mlp": predicted = (valid_rows(torch, predicted_all, batch, task_ids) @ rb) @ rb.T
                else:
                    wo = backend.model.transformer.h[layer].attn.c_proj.weight.detach().float()[:, atlasrun.site_parts(target)[2]*width:(atlasrun.site_parts(target)[2]+1)*width]
                    predicted = (actual @ rb) @ (wo @ rb).T; own = own @ wo.T
                closure = float((predicted-own).square().sum() / own.square().sum().clamp_min(1e-30))
                edge = {"signed_target_response": signed, "own_coordinate_energy": own_energy,
                        "cross_coordinate_energy": cross_energy, "own_cross_energy_ratio": own_energy / max(1e-30, cross_energy),
                        "exact_weight_closure_rse": closure}
                reports[source_site][target][task] = edge; finite += list(edge.values()); closure_values.append(closure)
                if abs(signed) >= LIVE_EDGE: task_edges[task].add(f"{source_site}->{target}"); promoted_closure.append(closure)
                contribution = valid_rows(torch, actual_all, batch, task_ids) @ rb
                additive[task][target] = contribution if additive[task][target] is None else additive[task][target] + contribution
    task_summary = {}; typed_sites = []
    for task in ids:
        positives = sorted((max(0.0, edge[task]["signed_target_response"])
                            for source in reports.values() for edge in source.values() if task in edge), reverse=True)
        qn = max(1, math.ceil(len(positives)/4)); concentration = sum(positives[:qn])/max(1e-30, sum(positives))
        maximum = max(positives) if positives else 0.0
        task_summary[task] = {"max_signed_target_response": maximum, "top_quartile_positive_incidence_fraction": concentration,
                              "live_edge_count": len(task_edges[task])}
    for target in RESPONSE_SITES:
        own_t = sum(abs(reports[s][target]["temporal"]["signed_target_response"]) for s in reports if target in reports[s])
        cross_t = sum(abs(reports[s][target]["iswas"]["signed_target_response"]) for s in reports if target in reports[s])
        if max(own_t, cross_t) / max(1e-30, min(own_t, cross_t)) >= 2: typed_sites.append(target)
    union, inter = task_edges["temporal"] | task_edges["iswas"], task_edges["temporal"] & task_edges["iswas"]
    edge_jaccard = len(inter)/max(1, len(union))
    additive_report = {}
    for task, task_ids in ids.items():
        site_projection = {}; combined_num = combined_den = 0.0
        for target in RESPONSE_SITES:
            rb = qs[target] @ modes[target][task]
            reference = valid_rows(torch, raw_response(donor, target, width)-raw_response(base, target, width), batch, task_ids) @ rb
            changed = additive[task][target]
            den = float(reference.square().sum()); num = float((changed*reference).sum())
            site_projection[target] = num/max(1e-30, den); combined_num += num; combined_den += den
        additive_report[task] = {"signed_projection": combined_num/max(1e-30, combined_den), "site_projection": site_projection}
        finite += [additive_report[task]["signed_projection"], *site_projection.values()]
    fit_ids = {row["row_id"] for rows in task_rows.values() for row in rows} | {row["row_id"] for row in fitted["control_rows"]}
    eval_ids = {row["row_id"] for row in fresh["rows"]}
    pa = (len(support) == 46 and json.loads(LADDER.read_text())["selected_rank"]["own"] == 4
          and not (fit_ids & eval_ids) and self_error <= 1e-4 and all(math.isfinite(float(v)) for v in finite))
    pb = all(x["max_signed_target_response"] >= .20 and x["top_quartile_positive_incidence_fraction"] >= .70 for x in task_summary.values())
    pc = len(typed_sites) >= 2 and edge_jaccard < .8
    pd = bool(promoted_closure) and sum(x <= .15 for x in promoted_closure)/len(promoted_closure) >= .8
    pe = all(report["signed_projection"] >= .8 for report in additive_report.values())
    predictions = {"pred_a_authority_alignment_self_patch_finiteness_and_price": bool(pa),
        "pred_b_sparse_causal_incidence": bool(pb), "pred_c_task_typed_edge_tensor": bool(pc),
        "pred_d_promoted_edges_close_through_exact_weights": bool(pd), "pred_e_additive_atlas_replays_response_order": bool(pe)}
    terminal = "invalid" if not pa else "task_typed_weight_edge_atlas" if all(predictions.values()) else "distributed_or_contextual_weight_edge_atlas"
    result = {"schema": "temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_result_v1",
        "started_utc": started, "finished_utc": now(), "serial_seconds": time.perf_counter()-tic,
        "authority_sha256": EXPECTED, "support": support, "response_sites": RESPONSE_SITES,
        "live_edge_cutoff": LIVE_EDGE, "task_summary": task_summary, "task_typed_sites": typed_sites,
        "edge_jaccard": edge_jaccard, "additive_report": additive_report, "site_reports": reports,
        "self_patch_max_abs": self_error, "promoted_edge_count": len(promoted_closure),
        "promoted_weight_closure_fraction": sum(x <= .15 for x in promoted_closure)/max(1, len(promoted_closure)),
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards_max": MAX_FORWARDS, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("task_summary", "task_typed_sites", "edge_jaccard", "additive_report", "self_patch_max_abs", "promoted_edge_count", "promoted_weight_closure_fraction", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
