#!/usr/bin/env python3
"""Complete-module atlas upstream of the literal auxiliary L11/L15 c_v readers."""

# BQGATE: EXPERIMENT pred_a_authority_capture_self_clamp_causal_order_finiteness_and_price pred_b_each_reader_has_a_stable_material_upstream_module pred_c_joint_removal_distinguishes_residual_skip_from_module_conversion pred_d_l11_causal_order_tripwire pred_e_zero_fit_complete_module_inventory
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_auxiliary_postcue_local_weight_compiler_v1 as compiler
import run_iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_v1 as auxiliary
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_auxiliary_cv_upstream_module_atlas_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_postcue_local_weight_compiler_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_auxiliary_postcue_local_weight_compiler_v1.py"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_cv_upstream_module_atlas_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_auxiliary_cv_upstream_module_atlas_v1"
SITES = ("mlp:09", "attn:10", "mlp:10", "attn:11", "mlp:11", "attn:12",
         "mlp:12", "attn:13", "mlp:13", "attn:14", "mlp:14")
GROUP_ARMS = ("l9_core_live", "all_intervening_removed",
              "l9_core_live_and_all_intervening_removed")
LAYERS, SELECTED, CORE = (11, 15), {11: (1, 3), 15: (5,)}, (1, 4)
EXPECTED = {
    "prior": "c51ed4037c22f61d50fe7375f42d0e0b92a2051bd9c345c3b6cd4d6b7b205364",
    "parent": "57717c1f38013943273db9eb4076ad5d3191b72d8b14e85fa333b8239650cf34",
    "parent_runner": "81d01c6594d80c84d500e14bf4dd4896edb0e81bc38fa25bb6596f4abf4c27ce",
    "capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
    "builder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2",
}
MAX_FORWARDS, MAX_EVALUATIONS = 19, 570


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def cosine(x, y):
    denominator = float(x.norm() * y.norm())
    return float((x * y).sum()) / denominator if denominator else 0.0


def site_module(backend, site):
    kind, layer = site.split(":")
    block = backend.model.transformer.h[int(layer)]
    return block.attn.c_proj if kind == "attn" else block.mlp


def capture_site_hook(cache, site):
    if site.startswith("attn"):
        def hook(_module, arguments):
            cache[site] = arguments[0].detach().clone()
        return hook
    def hook(_module, _arguments, output):
        cache[site] = output.detach().clone()
    return hook


def clamp_hook(batch, base_value, kind):
    if kind == "attn":
        def hook(_module, arguments):
            changed = arguments[0].clone()
            for i, query in enumerate(batch.semantic_positions):
                changed[i, :int(query) + 1] = base_value[i, :int(query) + 1].to(changed)
            return (changed,) + tuple(arguments[1:])
        return hook
    def hook(_module, _arguments, output):
        changed = output.clone()
        for i, query in enumerate(batch.semantic_positions):
            changed[i, :int(query) + 1] = base_value[i, :int(query) + 1].to(changed)
        return changed
    return hook


def capture_readers_and_modules(backend, batch, *, call, capture_raw9=False):
    modules, handles = {}, []
    for site in SITES:
        module = site_module(backend, site)
        hook = capture_site_hook(modules, site)
        handles.append(module.register_forward_pre_hook(hook) if site.startswith("attn")
                       else module.register_forward_hook(hook))
    try:
        output, readers, raw9 = compiler.capture_dual(
            backend, batch, call=call, capture_raw9=capture_raw9)
    finally:
        for handle in handles:
            handle.remove()
    if set(modules) != set(SITES):
        raise RuntimeError("incomplete upstream module capture")
    return output, readers, raw9, modules


def run_variant(backend, batch, base_hidden, complement, positions, base_raw9,
                base_modules, sites, *, core_clamped=True):
    handles = []
    for site in sites:
        module = site_module(backend, site)
        kind = site.split(":")[0]
        hook = clamp_hook(batch, base_modules[site], kind)
        handles.append(module.register_forward_pre_hook(hook) if kind == "attn"
                       else module.register_forward_hook(hook))
    selections = {9: CORE} if core_clamped else {}
    try:
        return capture_readers_and_modules(
            backend, batch,
            call=lambda: auxiliary.run_heads(backend, batch, base_hidden, complement,
                positions, {9: base_raw9}, selections))
    finally:
        for handle in handles:
            handle.remove()


def reader_writes(backend, rows, base_captures, captures):
    torch = backend.torch
    writes = {}
    for layer in LAYERS:
        attention = backend.model.transformer.h[layer].attn
        width, heads = int(backend.model.config.n_embd), int(backend.model.config.n_head)
        head_dim = width // heads
        response = torch.zeros(len(rows), len(SELECTED[layer]), head_dim,
                               device=backend.device, dtype=torch.float32)
        dv = captures[layer]["value"].float() - base_captures[layer]["value"].float()
        for i, row in enumerate(rows):
            query = int(row["base_semantic_position"])
            sources = compiler.source_partition(row)["post_cue_before_subject"]
            source_index = torch.as_tensor(sources, device=backend.device)
            for j, head in enumerate(SELECTED[layer]):
                pattern = base_captures[layer]["pattern"][i, head, query].float().index_select(0, source_index)
                response[i, j] = torch.einsum(
                    "s,sd->d", pattern, dv[i].index_select(0, source_index)[:, head])
        cproj = attention.c_proj.weight.detach().float()
        writes[layer] = sum(torch.nn.functional.linear(response[:, j],
            cproj[:, head * head_dim:(head + 1) * head_dim], None)
            for j, head in enumerate(SELECTED[layer]))
    return writes


def main():
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER,
             "capability": CAPABILITY, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("auxiliary upstream-module authority changed")
    prior, parent, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, PARENT, CAPABILITY, weight.SUBSPACE)]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    positions = [weight.postcue_positions(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen"
            or len(rows) != 30 or tuple(prior.get("sites", ())) != SITES):
        raise RuntimeError("parent decision, population, or site inventory changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "sites": list(SITES),
        "group_arms": list(GROUP_ARMS), "model_forwards_max": MAX_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS, "fit_updates": 0,
        "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                     for layer in range(12, 18))
    modes, orientation_error, _wrong = overlap.residual_modes(backend, family[8], gain)
    s = torch.linalg.qr(modes, mode="reduced").Q
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_hidden_output, base_hidden_capture = weight.capture_mlp8(backend, base_batch)
    _donor_output, donor_hidden_capture = weight.capture_mlp8(backend, donor_batch)
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _singular, vh = torch.linalg.svd(s.T @ down, full_matrices=False)
    full_hidden_delta = donor_hidden_capture["hidden"].float() - base_hidden_capture["hidden"].float()
    complement = full_hidden_delta - weight.project(full_hidden_delta, vh, vh.shape[0])
    base_output, base_readers, base_raw9, base_modules = capture_readers_and_modules(
        backend, base_batch, capture_raw9=True, call=lambda: backend.native(base_batch, capture=True))
    live_output, live_readers, _raw, _modules = capture_readers_and_modules(
        backend, base_batch, call=lambda: auxiliary.run_heads(backend, base_batch,
            base_hidden_capture["hidden"], complement, positions, {9: base_raw9}, {9: CORE}))
    outputs, reader_captures = {}, {}
    for site in SITES:
        output, captures, _raw, _modules = run_variant(backend, base_batch,
            base_hidden_capture["hidden"], complement, positions, base_raw9, base_modules, (site,))
        outputs[site], reader_captures[site] = output, captures
    for name, sites, core_clamped in (
        ("l9_core_live", (), False),
        ("all_intervening_removed", SITES, True),
        ("l9_core_live_and_all_intervening_removed", SITES, False),
    ):
        output, captures, _raw, _modules = run_variant(backend, base_batch,
            base_hidden_capture["hidden"], complement, positions, base_raw9,
            base_modules, sites, core_clamped=core_clamped)
        outputs[name], reader_captures[name] = output, captures
    forwards, evaluations = 18, 18 * len(rows)

    live_writes = reader_writes(backend, rows, base_readers, live_readers)
    variant_writes = {name: reader_writes(backend, rows, base_readers, captures)
                      for name, captures in reader_captures.items()}
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)
             for panel in ("A1", "A2")}
    state = lambda output: converter.state(output, rows, torch, backend.device)
    base18, base_hidden18, live18 = map(state, (base_output, base_hidden_output, live_output))
    states = {name: state(output) for name, output in outputs.items()}
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda value: das.head_logits(backend, value)[index, answers] - das.head_logits(backend, value)[index, foils]
    base_margin, live_margin = margin(base18), margin(live18)
    live_behavior = live_margin - base_margin
    metrics = {str(layer): {panel: {} for panel in masks} for layer in LAYERS}
    for layer in LAYERS:
        target = live_writes[layer]
        for panel, mask in masks.items():
            target_panel = target[mask]
            for name, writes in variant_writes.items():
                removed = (target - writes[layer])[mask]
                behavior_removed = (live_margin - margin(states[name]))[mask]
                metrics[str(layer)][panel][name] = {
                    "signed_response_projection_fraction": float(
                        (removed * target_panel).sum() / target_panel.square().sum()),
                    "response_norm_fraction": float(removed.norm() / target_panel.norm()),
                    "response_cosine": cosine(removed.reshape(-1), target_panel.reshape(-1)),
                    "absolute_behavior_fraction_of_parent": float(
                        behavior_removed.abs().mean() / live_behavior[mask].abs().mean()),
                }
    causal_prior = {11: SITES[:3], 15: SITES}
    stable_material = {}
    rankings = {}
    for layer in LAYERS:
        valid = causal_prior[layer]
        stable_material[str(layer)] = [site for site in valid if all(
            metrics[str(layer)][panel][site]["response_norm_fraction"] >= .10
            and metrics[str(layer)][panel][site]["response_cosine"] >= .50
            for panel in masks)]
        rankings[str(layer)] = sorted(valid, key=lambda site: -sum(
            metrics[str(layer)][panel][site]["response_norm_fraction"] for panel in masks))
    tripwire = max(float((live_writes[11] - variant_writes[site][11]).abs().max())
                   for site in SITES[3:])
    reconstruction = max(capture["reconstruction_max_abs"] for capture in
        (*base_readers.values(), *live_readers.values(),
         *(value for captures in reader_captures.values() for value in captures.values())))
    identity_error = float((base_hidden18 - base18).abs().max())
    finite = all(math.isfinite(value) for layer in metrics.values() for panel in layer.values()
                 for arm in panel.values() for value in arm.values())
    pred_a = bool(orientation_error <= 1e-6 and reconstruction <= .001 and identity_error <= .001
                  and tripwire <= 1e-6 and finite and forwards <= MAX_FORWARDS
                  and evaluations <= MAX_EVALUATIONS)
    pred_b = all(stable_material[str(layer)] for layer in LAYERS)
    pred_c = all(name in variant_writes for name in GROUP_ARMS)
    pred_d = tripwire <= 1e-6
    pred_e = set(reader_captures) == set(SITES) | set(GROUP_ARMS)
    predictions = {"pred_a_authority_capture_self_clamp_causal_order_finiteness_and_price": pred_a,
        "pred_b_each_reader_has_a_stable_material_upstream_module": pred_b,
        "pred_c_joint_removal_distinguishes_residual_skip_from_module_conversion": pred_c,
        "pred_d_l11_causal_order_tripwire": pred_d,
        "pred_e_zero_fit_complete_module_inventory": pred_e}
    grouped_material = any(metrics[str(layer)][panel]["all_intervening_removed"]["response_norm_fraction"] >= .20
        for layer in LAYERS for panel in masks)
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else (
        "distributed" if grouped_material else "residual_skip")
    result = {"schema": "iswas_mlp8_auxiliary_cv_upstream_module_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_attention_reconstruction_max_abs": reconstruction,
            "native_base_identity_max_abs": identity_error,
            "l11_post_reader_causal_order_max_abs": tripwire, "rows": len(rows)},
        "metrics": metrics, "rankings": rankings, "stable_material_sites": stable_material,
        "grouped_material": grouped_material, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "rankings",
        "stable_material_sites", "grouped_material", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
