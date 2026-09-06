#!/usr/bin/env python3
"""Conditional head atlas for auxiliary attention11/15 Q8 conversion."""

# BQGATE: EXPERIMENT pred_a_authority_replay_self_clamp_finiteness_and_price pred_b_at_least_one_stable_material_auxiliary_head pred_c_all_auxiliary_heads_replay_group_factorial pred_d_top_auxiliary_head_is_directionally_coherent pred_e_zero_fit_complete_head_inventory
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_mlp8_complement_downstream_group_factorial_v1 as group_parent
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_complement_downstream_group_factorial_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_complement_downstream_group_factorial_v1.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_v1"
EXPECTED = {
    "prior": "fbeaacaa115162741bb645bb2defb14c1d6b4bdf9e80dbcef8612193aa1c3d0d",
    "parent": "126f39650ace12c7888103cf2af68f36595bca9e6e4f8aa8fc7ecd8a23f20d9a",
    "parent_runner": "05ed68e5e8f376294fc1761c8bcabc604e52f28462ebde022769ead793392e9e",
}
LAYERS, HEADS = (11, 15), tuple(range(9))
MAX_FORWARDS, MAX_EVALUATIONS = 27, 783


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def capture_raws(backend, batch, base_hidden=None, delta=None, positions=None):
    cache, handles = {}, []
    if delta is not None:
        handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
            converter.actuation_hook(base_hidden, delta, positions)))
    for layer in (9,) + LAYERS:
        def capture(_module, arguments, layer=layer):
            cache[layer] = arguments[0].detach().clone()
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(capture))
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles: handle.remove()
    if set(cache) != {9, 11, 15}:
        raise RuntimeError("incomplete attention raw capture")
    return output, cache


def run_heads(backend, batch, base_hidden, delta, positions, base_raws, selections, *, actuate=True):
    handles = []
    if actuate:
        handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
            converter.actuation_hook(base_hidden, delta, positions)))
    n_head = int(backend.model.config.n_head)
    head_dim = int(backend.model.config.n_embd // n_head)
    for layer, selected in selections.items():
        if not selected:
            continue
        def patch(_module, arguments, layer=layer, selected=tuple(selected)):
            raw = arguments[0]
            changed = raw.clone().view(raw.shape[0], raw.shape[1], n_head, head_dim)
            base = base_raws[layer].view_as(changed)
            for i, query in enumerate(batch.semantic_positions):
                changed[i, :int(query)+1, list(selected)] = base[i, :int(query)+1, list(selected)].to(changed)
            return (changed.reshape_as(raw),) + tuple(arguments[1:])
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch))
    try:
        return backend.native(batch, capture=True)
    finally:
        for handle in handles: handle.remove()


def cosine(x, y) -> float:
    denominator = float(x.norm() * y.norm())
    return float((x * y).sum()) / denominator if denominator else 0.0


def main() -> None:
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("conditional auxiliary-head authority changed")
    inherited_paths = {
        "prior": converter.PRIOR, "weight_v2": converter.WEIGHT_V2,
        "weight_v2_runner": converter.WEIGHT_V2_RUNNER,
        "weight_instrument": converter.WEIGHT_INSTRUMENT, "source": converter.SOURCE,
        "capability": weight.CAPABILITY, "iswas": weight.ISWAS,
        "subspace": weight.SUBSPACE, "builder": weight.BUILDER,
        "family_runner": weight.FAMILY_RUNNER, "overlap_runner": weight.OVERLAP_RUNNER,
    }
    if {name: sha(path) for name, path in inherited_paths.items()} != converter.EXPECTED:
        raise RuntimeError("inherited converter authority changed")
    prior, parent = json.loads(PRIOR.read_text()), json.loads(PARENT.read_text())
    capability, subspace = json.loads(weight.CAPABILITY.read_text()), json.loads(weight.SUBSPACE.read_text())
    capable = {}
    for record in capability["native_records"]:
        capable.setdefault(record["row_id"], {})[record["side"]] = bool(record["correct"])
    allowed = {row_id for row_id, sides in capable.items() if sides == {"base": True, "donor": True}}
    rows = [row for row in weight.candidate.build_rows()
            if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    positions = [weight.postcue_positions(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "null"
            or not parent.get("predictions", {}).get("pred_b_four_group_program_is_near_complete")
            or len(rows) != 29):
        raise RuntimeError("parent decision or population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "layers": list(LAYERS), "heads_per_layer": list(HEADS),
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}
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
    base_hidden_output, base_capture = weight.capture_mlp8(backend, base_batch)
    _donor_output, donor_capture = weight.capture_mlp8(backend, donor_batch)
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _singular, vh = torch.linalg.svd(s.T @ down, full_matrices=False)
    full_delta = donor_capture["hidden"].float() - base_capture["hidden"].float()
    complement = full_delta - weight.project(full_delta, vh, vh.shape[0])
    base_output, base_raws = capture_raws(backend, base_batch)
    live_output, _live_raws = capture_raws(backend, base_batch, base_capture["hidden"], complement, positions)
    all_heads = HEADS
    core_selection = {9: all_heads}
    outputs = {"core": run_heads(backend, base_batch, base_capture["hidden"], complement,
                                  positions, base_raws, core_selection)}
    for layer in LAYERS:
        for head in HEADS:
            outputs[f"attn{layer}_head{head}"] = run_heads(
                backend, base_batch, base_capture["hidden"], complement, positions,
                base_raws, {9: all_heads, layer: (head,)})
    outputs["attn11_all"] = run_heads(backend, base_batch, base_capture["hidden"], complement,
        positions, base_raws, {9: all_heads, 11: all_heads})
    outputs["attn15_all"] = run_heads(backend, base_batch, base_capture["hidden"], complement,
        positions, base_raws, {9: all_heads, 15: all_heads})
    outputs["all_aux"] = run_heads(backend, base_batch, base_capture["hidden"], complement,
        positions, base_raws, {9: all_heads, 11: all_heads, 15: all_heads})
    self_output = run_heads(backend, base_batch, base_capture["hidden"], complement,
        positions, base_raws, {9: all_heads, 11: all_heads, 15: all_heads}, actuate=False)
    forwards, evaluations = 27, 27 * len(rows)

    state = lambda output: converter.state(output, rows, torch, backend.device)
    base18, base_hidden18, live18, self18 = map(state, (base_output, base_hidden_output, live_output, self_output))
    states = {name: state(value) for name, value in outputs.items()}
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda value: das.head_logits(backend, value)[index, answers] - das.head_logits(backend, value)[index, foils]
    base_margin, live_margin = margin(base18), margin(live18)
    arm_margins = {name: margin(value) for name, value in states.items()}
    live_effect, live_coord = live_margin - base_margin, (live18 - base18) @ s
    core_margin, core_state = arm_margins["core"], states["core"]
    full_aux_effect = core_margin - arm_margins["all_aux"]
    full_aux_coord = (core_state - states["all_aux"]) @ s
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)
             for panel in ("A1", "A2")}
    metrics = {panel: {} for panel in masks}
    for panel, mask in masks.items():
        aux_e, aux_c = full_aux_effect[mask], full_aux_coord[mask]
        for layer in LAYERS:
            for head in HEADS:
                name = f"attn{layer}_head{head}"
                marginal_e = (core_margin - arm_margins[name])[mask]
                marginal_c = ((core_state - states[name]) @ s)[mask]
                metrics[panel][name] = {
                    "absolute_behavior_fraction_of_aux": float(marginal_e.abs().mean() / aux_e.abs().mean()),
                    "signed_behavior_fraction_of_aux": float(marginal_e.mean() / aux_e.mean()),
                    "behavior_cosine_to_aux": cosine(marginal_e, aux_e),
                    "q8_norm_fraction_of_aux": float(marginal_c.norm() / aux_c.norm()),
                    "q8_cosine_to_aux": cosine(marginal_c.reshape(-1), aux_c.reshape(-1)),
                    "absolute_behavior_fraction_of_live": float(marginal_e.abs().mean() / live_effect[mask].abs().mean()),
                    "q8_norm_fraction_of_live": float(marginal_c.norm() / live_coord[mask].norm()),
                }
        for name in ("core", "attn11_all", "attn15_all", "all_aux"):
            removed_e = (live_margin - arm_margins[name])[mask]
            removed_c = ((live18 - states[name]) @ s)[mask]
            metrics[panel][name] = {
                "absolute_behavior_fraction_of_live": float(removed_e.abs().mean() / live_effect[mask].abs().mean()),
                "behavior_cosine_to_live": cosine(removed_e, live_effect[mask]),
                "q8_norm_fraction_of_live": float(removed_c.norm() / live_coord[mask].norm()),
                "q8_cosine_to_live": cosine(removed_c.reshape(-1), live_coord[mask].reshape(-1)),
            }
    stable = []
    for layer in LAYERS:
        for head in HEADS:
            name = f"attn{layer}_head{head}"
            if all(metrics[p][name]["absolute_behavior_fraction_of_aux"] >= .10
                   and metrics[p][name]["q8_norm_fraction_of_aux"] >= .10 for p in ("A1", "A2")):
                stable.append(name)
    ranking = sorted(stable, key=lambda name: -sum(
        metrics[p][name]["absolute_behavior_fraction_of_aux"]
        + metrics[p][name]["q8_norm_fraction_of_aux"] for p in ("A1", "A2")))
    top = ranking[0] if ranking else None
    parent_core = "core_attention"
    parent_aux = "core_attention+auxiliary_attention"
    replay_error = max(abs(metrics[p][arm][key] - parent["metrics"][p][parent_arm][parent_key])
        for p in ("A1", "A2")
        for arm, parent_arm in (("core", parent_core), ("all_aux", parent_aux))
        for key, parent_key in (("absolute_behavior_fraction_of_live", "absolute_behavior_fraction"),
                                ("q8_norm_fraction_of_live", "q8_norm_fraction")))
    identity_error = max(float((base_hidden18 - base18).abs().max()), float((self18 - base18).abs().max()))
    finite = all(math.isfinite(value) for panel in metrics.values()
                 for arm in panel.values() for value in arm.values())
    pred_a = bool(orientation_error <= 1e-6 and identity_error <= 1e-4
                  and replay_error <= .001 and finite and forwards <= MAX_FORWARDS
                  and evaluations <= MAX_EVALUATIONS)
    pred_b = bool(stable)
    pred_c = replay_error <= .001
    pred_d = bool(top and all(metrics[p][top]["behavior_cosine_to_aux"] >= .85
        and metrics[p][top]["q8_cosine_to_aux"] >= .85 for p in ("A1", "A2")))
    pred_e = len([name for name in outputs if "_head" in name]) == 18 and len(outputs) == 22
    predictions = {
        "pred_a_authority_replay_self_clamp_finiteness_and_price": pred_a,
        "pred_b_at_least_one_stable_material_auxiliary_head": pred_b,
        "pred_c_all_auxiliary_heads_replay_group_factorial": pred_c,
        "pred_d_top_auxiliary_head_is_directionally_coherent": pred_d,
        "pred_e_zero_fit_complete_head_inventory": pred_e,
    }
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "distributed_heads"
    result = {"schema": "iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": {**EXPECTED, **{f"inherited_{name}": value for name, value in converter.EXPECTED.items()}},
        "dryrun": dryrun, "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_base_self_clamp_resid18_max_abs": identity_error,
            "core_and_aux_parent_replay_max_abs": replay_error, "rows": len(rows)},
        "metrics": metrics, "stable_material_heads": stable, "stable_ranking": ranking,
        "top_stable_head": top, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "stable_material_heads",
        "stable_ranking", "top_stable_head", "metrics", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
