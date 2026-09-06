#!/usr/bin/env python3
"""Nine-head converter atlas inside attention9 under MLP8 complement actuation."""

# BQGATE: EXPERIMENT pred_a_authority_replay_self_clamp_finiteness_and_price pred_b_at_least_one_stable_material_head pred_c_all_heads_replay_complete_attention9_removal pred_d_top_head_is_directionally_coherent pred_e_head_program_is_compressive_and_zero_fit
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
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_complement_attn9_head_converter_atlas_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_complement_downstream_converter_atlas_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_complement_downstream_converter_atlas_v1.py"
WEIGHT_RESULT = ROOT / "circuits/followups/iswas_shared_q8_mlp8_postcue_weight_modes_v2_result.json"
WEIGHT_INSTRUMENT = ROOT / "ops/run_iswas_shared_q8_mlp8_postcue_weight_modes_v1.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_complement_attn9_head_converter_atlas_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_complement_attn9_head_converter_atlas_v1"
EXPECTED = {
    "prior": "b8bdabfd5a8f8a229714f1b4b00e79a9aa8afe143b4fac2a8a126b302f6f44b1",
    "parent": "2ccfb7116e45665820a546ebd5edc2bf4e2616b49cf46723064891640835ac5b",
    "parent_runner": "d757d4c54b53d5cda8d877bfe9a01e636f4459582be6cb379b28f0e8823be218",
    "weight_result": "623824f843e2e5b96e3ddfab9ad1e36118136949db8d7e3dcc4594e6b104980c",
    "weight_instrument": "d82035e45f8818a7de73024500592f1677f887e8be5d8a2e617c28afc6c1238c",
    "capability": "86ec66fa81346e61382c951e46899236ee1b7b7ec32c16948936fd9de6f77940",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "overlap_runner": "15f2e1119c4df7c29aaf51c615ec92d87555d65efce029f16a89478212f29af1",
}
HEADS = tuple(range(9))
MAX_FORWARDS, MAX_EVALUATIONS = 15, 435


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def capture_attn9(backend, batch, base_hidden=None, delta=None, positions=None):
    cache, handles = {}, []
    if delta is not None:
        handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
            converter.actuation_hook(base_hidden, delta, positions)))
    def capture(_module, arguments):
        cache["raw"] = arguments[0].detach().clone()
    handles.append(backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(capture))
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()
    return output, cache["raw"]


def run_heads(backend, batch, base_hidden, delta, positions, base_raw, selected, *, actuate=True):
    handles = []
    if actuate:
        handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
            converter.actuation_hook(base_hidden, delta, positions)))
    n_head = int(backend.model.config.n_head)
    head_dim = int(backend.model.config.n_embd//n_head)
    def patch(_module, arguments):
        raw = arguments[0]
        changed = raw.clone().view(raw.shape[0], raw.shape[1], n_head, head_dim)
        base = base_raw.view_as(changed)
        for i, query in enumerate(batch.semantic_positions):
            changed[i, :int(query)+1, list(selected)] = base[i, :int(query)+1, list(selected)].to(changed)
        return (changed.reshape_as(raw),)+tuple(arguments[1:])
    handles.append(backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(patch))
    try:
        return backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()


def cosine(x, y):
    denominator = float(x.norm()*y.norm())
    return float((x*y).sum())/denominator if denominator else 0.0


def main():
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER,
        "weight_result": WEIGHT_RESULT, "weight_instrument": WEIGHT_INSTRUMENT,
        "capability": weight.CAPABILITY, "subspace": weight.SUBSPACE,
        "family_runner": weight.FAMILY_RUNNER, "overlap_runner": weight.OVERLAP_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("attention9 head-atlas authority changed")
    prior, parent, weight_result, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, PARENT, WEIGHT_RESULT, weight.CAPABILITY, weight.SUBSPACE)]
    capable = {}
    for record in capability["native_records"]:
        capable.setdefault(record["row_id"], {})[record["side"]] = bool(record["correct"])
    allowed = {row_id for row_id, sides in capable.items() if sides == {"base": True, "donor": True}}
    rows = [row for row in weight.candidate.build_rows()
            if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    positions = [weight.postcue_positions(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen"
            or parent.get("top_stable_site") != "attn:09" or weight_result.get("terminal") != "null"
            or len(rows) != 29):
        raise RuntimeError("parent decision or population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "heads": list(HEADS),
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
    _u, _singular, vh = torch.linalg.svd(s.T@down, full_matrices=False)
    full_delta = donor_capture["hidden"].float()-base_capture["hidden"].float()
    complement = full_delta-weight.project(full_delta, vh, vh.shape[0])
    base_output, base_raw = capture_attn9(backend, base_batch)
    live_output, _live_raw = capture_attn9(
        backend, base_batch, base_capture["hidden"], complement, positions)
    outputs = {f"head{head}": run_heads(backend, base_batch, base_capture["hidden"],
        complement, positions, base_raw, (head,)) for head in HEADS}
    outputs["all"] = run_heads(backend, base_batch, base_capture["hidden"], complement,
                                positions, base_raw, HEADS)
    self_output = run_heads(backend, base_batch, base_capture["hidden"], complement,
                            positions, base_raw, HEADS, actuate=False)
    forwards, evaluations = 15, 15*len(rows)
    base18 = converter.state(base_output, rows, torch, backend.device)
    base_hidden18 = converter.state(base_hidden_output, rows, torch, backend.device)
    live18 = converter.state(live_output, rows, torch, backend.device)
    self18 = converter.state(self_output, rows, torch, backend.device)
    states = {name: converter.state(output, rows, torch, backend.device)
              for name, output in outputs.items()}
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda state: das.head_logits(backend, state)[index, answers]-das.head_logits(backend, state)[index, foils]
    base_margin, live_margin = margin(base18), margin(live18)
    live_effect, live_coord = live_margin-base_margin, (live18-base18)@s
    clamped_margin = {name: margin(value) for name, value in states.items()}
    metrics = {panel: {} for panel in ("A1", "A2")}
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)
             for panel in ("A1", "A2")}
    for panel, mask in masks.items():
        pe, pc = live_effect[mask], live_coord[mask]
        for name in tuple(f"head{head}" for head in HEADS)+("all",):
            removed_e = (live_margin-clamped_margin[name])[mask]
            removed_c = ((live18-states[name])@s)[mask]
            metrics[panel][name] = {"absolute_behavior_fraction": float(removed_e.abs().mean()/pe.abs().mean()),
                "signed_behavior_fraction": float(removed_e.mean()/pe.mean()),
                "behavior_cosine": cosine(removed_e, pe),
                "q8_norm_fraction": float(removed_c.norm()/pc.norm()),
                "q8_cosine": cosine(removed_c.reshape(-1), pc.reshape(-1))}
    rankings = {panel: sorted(({"head": head, **metrics[panel][f"head{head}"]} for head in HEADS),
        key=lambda row: -(row["absolute_behavior_fraction"]+row["q8_norm_fraction"]))
        for panel in ("A1", "A2")}
    material = [head for head in HEADS if all(
        metrics[panel][f"head{head}"]["absolute_behavior_fraction"] >= .15
        and metrics[panel][f"head{head}"]["q8_norm_fraction"] >= .15 for panel in ("A1", "A2"))]
    top = sorted(material, key=lambda head: -sum(
        metrics[panel][f"head{head}"]["absolute_behavior_fraction"]
        + metrics[panel][f"head{head}"]["q8_norm_fraction"] for panel in ("A1", "A2")))[0] if material else None
    all_replay_error = max(abs(metrics[panel]["all"][key]-parent["metrics"][panel]["attn:09"][key])
        for panel in ("A1", "A2") for key in ("absolute_behavior_fraction", "q8_norm_fraction"))
    identity_error = max(float((base_hidden18-base18).abs().max()), float((self18-base18).abs().max()))
    finite = all(math.isfinite(value) for panel in metrics.values() for row in panel.values()
                 for value in row.values())
    pred_a = bool(orientation_error <= 1e-6 and identity_error <= 1e-4 and finite
        and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS and len(HEADS) == 9)
    pred_b = bool(material)
    pred_c = all_replay_error <= .001
    pred_d = bool(top is not None and all(metrics[panel][f"head{top}"]["behavior_cosine"] >= .90
        and metrics[panel][f"head{top}"]["q8_cosine"] >= .90 for panel in ("A1", "A2")))
    pred_e = bool(len(material) <= 3 and all(sum(metrics[panel][f"head{head}"]["absolute_behavior_fraction"]
        for head in material) >= .75*metrics[panel]["all"]["absolute_behavior_fraction"]
        for panel in ("A1", "A2")))
    predictions = {"pred_a_authority_replay_self_clamp_finiteness_and_price": pred_a,
        "pred_b_at_least_one_stable_material_head": pred_b,
        "pred_c_all_heads_replay_complete_attention9_removal": pred_c,
        "pred_d_top_head_is_directionally_coherent": pred_d,
        "pred_e_head_program_is_compressive_and_zero_fit": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "distributed_heads"
    result = {"schema": "iswas_mlp8_complement_attn9_head_converter_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_base_self_clamp_resid18_max_abs": identity_error,
            "all_head_parent_replay_max_abs": all_replay_error, "rows": len(rows)},
        "metrics": metrics, "rankings": rankings, "stable_material_heads": material,
        "top_stable_head": top, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "rankings",
        "stable_material_heads", "top_stable_head", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__":
    main()
