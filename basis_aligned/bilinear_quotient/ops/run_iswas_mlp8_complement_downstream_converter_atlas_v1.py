#!/usr/bin/env python3
"""Complete-module atlas for conversion of the MLP8 complement into final Q8."""

# BQGATE: EXPERIMENT pred_a_authority_replay_self_clamp_finiteness_and_price pred_b_at_least_one_stable_material_converter pred_c_joint_downstream_responses_mediate_the_complement pred_d_converter_removal_is_directionally_coherent pred_e_zero_fit_complete_inventory
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
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_complement_downstream_converter_atlas_v1.json"
WEIGHT_V2 = ROOT / "circuits/followups/iswas_shared_q8_mlp8_postcue_weight_modes_v2_result.json"
WEIGHT_V2_RUNNER = ROOT / "ops/run_iswas_shared_q8_mlp8_postcue_weight_modes_v2.py"
WEIGHT_INSTRUMENT = ROOT / "ops/run_iswas_shared_q8_mlp8_postcue_weight_modes_v1.py"
SOURCE = ROOT / "circuits/followups/iswas_shared_q8_mlp8_source_position_atlas_v1_result.json"
OUT = ROOT / "circuits/followups/iswas_mlp8_complement_downstream_converter_atlas_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_complement_downstream_converter_atlas_v1"
EXPECTED = {
    "prior": "8e5a0dc170126f4d88528363de3dd3e756097536c9d0db005c89132bab957d66",
    "weight_v2": "623824f843e2e5b96e3ddfab9ad1e36118136949db8d7e3dcc4594e6b104980c",
    "weight_v2_runner": "3c1adf53bd1e4e2fc77395ec905dcb9c47bbf1e02ade42b0927d3ad1f9bdaa8b",
    "weight_instrument": "d82035e45f8818a7de73024500592f1677f887e8be5d8a2e617c28afc6c1238c",
    "source": "ffa8596eea052e72eba3a5823dfdfcd124ee28ccc91ca48671efff567f23b14a",
    "capability": "86ec66fa81346e61382c951e46899236ee1b7b7ec32c16948936fd9de6f77940",
    "iswas": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "builder": "b8541360334bd2793a02fae525a94dda05ce600fd4de5b6c3d953063d4c6b0ae",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "overlap_runner": "15f2e1119c4df7c29aaf51c615ec92d87555d65efce029f16a89478212f29af1",
}
SITES = tuple(f"{kind}:{layer:02d}" for layer in range(9, 18) for kind in ("attn", "mlp"))
MAX_FORWARDS, MAX_EVALUATIONS = 24, 696


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def site_module(backend, site):
    kind, layer = site.split(":")
    block = backend.model.transformer.h[int(layer)]
    return block.attn.c_proj if kind == "attn" else block.mlp


def actuation_hook(base_hidden, delta, positions):
    def patch(_module, arguments):
        changed = arguments[0].clone()
        for i, selected in enumerate(positions):
            changed[i, list(selected)] = (base_hidden[i, list(selected)].float()
                + delta[i, list(selected)].float()).to(changed)
        return (changed,) + tuple(arguments[1:])
    return patch


def capture_modules(backend, batch, base_hidden=None, delta=None, positions=None):
    cache, handles = {}, []
    if delta is not None:
        handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
            actuation_hook(base_hidden, delta, positions)))
    for site in SITES:
        kind = site.split(":")[0]
        if kind == "attn":
            def capture(_module, arguments, site=site):
                cache[site] = arguments[0].detach().clone()
            handles.append(site_module(backend, site).register_forward_pre_hook(capture))
        else:
            def capture(_module, _arguments, output, site=site):
                cache[site] = output.detach().clone()
            handles.append(site_module(backend, site).register_forward_hook(capture))
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(cache) != set(SITES):
        raise RuntimeError("incomplete downstream module capture")
    return output, cache


def clamp_hook(batch, base_value, kind):
    if kind == "attn":
        def patch(_module, arguments):
            changed = arguments[0].clone()
            for i, query in enumerate(batch.semantic_positions):
                changed[i, :int(query)+1] = base_value[i, :int(query)+1].to(changed)
            return (changed,) + tuple(arguments[1:])
    else:
        def patch(_module, _arguments, output):
            changed = output.clone()
            for i, query in enumerate(batch.semantic_positions):
                changed[i, :int(query)+1] = base_value[i, :int(query)+1].to(changed)
            return changed
    return patch


def run_clamped(backend, batch, base_hidden, delta, positions, base_modules, sites,
                *, actuate=True):
    handles = []
    if actuate:
        handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
            actuation_hook(base_hidden, delta, positions)))
    for site in sites:
        kind = site.split(":")[0]
        hook = clamp_hook(batch, base_modules[site], kind)
        module = site_module(backend, site)
        handles.append(module.register_forward_pre_hook(hook) if kind == "attn"
                       else module.register_forward_hook(hook))
    try:
        return backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()


def state(output, rows, torch, device):
    return torch.stack([torch.as_tensor(output.captured[(row["row_id"], "resid:18")])
                        for row in rows]).to(device).float()


def cosine(x, y):
    denominator = float(x.norm()*y.norm())
    return float((x*y).sum())/denominator if denominator else 0.0


def main():
    paths = {"prior": PRIOR, "weight_v2": WEIGHT_V2, "weight_v2_runner": WEIGHT_V2_RUNNER,
        "weight_instrument": WEIGHT_INSTRUMENT, "source": SOURCE,
        "capability": weight.CAPABILITY, "iswas": weight.ISWAS, "subspace": weight.SUBSPACE,
        "builder": weight.BUILDER, "family_runner": weight.FAMILY_RUNNER,
        "overlap_runner": weight.OVERLAP_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("downstream converter authority changed")
    prior = json.loads(PRIOR.read_text())
    parent = json.loads(WEIGHT_V2.read_text())
    capability = json.loads(weight.CAPABILITY.read_text())
    subspace = json.loads(weight.SUBSPACE.read_text())
    capable = {}
    for record in capability["native_records"]:
        capable.setdefault(record["row_id"], {})[record["side"]] = bool(record["correct"])
    allowed = {row_id for row_id, sides in capable.items() if sides == {"base": True, "donor": True}}
    rows = [row for row in weight.candidate.build_rows()
            if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    positions = [weight.postcue_positions(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "null"
            or len(rows) != 29 or len(SITES) != 18):
        raise RuntimeError("parent terminal, row population, or site inventory changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "sites": list(SITES),
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
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    base_output, base_capture = weight.capture_mlp8(backend, base_batch)
    _donor_output, donor_capture = weight.capture_mlp8(backend, donor_batch)
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _singular, vh = torch.linalg.svd(s.T@down, full_matrices=False)
    full_delta = donor_capture["hidden"].float()-base_capture["hidden"].float()
    complement = full_delta-weight.project(full_delta, vh, vh.shape[0])
    base_modules_output, base_modules = capture_modules(backend, base_batch)
    live_output, _live_modules = capture_modules(
        backend, base_batch, base_capture["hidden"], complement, positions)
    outputs = {}
    for site in SITES:
        outputs[site] = run_clamped(backend, base_batch, base_capture["hidden"], complement,
                                    positions, base_modules, (site,))
    outputs["all"] = run_clamped(backend, base_batch, base_capture["hidden"], complement,
                                  positions, base_modules, SITES)
    self_output = run_clamped(backend, base_batch, base_capture["hidden"], complement,
                              positions, base_modules, SITES, actuate=False)
    forwards, evaluations = 24, 24*len(rows)
    base18 = state(base_output, rows, torch, backend.device)
    base_modules18 = state(base_modules_output, rows, torch, backend.device)
    live18 = state(live_output, rows, torch, backend.device)
    self18 = state(self_output, rows, torch, backend.device)
    states = {name: state(output, rows, torch, backend.device) for name, output in outputs.items()}
    base_logits, live_logits = das.head_logits(backend, base18), das.head_logits(backend, live18)
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda logits: logits[index, answers]-logits[index, foils]
    base_margin, live_margin = margin(base_logits), margin(live_logits)
    clamped_margins = {name: margin(das.head_logits(backend, value)) for name, value in states.items()}
    live_effect = live_margin-base_margin
    live_coordinates = (live18-base18)@s
    replay_effect_error = abs(float(live_effect.abs().mean())
        - parent["metrics"]["rank8_complement"]["mean_absolute_effect"])
    replay_coordinate_error = abs(float(torch.sqrt(live_coordinates.square().mean()))
        - parent["metrics"]["rank8_complement"]["coordinate_rms"])
    metrics = {panel: {} for panel in ("A1", "A2")}
    panel_masks = {panel: torch.as_tensor([row["family"] == panel for row in rows],
                                           device=backend.device) for panel in ("A1", "A2")}
    for panel, mask in panel_masks.items():
        panel_live_effect = live_effect[mask]
        panel_live_coord = live_coordinates[mask]
        for name in tuple(SITES)+("all",):
            removed_effect = (live_margin-clamped_margins[name])[mask]
            removed_coord = ((live18-states[name])@s)[mask]
            metrics[panel][name] = {
                "signed_behavior_fraction": float(removed_effect.mean()/panel_live_effect.mean()),
                "absolute_behavior_fraction": float(removed_effect.abs().mean()/panel_live_effect.abs().mean()),
                "behavior_cosine": cosine(removed_effect, panel_live_effect),
                "q8_norm_fraction": float(removed_coord.norm()/panel_live_coord.norm()),
                "q8_cosine": cosine(removed_coord.reshape(-1), panel_live_coord.reshape(-1)),
            }
    rankings = {panel: sorted(({"site": site, **metrics[panel][site]} for site in SITES),
        key=lambda row: -(row["absolute_behavior_fraction"]+row["q8_norm_fraction"]))
        for panel in ("A1", "A2")}
    material = [site for site in SITES if all(
        metrics[panel][site]["absolute_behavior_fraction"] >= .10
        and metrics[panel][site]["q8_norm_fraction"] >= .10 for panel in ("A1", "A2"))]
    stable_ranking = sorted(material, key=lambda site: -sum(
        metrics[panel][site]["absolute_behavior_fraction"]+metrics[panel][site]["q8_norm_fraction"]
        for panel in ("A1", "A2")))
    top = stable_ranking[0] if stable_ranking else None
    finite = all(math.isfinite(value) for panel in metrics.values()
                 for site in panel.values() for value in site.values())
    identity_error = max(float((base_modules18-base18).abs().max()),
                         float((self18-base18).abs().max()))
    pred_a = bool(orientation_error <= 1e-6 and identity_error <= 1e-4
        and replay_effect_error <= 1e-5 and replay_coordinate_error <= 1e-3
        and finite and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    pred_b = bool(material)
    pred_c = all(metrics[panel]["all"]["absolute_behavior_fraction"] >= .80
                 and metrics[panel]["all"]["q8_norm_fraction"] >= .80 for panel in ("A1", "A2"))
    pred_d = bool(top and all(metrics[panel][top]["behavior_cosine"] >= .70
                              for panel in ("A1", "A2")))
    pred_e = len(SITES) == 18
    predictions = {"pred_a_authority_replay_self_clamp_finiteness_and_price": pred_a,
        "pred_b_at_least_one_stable_material_converter": pred_b,
        "pred_c_joint_downstream_responses_mediate_the_complement": pred_c,
        "pred_d_converter_removal_is_directionally_coherent": pred_d,
        "pred_e_zero_fit_complete_inventory": pred_e}
    joint_material = any(metrics[panel]["all"]["absolute_behavior_fraction"] >= .20
                         or metrics[panel]["all"]["q8_norm_fraction"] >= .20
                         for panel in ("A1", "A2"))
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else (
        "distributed_converter" if joint_material else "residual_normalization_route")
    result = {"schema": "iswas_mlp8_complement_downstream_converter_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_base_self_clamp_resid18_max_abs": identity_error,
            "parent_mean_absolute_effect_replay_abs": replay_effect_error,
            "parent_coordinate_rms_replay_abs": replay_coordinate_error, "rows": len(rows)},
        "metrics": metrics, "rankings": rankings, "stable_material_sites": material,
        "top_stable_site": top, "joint_material": joint_material,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "rankings",
        "stable_material_sites", "top_stable_site", "joint_material", "predictions",
        "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
