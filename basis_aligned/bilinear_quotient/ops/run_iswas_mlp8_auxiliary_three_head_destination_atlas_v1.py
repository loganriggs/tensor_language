#!/usr/bin/env python3
"""Destination atlas for the confirmed L11H1/H3 and L15H5 auxiliary converter."""

# BQGATE: EXPERIMENT pred_a_authority_partition_self_clamp_finiteness_and_price pred_b_all_destinations_replay_confirmed_auxiliary_marginal pred_c_at_least_one_proper_destination_is_material pred_d_causal_prefix_destination_is_zero pred_e_destination_program_is_compressive
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_v1 as auxiliary
import run_iswas_mlp8_complement_attn9_h1h4_destination_atlas_v1 as destination
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_auxiliary_three_head_destination_atlas_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_complement_five_head_fresh_v10_confirmation_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_complement_five_head_fresh_v10_confirmation_v1.py"
DESTINATION_RUNNER = ROOT / "ops/run_iswas_mlp8_complement_attn9_h1h4_destination_atlas_v1.py"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_three_head_destination_atlas_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_auxiliary_three_head_destination_atlas_v1"
EXPECTED = {
    "prior": "df4821ea7439f533d4064e31fe6095b5bbf4739d5b5eaa6ec785e7f7036ecaf9",
    "parent": "62fe910a3c034d5f3121b12b331dc6d80416b32553997cc55c76574fce0a2567",
    "parent_runner": "45dd2232a2609afd121cf7e4f7d2c6de8a960d76d4b3cba01358a441cf65c43f",
    "destination_runner": "2fb6d8ed81e122161eb4ee8669b84639a3e26aa5859c2936ae3b39cb5f46ed3c",
    "capability": "77e3c5b6e47dc9416133643f422c130427f90fed0746d26f211f54b681718b3d",
    "builder": "13e7954cde01b6b7d826915fc2ae02d4b9e16975150cf73aa5e0a1f906c1b757",
}
GROUPS = destination.GROUPS
CORE, AUX = {9: (1, 4)}, {11: (1, 3), 15: (5,)}
MAX_FORWARDS, MAX_EVALUATIONS = 12, 324


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def cosine(x, y) -> float:
    denominator = float(x.norm() * y.norm())
    return float((x * y).sum()) / denominator if denominator else 0.0


def run_destinations(backend, batch, base_hidden, delta, source_positions, base_raws,
                     auxiliary_destinations, *, actuate=True):
    handles = []
    if actuate:
        handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
            converter.actuation_hook(base_hidden, delta, source_positions)))
    n_head = int(backend.model.config.n_head)
    head_dim = int(backend.model.config.n_embd // n_head)
    for layer, heads in {**CORE, **AUX}.items():
        def patch(_module, arguments, layer=layer, heads=heads):
            raw = arguments[0]
            changed = raw.clone().view(raw.shape[0], raw.shape[1], n_head, head_dim)
            base = base_raws[layer].view_as(changed)
            selected_positions = ([tuple(range(int(query) + 1)) for query in batch.semantic_positions]
                                  if layer == 9 else auxiliary_destinations)
            for i, selected in enumerate(selected_positions):
                for position in selected:
                    changed[i, position, list(heads)] = base[i, position, list(heads)].to(changed)
            return (changed.reshape_as(raw),) + tuple(arguments[1:])
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch))
    try:
        return backend.native(batch, capture=True)
    finally:
        for handle in handles: handle.remove()


def main() -> None:
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER,
             "destination_runner": DESTINATION_RUNNER, "capability": CAPABILITY, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("auxiliary destination authority changed")
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
    prior, parent, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, PARENT, CAPABILITY, weight.SUBSPACE)]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    source_positions = [weight.postcue_positions(row) for row in rows]
    partitions = [destination.destination_partition(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen"
            or len(rows) != 27 or any(set(value) != set(GROUPS) for value in partitions)):
        raise RuntimeError("parent terminal, population, or partition changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "groups": list(GROUPS),
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
    base_output, base_raws = auxiliary.capture_raws(backend, base_batch)
    live_output, live_raws = auxiliary.capture_raws(
        backend, base_batch, base_capture["hidden"], complement, source_positions)
    empty_positions = [() for _row in rows]
    core_output = run_destinations(backend, base_batch, base_capture["hidden"], complement,
        source_positions, base_raws, empty_positions)
    arm_positions = {name: [partition[name] for partition in partitions] for name in GROUPS}
    arm_positions["all"] = [tuple(range(int(row["base_semantic_position"]) + 1)) for row in rows]
    outputs = {name: run_destinations(backend, base_batch, base_capture["hidden"], complement,
        source_positions, base_raws, positions) for name, positions in arm_positions.items()}
    self_output = run_destinations(backend, base_batch, base_capture["hidden"], complement,
        source_positions, base_raws, arm_positions["all"], actuate=False)
    forwards, evaluations = 12, 12 * len(rows)
    state = lambda output: converter.state(output, rows, torch, backend.device)
    base18, base_hidden18, live18, core18, self18 = map(
        state, (base_output, base_hidden_output, live_output, core_output, self_output))
    states = {name: state(value) for name, value in outputs.items()}
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda value: das.head_logits(backend, value)[index, answers] - das.head_logits(backend, value)[index, foils]
    live_margin, core_margin = margin(live18), margin(core18)
    full_aux_e, full_aux_c = core_margin - margin(states["all"]), (core18 - states["all"]) @ s
    live_e, live_c = live_margin - margin(base18), (live18 - base18) @ s
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)
             for panel in ("A1", "A2")}
    metrics = {panel: {} for panel in masks}
    for panel, mask in masks.items():
        for name, value in states.items():
            removed_e, removed_c = (core_margin - margin(value))[mask], ((core18 - value) @ s)[mask]
            metrics[panel][name] = {
                "absolute_behavior_fraction_of_aux": float(removed_e.abs().mean() / full_aux_e[mask].abs().mean()),
                "signed_behavior_fraction_of_aux": float(removed_e.mean() / full_aux_e[mask].mean()),
                "behavior_cosine_to_aux": cosine(removed_e, full_aux_e[mask]),
                "q8_norm_fraction_of_aux": float(removed_c.norm() / full_aux_c[mask].norm()),
                "q8_cosine_to_aux": cosine(removed_c.reshape(-1), full_aux_c[mask].reshape(-1)),
            }
        for name, value in (("core", core18), ("five", states["all"])):
            removed_e, removed_c = (live_margin - margin(value))[mask], ((live18 - value) @ s)[mask]
            metrics[panel][name] = {"absolute_behavior_fraction_of_live": float(removed_e.abs().mean() / live_e[mask].abs().mean()),
                "q8_norm_fraction_of_live": float(removed_c.norm() / live_c[mask].norm())}
    prefix_raw = max(max(float((live_raws[layer][i, list(partitions[i]["prefix_before_cue"])]
        - base_raws[layer][i, list(partitions[i]["prefix_before_cue"])]).view(
            len(partitions[i]["prefix_before_cue"]), int(backend.model.config.n_head), -1)[:, list(heads)].abs().max())
        if partitions[i]["prefix_before_cue"] else 0.0 for i in range(len(rows))) for layer, heads in AUX.items())
    parent_replay = max(abs(metrics[p][arm][key] - parent["metrics"][p][parent_arm][key])
        for p in ("A1", "A2") for arm, parent_arm in (("core", "core_selected"), ("five", "five_selected"))
        for key in ("absolute_behavior_fraction_of_live", "q8_norm_fraction_of_live"))
    material = [name for name in GROUPS if all(metrics[p][name]["absolute_behavior_fraction_of_aux"] >= .50
        and metrics[p][name]["q8_norm_fraction_of_aux"] >= .50 for p in ("A1", "A2"))]
    identity_error = max(float((base_hidden18 - base18).abs().max()), float((self18 - base18).abs().max()))
    finite = all(math.isfinite(value) for panel in metrics.values() for arm in panel.values() for value in arm.values())
    pred_a = bool(orientation_error <= 1e-6 and identity_error <= 1e-4 and finite
                  and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    pred_b = parent_replay <= .001
    pred_c = bool(material)
    pred_d = bool(prefix_raw <= 1e-6 and all(metrics[p]["prefix_before_cue"]["absolute_behavior_fraction_of_aux"] <= 1e-6
        and metrics[p]["prefix_before_cue"]["q8_norm_fraction_of_aux"] <= 1e-6 for p in ("A1", "A2")))
    pred_e = bool(len(material) <= 2 and all(sum(metrics[p][name]["absolute_behavior_fraction_of_aux"]
        for name in material) >= .75 for p in ("A1", "A2")))
    predictions = {"pred_a_authority_partition_self_clamp_finiteness_and_price": pred_a,
        "pred_b_all_destinations_replay_confirmed_auxiliary_marginal": pred_b,
        "pred_c_at_least_one_proper_destination_is_material": pred_c,
        "pred_d_causal_prefix_destination_is_zero": pred_d,
        "pred_e_destination_program_is_compressive": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "iswas_mlp8_auxiliary_three_head_destination_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_base_self_clamp_resid18_max_abs": identity_error,
            "parent_core_and_five_metric_replay_max_abs": parent_replay,
            "prefix_raw_auxiliary_response_delta_max_abs": prefix_raw, "rows": len(rows)},
        "metrics": metrics, "stable_material_destinations": material,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "metrics",
        "stable_material_destinations", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
