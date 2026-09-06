#!/usr/bin/env python3
"""Destination-position atlas for the confirmed attention9 H1+H4 converter."""

# BQGATE: EXPERIMENT pred_a_authority_partition_self_clamp_finiteness_and_price pred_b_all_destinations_replay_confirmed_h1h4_removal pred_c_at_least_one_proper_destination_is_material pred_d_causal_prefix_destination_is_zero pred_e_destination_program_is_compressive
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
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10 as candidate
import run_iswas_mlp8_complement_attn9_head_converter_atlas_v1 as head_atlas
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_complement_attn9_h1h4_destination_atlas_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_complement_attn9_h1h4_fresh_v10_confirmation_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_complement_attn9_h1h4_fresh_v10_confirmation_v1.py"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10.py"
HEAD_RUNNER = ROOT / "ops/run_iswas_mlp8_complement_attn9_head_converter_atlas_v1.py"
WEIGHT_INSTRUMENT = ROOT / "ops/run_iswas_shared_q8_mlp8_postcue_weight_modes_v1.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_complement_attn9_h1h4_destination_atlas_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_complement_attn9_h1h4_destination_atlas_v1"
EXPECTED = {"prior": "0c0a80cda71921f96d83e7a9ef3ca03597f1b8071c44d65953960082351b574c",
    "parent": "eb4470b3ed1495a5ad11c0a1e685d0c6ca4e9942532a872f165bc16f07eb69f5",
    "parent_runner": "d0209e6da76eb6cd1a11ef5f3487c693cd878b97242383e7fda00ddac10c362b",
    "capability": "77e3c5b6e47dc9416133643f422c130427f90fed0746d26f211f54b681718b3d",
    "builder": "13e7954cde01b6b7d826915fc2ae02d4b9e16975150cf73aa5e0a1f906c1b757",
    "head_runner": "12cd3dc948010a0a2e96b27a69272af81d2e394310ee2da3f691cb8d18718f0c",
    "weight_instrument": "d82035e45f8818a7de73024500592f1677f887e8be5d8a2e617c28afc6c1238c",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "overlap_runner": "15f2e1119c4df7c29aaf51c615ec92d87555d65efce029f16a89478212f29af1"}
GROUPS = ("prefix_before_cue", "cue", "post_cue_before_subject", "subject_determiner", "final_query")
MAX_FORWARDS, MAX_EVALUATIONS = 11, 297


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def utc_now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
def cosine(x, y):
    denominator = float(x.norm()*y.norm())
    return float((x*y).sum())/denominator if denominator else 0.0


def destination_partition(row):
    differences = [i for i, pair in enumerate(zip(row["base_ids"], row["donor_ids"])) if pair[0] != pair[1]]
    if len(differences) != 1:
        raise RuntimeError("destination atlas requires one aligned cue")
    cue, query = differences[0], int(row["base_semantic_position"])
    groups = {"prefix_before_cue": tuple(range(cue)), "cue": (cue,),
        "post_cue_before_subject": tuple(range(cue+1, query-1)),
        "subject_determiner": (query-1,), "final_query": (query,)}
    flat = tuple(position for name in GROUPS for position in groups[name])
    if tuple(sorted(flat)) != tuple(range(query+1)) or len(flat) != len(set(flat)):
        raise RuntimeError("destination groups do not partition causal positions")
    return groups


def run_destinations(backend, batch, base_hidden, delta, source_positions, base_raw,
                     destination_positions, *, actuate=True):
    handles = []
    if actuate:
        handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
            head_atlas.converter.actuation_hook(base_hidden, delta, source_positions)))
    n_head, head_dim = int(backend.model.config.n_head), int(backend.model.config.n_embd//backend.model.config.n_head)
    def patch(_module, arguments):
        raw = arguments[0]
        changed, base = raw.clone().view(raw.shape[0], raw.shape[1], n_head, head_dim), base_raw.view(raw.shape[0], raw.shape[1], n_head, head_dim)
        for i, selected in enumerate(destination_positions):
            for destination in selected:
                for head in (1, 4):
                    changed[i, destination, head] = base[i, destination, head].to(changed)
        return (changed.reshape_as(raw),)+tuple(arguments[1:])
    handles.append(backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(patch))
    try: return backend.native(batch, capture=True)
    finally:
        for handle in handles: handle.remove()


def main():
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER,
        "capability": CAPABILITY, "builder": BUILDER, "head_runner": HEAD_RUNNER,
        "weight_instrument": WEIGHT_INSTRUMENT, "subspace": weight.SUBSPACE,
        "family_runner": weight.FAMILY_RUNNER, "overlap_runner": weight.OVERLAP_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("destination-atlas authority changed")
    prior, parent, capability, subspace = [json.loads(path.read_text()) for path in (PRIOR, PARENT, CAPABILITY, weight.SUBSPACE)]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows() if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    source_positions = [weight.postcue_positions(row) for row in rows]
    partitions = [destination_partition(row) for row in rows]
    if prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen" or len(rows) != 27:
        raise RuntimeError("parent terminal or manifest population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "groups": list(GROUPS),
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend, torch = producer.Bilin18TorchBackend.load("cuda"), None
    torch = backend.torch
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float()) for layer in range(12, 18))
    modes, orientation_error, _wrong = overlap.residual_modes(backend, family[8], gain)
    s = torch.linalg.qr(modes, mode="reduced").Q
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_hidden_output, base_capture = weight.capture_mlp8(backend, base_batch)
    _donor_output, donor_capture = weight.capture_mlp8(backend, donor_batch)
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _singular, vh = torch.linalg.svd(s.T@down, full_matrices=False)
    full_delta = donor_capture["hidden"].float()-base_capture["hidden"].float()
    complement = full_delta-weight.project(full_delta, vh, vh.shape[0])
    base_output, base_raw = head_atlas.capture_attn9(backend, base_batch)
    live_output, live_raw = head_atlas.capture_attn9(backend, base_batch, base_capture["hidden"], complement, source_positions)
    arm_positions = {group: [partition[group] for partition in partitions] for group in GROUPS}
    arm_positions["all"] = [tuple(range(int(row["base_semantic_position"])+1)) for row in rows]
    outputs = {name: run_destinations(backend, base_batch, base_capture["hidden"], complement,
        source_positions, base_raw, positions) for name, positions in arm_positions.items()}
    self_output = run_destinations(backend, base_batch, base_capture["hidden"], complement,
        source_positions, base_raw, arm_positions["all"], actuate=False)
    forwards, evaluations = 11, 11*len(rows)
    state = lambda output: head_atlas.converter.state(output, rows, torch, backend.device)
    base18, base_hidden18, live18, self18 = state(base_output), state(base_hidden_output), state(live_output), state(self_output)
    states = {name: state(output) for name, output in outputs.items()}
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda value: das.head_logits(backend, value)[index, answers]-das.head_logits(backend, value)[index, foils]
    base_margin, live_margin = margin(base18), margin(live18)
    live_effect, live_coord = live_margin-base_margin, (live18-base18)@s
    metrics = {panel: {} for panel in ("A1", "A2")}
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device) for panel in ("A1", "A2")}
    for panel, mask in masks.items():
        for name, value in states.items():
            removed_e, removed_c = (live_margin-margin(value))[mask], ((live18-value)@s)[mask]
            metrics[panel][name] = {"absolute_behavior_fraction": float(removed_e.abs().mean()/live_effect[mask].abs().mean()),
                "signed_behavior_fraction": float(removed_e.mean()/live_effect[mask].mean()),
                "behavior_cosine": cosine(removed_e, live_effect[mask]),
                "q8_norm_fraction": float(removed_c.norm()/live_coord[mask].norm()),
                "q8_cosine": cosine(removed_c.reshape(-1), live_coord[mask].reshape(-1))}
    prefix_raw = max(float((live_raw[i, list(partitions[i]["prefix_before_cue"])]
        - base_raw[i, list(partitions[i]["prefix_before_cue"])]).abs().max())
        if partitions[i]["prefix_before_cue"] else 0.0 for i in range(len(rows)))
    material = [group for group in GROUPS if all(metrics[p][group]["absolute_behavior_fraction"] >= .20
        and metrics[p][group]["q8_norm_fraction"] >= .20 for p in ("A1", "A2"))]
    replay_error = max(abs(metrics[p]["all"][key]-parent["metrics"][p]["removals"]["h1h4"][parent_key])
        for p in ("A1", "A2") for key, parent_key in (("absolute_behavior_fraction", "absolute_behavior_fraction_of_live"),
        ("q8_norm_fraction", "q8_norm_fraction_of_live")))
    identity_error = max(float((base_hidden18-base18).abs().max()), float((self18-base18).abs().max()))
    finite = all(math.isfinite(value) for p in metrics.values() for row in p.values() for value in row.values())
    pred_a = bool(orientation_error <= 1e-6 and identity_error <= 1e-4 and finite and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    pred_b = replay_error <= .001
    pred_c = bool(material)
    pred_d = bool(prefix_raw <= 1e-6 and all(metrics[p]["prefix_before_cue"]["absolute_behavior_fraction"] <= 1e-6
        and metrics[p]["prefix_before_cue"]["q8_norm_fraction"] <= 1e-6 for p in ("A1", "A2")))
    pred_e = bool(len(material) <= 2 and all(sum(metrics[p][g]["absolute_behavior_fraction"] for g in material)
        >= .75*metrics[p]["all"]["absolute_behavior_fraction"] for p in ("A1", "A2")))
    predictions = {"pred_a_authority_partition_self_clamp_finiteness_and_price": pred_a,
        "pred_b_all_destinations_replay_confirmed_h1h4_removal": pred_b,
        "pred_c_at_least_one_proper_destination_is_material": pred_c,
        "pred_d_causal_prefix_destination_is_zero": pred_d,
        "pred_e_destination_program_is_compressive": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "iswas_mlp8_complement_attn9_h1h4_destination_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter()-started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_base_self_clamp_resid18_max_abs": identity_error,
            "all_destination_parent_replay_max_abs": replay_error,
            "prefix_raw_h1h4_response_delta_max_abs": prefix_raw, "rows": len(rows)},
        "metrics": metrics, "stable_material_destinations": material,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "metrics",
        "stable_material_destinations", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
