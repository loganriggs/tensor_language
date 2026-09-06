#!/usr/bin/env python3
"""Exact source/factor and c_v/c_proj tensor account of the H1/H4 query converter."""

# BQGATE: EXPERIMENT pred_a_authority_partition_factor_native_weight_closure_finiteness_and_price pred_b_postcue_sources_dominate_query_conversion pred_c_base_pattern_value_transport_is_dominant pred_d_pattern_value_interaction_is_secondary pred_e_zero_fit_literal_weight_interface
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch.nn.functional as F
import attention_source_destination_eval as attention_eval
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_complement_attn9_head_converter_atlas_v1 as head_atlas
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_attn9_h1h4_query_source_factor_tensor_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_complement_attn9_h1h4_destination_atlas_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_complement_attn9_h1h4_destination_atlas_v1.py"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10.py"
ATTENTION_LIBRARY = ROOT / "ops/attention_source_destination_eval.py"
WEIGHT_INSTRUMENT = ROOT / "ops/run_iswas_shared_q8_mlp8_postcue_weight_modes_v1.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_attn9_h1h4_query_source_factor_tensor_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_attn9_h1h4_query_source_factor_tensor_v1"
RESULT_SCHEMA = "iswas_mlp8_attn9_h1h4_query_source_factor_tensor_result_v1"
COMPLETE_RESID18_ABS_TOLERANCE = .001
EXPECTED = {"prior": "d192ba0e2cbdd1043f165278e86f9768d1936440d17e11b95f303649cbd62662",
    "parent": "25c239436eb3cbf6fdd18237132bbd3a11683d977dd57dfe3877a2db78b95e7e",
    "parent_runner": "2fb6d8ed81e122161eb4ee8669b84639a3e26aa5859c2936ae3b39cb5f46ed3c",
    "capability": "77e3c5b6e47dc9416133643f422c130427f90fed0746d26f211f54b681718b3d",
    "builder": "13e7954cde01b6b7d826915fc2ae02d4b9e16975150cf73aa5e0a1f906c1b757",
    "attention_library": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
    "weight_instrument": "d82035e45f8818a7de73024500592f1677f887e8be5d8a2e617c28afc6c1238c",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "overlap_runner": "15f2e1119c4df7c29aaf51c615ec92d87555d65efce029f16a89478212f29af1"}
GROUPS = ("prefix_before_cue", "cue", "post_cue_before_subject", "subject_determiner", "self")
FACTORS = attention_eval.RESPONSE_FACTORS
MAX_FORWARDS, MAX_EVALUATIONS = 29, 783


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def utc_now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
def cosine(x, y):
    denominator = float(x.norm()*y.norm())
    return float((x*y).sum())/denominator if denominator else 0.0


def source_partition(row):
    differences = [i for i, pair in enumerate(zip(row["base_ids"], row["donor_ids"])) if pair[0] != pair[1]]
    if len(differences) != 1: raise RuntimeError("source tensor requires one aligned cue")
    cue, query = differences[0], int(row["base_semantic_position"])
    groups = {"prefix_before_cue": tuple(range(cue)), "cue": (cue,),
        "post_cue_before_subject": tuple(range(cue+1, query-1)),
        "subject_determiner": (query-1,), "self": (query,)}
    flat = tuple(position for name in GROUPS for position in groups[name])
    if tuple(sorted(flat)) != tuple(range(query+1)) or len(flat) != len(set(flat)) or any(not groups[g] for g in GROUPS):
        raise RuntimeError("source groups are not a nonempty causal partition")
    return groups


def capture_attention9(backend, batch, base_hidden=None, delta=None, positions=None):
    projected, handles = {}, []
    if delta is not None:
        handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
            head_atlas.converter.actuation_hook(base_hidden, delta, positions)))
    def capture_output(_module, _arguments, output):
        projected["output"] = output.detach().clone()
    handles.append(backend.model.transformer.h[9].attn.c_proj.register_forward_hook(capture_output))
    try:
        output, capture = attention_eval.capture_layer_attention(
            backend, batch, 9, call=lambda: backend.native(batch, capture=True))
    finally:
        for handle in handles: handle.remove()
    return output, capture, projected["output"]


def run_delta(backend, batch, base_hidden, complement, source_positions, base_capture, delta):
    handles = [backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
        head_atlas.converter.actuation_hook(base_hidden, complement, source_positions))]
    n_head, head_dim = int(backend.model.config.n_head), int(backend.model.config.n_embd//backend.model.config.n_head)
    def patch(_module, arguments):
        raw = arguments[0]
        changed = raw.clone().view(raw.shape[0], raw.shape[1], n_head, head_dim)
        for i, query in enumerate(batch.semantic_positions):
            for head in (1, 4):
                changed[i, int(query), head] = (base_capture["head_output"][i, int(query), head].float()
                    + delta[i, int(query), head].float()).to(changed)
        return (changed.reshape_as(raw),)+tuple(arguments[1:])
    handles.append(backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(patch))
    try: return backend.native(batch, capture=True)
    finally:
        for handle in handles: handle.remove()


def main():
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER,
        "capability": CAPABILITY, "builder": BUILDER, "attention_library": ATTENTION_LIBRARY,
        "weight_instrument": WEIGHT_INSTRUMENT, "subspace": weight.SUBSPACE,
        "family_runner": weight.FAMILY_RUNNER, "overlap_runner": weight.OVERLAP_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED: raise RuntimeError("source-factor authority changed")
    prior, parent, capability, subspace = [json.loads(path.read_text()) for path in (PRIOR, PARENT, CAPABILITY, weight.SUBSPACE)]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows() if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    source_positions, partitions = [weight.postcue_positions(row) for row in rows], [source_partition(row) for row in rows]
    if prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen" or parent.get("stable_material_destinations") != ["final_query"] or len(rows) != 27:
        raise RuntimeError("parent decision or population changed")
    arm_names = (["empty"]+[f"cell:{group}:{factor}" for group in GROUPS for factor in FACTORS]
        +[f"group:{group}" for group in GROUPS]+[f"factor:{factor}" for factor in FACTORS]+["complete"])
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "arms": arm_names,
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend, torch = producer.Bilin18TorchBackend.load("cuda"), None; torch = backend.torch
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float()) for layer in range(12, 18))
    modes, orientation_error, _wrong = overlap.residual_modes(backend, family[8], gain); s = torch.linalg.qr(modes, mode="reduced").Q
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_hidden_output, base_hidden_capture = weight.capture_mlp8(backend, base_batch)
    _donor_output, donor_hidden_capture = weight.capture_mlp8(backend, donor_batch)
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _singular, vh = torch.linalg.svd(s.T@down, full_matrices=False)
    full_hidden_delta = donor_hidden_capture["hidden"].float()-base_hidden_capture["hidden"].float()
    complement = full_hidden_delta-weight.project(full_hidden_delta, vh, vh.shape[0])
    base_output, base_capture, base_projected = capture_attention9(backend, base_batch)
    live_output, live_capture, live_projected = capture_attention9(backend, base_batch,
        base_hidden_capture["hidden"], complement, source_positions)
    destinations = [int(row["base_semantic_position"]) for row in rows]
    cells = {}
    for group in GROUPS:
        terms = attention_eval.attention_response_factor_deltas(base_capture, live_capture,
            destinations, [partition[group] for partition in partitions], selected_heads=(1, 4))
        for factor in FACTORS: cells[(group, factor)] = terms[factor]
    zero = torch.zeros_like(base_capture["head_output"], dtype=torch.float32)
    arm_delta = {"empty": zero}
    arm_delta.update({f"cell:{g}:{f}": cells[(g, f)] for g in GROUPS for f in FACTORS})
    arm_delta.update({f"group:{g}": sum(cells[(g, f)] for f in FACTORS) for g in GROUPS})
    arm_delta.update({f"factor:{f}": sum(cells[(g, f)] for g in GROUPS) for f in FACTORS})
    arm_delta["complete"] = sum(cells.values())
    outputs = {name: run_delta(backend, base_batch, base_hidden_capture["hidden"], complement,
        source_positions, base_capture, arm_delta[name]) for name in arm_names}
    forwards, evaluations = 29, 29*len(rows)
    raw_expected = live_capture["head_output"].float()-base_capture["head_output"].float()
    selected_expected = torch.zeros_like(raw_expected)
    for i, query in enumerate(destinations): selected_expected[i, query, [1, 4]] = raw_expected[i, query, [1, 4]]
    raw_closure = float((arm_delta["complete"]-selected_expected).abs().max())
    cproj_weight = backend.model.transformer.h[9].attn.c_proj.weight.detach().float()
    compiled = F.linear(selected_expected.reshape(selected_expected.shape[0], selected_expected.shape[1], -1), cproj_weight, None)
    head_dim = int(backend.model.config.n_embd//backend.model.config.n_head)
    compiled_by_head = sum(F.linear(selected_expected[:, :, head],
        cproj_weight[:, head*head_dim:(head+1)*head_dim], None) for head in (1, 4))
    cproj_closure = float((compiled-compiled_by_head).abs().max())
    state = lambda output: head_atlas.converter.state(output, rows, torch, backend.device)
    base18, base_hidden18, live18 = state(base_output), state(base_hidden_output), state(live_output)
    states = {name: state(output) for name, output in outputs.items()}
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda value: das.head_logits(backend, value)[index, answers]-das.head_logits(backend, value)[index, foils]
    empty_margin, full_margin = margin(states["empty"]), margin(states["complete"])
    full_effect, full_coord = full_margin-empty_margin, (states["complete"]-states["empty"])@s
    metrics = {panel: {} for panel in ("A1", "A2")}
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device) for panel in ("A1", "A2")}
    for panel, mask in masks.items():
        for name in arm_names:
            effect, coord = (margin(states[name])-empty_margin)[mask], ((states[name]-states["empty"])@s)[mask]
            metrics[panel][name] = {"absolute_behavior_fraction_of_complete": float(effect.abs().mean()/full_effect[mask].abs().mean()),
                "signed_behavior_fraction_of_complete": float(effect.mean()/full_effect[mask].mean()),
                "behavior_cosine_to_complete": cosine(effect, full_effect[mask]),
                "q8_norm_fraction_of_complete": float(coord.norm()/full_coord[mask].norm()),
                "q8_cosine_to_complete": cosine(coord.reshape(-1), full_coord[mask].reshape(-1))}
    complete_replay = float((states["complete"]-live18).abs().max())
    complete_replay_relative = float((states["complete"]-live18).norm()
        / (live18-states["empty"]).norm())
    complete_logit_replay = float((das.head_logits(backend, states["complete"])
        - das.head_logits(backend, live18)).abs().max())
    identity_error = float((base_hidden18-base18).abs().max())
    finite = all(math.isfinite(value) for p in metrics.values() for row in p.values() for value in row.values())
    pred_a = bool(orientation_error <= 1e-6 and max(base_capture["reconstruction_max_abs"], live_capture["reconstruction_max_abs"]) <= .001
        and raw_closure <= .001 and cproj_closure <= .001
        and complete_replay <= COMPLETE_RESID18_ABS_TOLERANCE
        and complete_replay_relative <= 1e-4 and complete_logit_replay <= 1e-4
        and identity_error <= .001
        and finite and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    pred_b = all(metrics[p]["group:post_cue_before_subject"][key] >= .75 for p in ("A1", "A2")
        for key in ("absolute_behavior_fraction_of_complete", "q8_norm_fraction_of_complete"))
    value_name = "factor:base_pattern_on_value_change"
    pred_c = all(metrics[p][value_name][key] >= .75 for p in ("A1", "A2")
        for key in ("absolute_behavior_fraction_of_complete", "q8_norm_fraction_of_complete"))
    interaction_name = "factor:pattern_value_interaction"
    pred_d = all(metrics[p][interaction_name][key] <= .20 for p in ("A1", "A2")
        for key in ("absolute_behavior_fraction_of_complete", "q8_norm_fraction_of_complete"))
    pred_e = True
    predictions = {"pred_a_authority_partition_factor_native_weight_closure_finiteness_and_price": pred_a,
        "pred_b_postcue_sources_dominate_query_conversion": pred_b,
        "pred_c_base_pattern_value_transport_is_dominant": pred_c,
        "pred_d_pattern_value_interaction_is_secondary": pred_d,
        "pred_e_zero_fit_literal_weight_interface": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    value_delta = live_capture["value"].float()-base_capture["value"].float()
    result = {"schema": RESULT_SCHEMA,
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter()-started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_attention_reconstruction_max_abs": max(base_capture["reconstruction_max_abs"], live_capture["reconstruction_max_abs"]),
            "raw_factor_closure_max_abs": raw_closure, "cproj_weight_replay_max_abs": cproj_closure,
            "complete_arm_live_resid18_max_abs": complete_replay,
            "complete_arm_live_resid18_relative_norm": complete_replay_relative,
            "complete_arm_live_full_logit_max_abs": complete_logit_replay,
            "native_base_identity_max_abs": identity_error, "rows": len(rows)},
        "weight_tensor": {"postcue_h1h4_value_delta_rms": float(torch.sqrt(torch.stack([
            value_delta[i, list(source_positions[i])][:, [1, 4]].square().mean() for i in range(len(rows))]).mean())),
            "compiled_query_write_rms": float(torch.sqrt(compiled[index, destinations].square().mean())),
            "compiled_query_q8_write_rms": float(torch.sqrt((compiled[index, destinations]@s).square().mean()))},
        "metrics": metrics, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "weight_tensor",
        "metrics", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
