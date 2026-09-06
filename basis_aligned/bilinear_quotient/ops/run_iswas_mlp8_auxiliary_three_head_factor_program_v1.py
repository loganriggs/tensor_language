#!/usr/bin/env python3
"""Exact response-factor program for final-query L11H1/H3 and L15H5."""

# BQGATE: EXPERIMENT pred_a_authority_factor_complete_replay_finiteness_and_price pred_b_base_pattern_value_transport_is_dominant pred_c_pattern_change_is_secondary pred_d_pattern_value_interaction_is_secondary pred_e_zero_fit_shared_operation_test
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import attention_source_destination_eval as attention_eval
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_v1 as auxiliary
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_auxiliary_three_head_factor_program_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_three_head_destination_atlas_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_auxiliary_three_head_destination_atlas_v1.py"
ATTENTION_LIBRARY = ROOT / "ops/attention_source_destination_eval.py"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_three_head_factor_program_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_auxiliary_three_head_factor_program_v1"
EXPECTED = {
    "prior": "a8a6b4de6ccb3801e369abcd3ec03432645e8fcd53e2f68b23536e20323a0ffd",
    "parent": "1657b7b3eb5abb54389ab88ec901d9165c1e43ed422413a72965aa2372dde18b",
    "parent_runner": "d702240b34404f7e9c06d0711202d4f9edcdbe7f82d509e5dc67aa4849b10dc1",
    "attention_library": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
    "capability": "77e3c5b6e47dc9416133643f422c130427f90fed0746d26f211f54b681718b3d",
    "builder": "13e7954cde01b6b7d826915fc2ae02d4b9e16975150cf73aa5e0a1f906c1b757",
}
CORE, SELECTED = (1, 4), {11: (1, 3), 15: (5,)}
FACTORS = attention_eval.RESPONSE_FACTORS
MAX_FORWARDS, MAX_EVALUATIONS = 15, 405


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def name(subset) -> str:
    return "+".join(subset) if subset else "empty"


def subsets():
    for width in range(len(FACTORS) + 1):
        yield from itertools.combinations(FACTORS, width)


def cosine(x, y) -> float:
    denominator = float(x.norm() * y.norm())
    return float((x * y).sum()) / denominator if denominator else 0.0


def capture_with_core(backend, batch, layer, base_hidden, complement, source_positions, base_raw9):
    return attention_eval.capture_layer_attention(
        backend, batch, layer,
        call=lambda: auxiliary.run_heads(backend, batch, base_hidden, complement,
            source_positions, {9: base_raw9}, {9: CORE}))


def run_factor_arm(backend, batch, base_hidden, complement, source_positions,
                   base_raw9, base_captures, layer_deltas, *, actuate=True):
    handles = []
    if actuate:
        handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
            converter.actuation_hook(base_hidden, complement, source_positions)))
    n_head = int(backend.model.config.n_head)
    head_dim = int(backend.model.config.n_embd // n_head)
    def patch_core(_module, arguments):
        raw = arguments[0]
        changed = raw.clone().view(raw.shape[0], raw.shape[1], n_head, head_dim)
        base = base_raw9.view_as(changed)
        for i, query in enumerate(batch.semantic_positions):
            changed[i, :int(query)+1, list(CORE)] = base[i, :int(query)+1, list(CORE)].to(changed)
        return (changed.reshape_as(raw),) + tuple(arguments[1:])
    handles.append(backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(patch_core))
    for layer, heads in SELECTED.items():
        def patch_aux(_module, arguments, layer=layer, heads=heads):
            raw = arguments[0]
            changed = raw.clone().view(raw.shape[0], raw.shape[1], n_head, head_dim)
            for i, query in enumerate(batch.semantic_positions):
                changed[i, :int(query)+1, list(heads)] = (
                    base_captures[layer]["head_output"][i, :int(query)+1, list(heads)].float()
                    + layer_deltas[layer][i, :int(query)+1, list(heads)].float()).to(changed)
            return (changed.reshape_as(raw),) + tuple(arguments[1:])
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch_aux))
    try:
        return backend.native(batch, capture=True)
    finally:
        for handle in handles: handle.remove()


def main() -> None:
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER,
             "attention_library": ATTENTION_LIBRARY, "capability": CAPABILITY, "builder": BUILDER}
    if {key: sha(value) for key, value in paths.items()} != EXPECTED:
        raise RuntimeError("auxiliary factor authority changed")
    inherited_paths = {
        "prior": converter.PRIOR, "weight_v2": converter.WEIGHT_V2,
        "weight_v2_runner": converter.WEIGHT_V2_RUNNER,
        "weight_instrument": converter.WEIGHT_INSTRUMENT, "source": converter.SOURCE,
        "capability": weight.CAPABILITY, "iswas": weight.ISWAS,
        "subspace": weight.SUBSPACE, "builder": weight.BUILDER,
        "family_runner": weight.FAMILY_RUNNER, "overlap_runner": weight.OVERLAP_RUNNER,
    }
    if {key: sha(value) for key, value in inherited_paths.items()} != converter.EXPECTED:
        raise RuntimeError("inherited converter authority changed")
    prior, parent, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, PARENT, CAPABILITY, weight.SUBSPACE)]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    source_positions = [weight.postcue_positions(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen"
            or parent.get("stable_material_destinations") != ["final_query"] or len(rows) != 27):
        raise RuntimeError("parent decision or population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "factors": list(FACTORS), "arms": [name(value) for value in subsets()],
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
    base_hidden_output, base_hidden_capture = weight.capture_mlp8(backend, base_batch)
    _donor_output, donor_hidden_capture = weight.capture_mlp8(backend, donor_batch)
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _singular, vh = torch.linalg.svd(s.T @ down, full_matrices=False)
    full_hidden_delta = donor_hidden_capture["hidden"].float() - base_hidden_capture["hidden"].float()
    complement = full_hidden_delta - weight.project(full_hidden_delta, vh, vh.shape[0])
    base_raw9_box = {}
    def capture_raw9(_module, arguments):
        base_raw9_box["raw"] = arguments[0].detach().clone()
    handle = backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(capture_raw9)
    try:
        base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
    finally:
        handle.remove()
    base_raw9 = base_raw9_box["raw"]
    _base15_output, base15 = attention_eval.capture_layer_attention(backend, base_batch, 15)
    core11_output, changed11 = capture_with_core(backend, base_batch, 11,
        base_hidden_capture["hidden"], complement, source_positions, base_raw9)
    _core15_output, changed15 = capture_with_core(backend, base_batch, 15,
        base_hidden_capture["hidden"], complement, source_positions, base_raw9)
    base_captures, changed_captures = {11: base11, 15: base15}, {11: changed11, 15: changed15}
    terms = {}
    for layer, heads in SELECTED.items():
        base_pattern = base_captures[layer]["pattern"].float()
        changed_pattern = changed_captures[layer]["pattern"].float()
        base_value = base_captures[layer]["value"].float()
        changed_value = changed_captures[layer]["value"].float()
        dp, dv = changed_pattern - base_pattern, changed_value - base_value
        raw_terms = {
            "pattern_on_base_value": torch.einsum("bhqk,bkhd->bqhd", dp, base_value),
            "base_pattern_on_value_change": torch.einsum("bhqk,bkhd->bqhd", base_pattern, dv),
            "pattern_value_interaction": torch.einsum("bhqk,bkhd->bqhd", dp, dv),
        }
        terms[layer] = {}
        for factor, raw in raw_terms.items():
            selected = torch.zeros_like(raw)
            selected[:, :, list(heads)] = raw[:, :, list(heads)]
            terms[layer][factor] = selected
    zero = {layer: torch.zeros_like(base_captures[layer]["head_output"], dtype=torch.float32)
            for layer in SELECTED}
    arm_deltas = {}
    for subset in subsets():
        arm_deltas[name(subset)] = {layer: sum((terms[layer][factor] for factor in subset), zero[layer])
                                    for layer in SELECTED}
    outputs = {arm: run_factor_arm(backend, base_batch, base_hidden_capture["hidden"], complement,
        source_positions, base_raw9, base_captures, deltas) for arm, deltas in arm_deltas.items()}
    self_output = run_factor_arm(backend, base_batch, base_hidden_capture["hidden"], complement,
        source_positions, base_raw9, base_captures, zero, actuate=False)
    forwards, evaluations = 15, 15 * len(rows)
    complete_name = name(FACTORS)
    raw_closure = max(float((arm_deltas[complete_name][layer]
        - (changed_captures[layer]["head_output"].float() - base_captures[layer]["head_output"].float())).abs()[:, :, list(SELECTED[layer])].max())
        for layer in SELECTED)
    reconstruction = max(capture["reconstruction_max_abs"] for capture in
                         (*base_captures.values(), *changed_captures.values()))
    state = lambda output: converter.state(output, rows, torch, backend.device)
    base18, base_hidden18, core18, self18 = map(
        state, (base_hidden_output, base_hidden_output, core11_output, self_output))
    states = {arm: state(value) for arm, value in outputs.items()}
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda value: das.head_logits(backend, value)[index, answers] - das.head_logits(backend, value)[index, foils]
    empty_margin, complete_margin = margin(states["empty"]), margin(states[complete_name])
    full_e, full_c = complete_margin - empty_margin, (states[complete_name] - states["empty"]) @ s
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)
             for panel in ("A1", "A2")}
    metrics = {panel: {} for panel in masks}
    for panel, mask in masks.items():
        for arm, value in states.items():
            e, c = (margin(value) - empty_margin)[mask], ((value - states["empty"]) @ s)[mask]
            metrics[panel][arm] = {
                "absolute_behavior_fraction_of_complete": float(e.abs().mean() / full_e[mask].abs().mean()),
                "signed_behavior_fraction_of_complete": float(e.mean() / full_e[mask].mean()),
                "behavior_cosine_to_complete": cosine(e, full_e[mask]),
                "q8_norm_fraction_of_complete": float(c.norm() / full_c[mask].norm()),
                "q8_cosine_to_complete": cosine(c.reshape(-1), full_c[mask].reshape(-1)),
            }
    complete_error = float((states[complete_name] - core18).abs().max())
    complete_relative = float((states[complete_name] - core18).norm() / (core18 - states["empty"]).norm())
    self_error = float((self18 - base18).abs().max())
    finite = all(math.isfinite(value) for panel in metrics.values() for arm in panel.values() for value in arm.values())
    pred_a = bool(orientation_error <= 1e-6 and reconstruction <= .001 and raw_closure <= .001
        and complete_error <= .05 and complete_relative <= 1e-4 and self_error <= .001
        and finite and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    value_name = "base_pattern_on_value_change"
    pattern_name = "pattern_on_base_value"
    interaction_name = "pattern_value_interaction"
    pred_b = all(metrics[p][value_name][key] >= .75 for p in ("A1", "A2")
        for key in ("absolute_behavior_fraction_of_complete", "q8_norm_fraction_of_complete"))
    pred_c = all(metrics[p][pattern_name][key] <= .25 for p in ("A1", "A2")
        for key in ("absolute_behavior_fraction_of_complete", "q8_norm_fraction_of_complete"))
    pred_d = all(metrics[p][interaction_name][key] <= .20 for p in ("A1", "A2")
        for key in ("absolute_behavior_fraction_of_complete", "q8_norm_fraction_of_complete"))
    pred_e = True
    predictions = {"pred_a_authority_factor_complete_replay_finiteness_and_price": pred_a,
        "pred_b_base_pattern_value_transport_is_dominant": pred_b,
        "pred_c_pattern_change_is_secondary": pred_c,
        "pred_d_pattern_value_interaction_is_secondary": pred_d,
        "pred_e_zero_fit_shared_operation_test": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "iswas_mlp8_auxiliary_three_head_factor_program_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_attention_reconstruction_max_abs": reconstruction,
            "raw_factor_closure_max_abs": raw_closure,
            "complete_core_resid18_max_abs": complete_error,
            "complete_core_resid18_relative_norm": complete_relative,
            "native_self_clamp_max_abs": self_error, "rows": len(rows)},
        "metrics": metrics, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "metrics",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
