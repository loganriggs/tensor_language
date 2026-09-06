#!/usr/bin/env python3
"""Compile the fully isolated MLP8-to-auxiliary-reader residual skip."""

# BQGATE: EXPERIMENT pred_a_authority_full_clamp_closure_finiteness_and_price pred_b_exact_lambda_rms_cv_response_compiler pred_c_compiled_full_vocab_arm_replays_exact pred_d_direct_skip_is_material_in_both_panels pred_e_zero_fit_literal_weight_program
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_source_destination_eval as attention_eval
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_mlp8_auxiliary_postcue_local_weight_compiler_v1 as local
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_auxiliary_direct_skip_weight_compiler_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_cv_greedy_mlp_program_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_auxiliary_cv_greedy_mlp_program_v1.py"
ATLAS = ROOT / "circuits/followups/iswas_mlp8_auxiliary_cv_upstream_module_atlas_v2_result.json"
ATLAS_RUNNER = ROOT / "ops/run_iswas_mlp8_auxiliary_cv_upstream_module_atlas_v2.py"
LOCAL = ROOT / "circuits/followups/iswas_mlp8_auxiliary_postcue_local_weight_compiler_v1_result.json"
LOCAL_RUNNER = ROOT / "ops/run_iswas_mlp8_auxiliary_postcue_local_weight_compiler_v1.py"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_direct_skip_weight_compiler_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_auxiliary_direct_skip_weight_compiler_v1"
LAYERS, SELECTED = (11, 15), {11: (1, 3), 15: (5,)}
CLAMP_ATTN, CLAMP_MLP = tuple(range(9, 15)), tuple(range(9, 15))
EXPECTED = {
    "prior": "6011ca4825ec6a23674e822c1dc8e47fed184d7185fc23b2064b99a31205b5cc",
    "parent": "7d530f1fab59de9a26b13905629bdfff847339b6b171a3a95242689a241c85d9",
    "parent_runner": "310a093f843f4199d32e479887eb0e3f5e3f6ccd3ae6487620de55e24331bbb3",
    "atlas": "da3bd0473cf259537f5d70f1dff0ffe513bd5a7b420f6b6317259b522e4a3e67",
    "atlas_runner": "ce3df585dea1cc20bacada055bf9926c0752839748075daec2f85759474ec63b",
    "local": "57717c1f38013943273db9eb4076ad5d3191b72d8b14e85fa333b8239650cf34",
    "local_runner": "81d01c6594d80c84d500e14bf4dd4896edb0e81bc38fa25bb6596f4abf4c27ce",
    "capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
    "builder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2"
}
MAX_FORWARDS, MAX_EVALUATIONS = 8, 240


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def utc_now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
def cosine(x, y):
    denominator = float(x.norm() * y.norm())
    return float((x * y).sum()) / denominator if denominator else 0.0


def forward_program(backend, batch, *, base_modules=None, base_readers=None,
                    base_hidden=None, hidden_delta=None, positions=None, arm=None):
    """Exact model forward with explicit complete-module clamps and optional reader arm."""
    torch, F, model = backend.torch, backend.F, backend.model
    tokens, lengths = backend._tensor_batch(batch)
    heads, width = int(model.config.n_head), int(model.config.n_embd)
    head_dim = width // heads
    modules, readers = {"attn": {}, "mlp": {}}, {}
    handle = None
    if hidden_delta is not None:
        handle = model.transformer.h[8].mlp.Down.register_forward_pre_hook(
            converter.actuation_hook(base_hidden, hidden_delta, positions))
    try:
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (width,))
            x0, v1 = x, None
            for layer, block in enumerate(model.transformer.h):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                current = F.rms_norm(live, (width,))
                if layer in LAYERS:
                    pattern, value, reconstructed = attention_eval._attention_terms(
                        backend, block.attn, current, v1)
                    readers[layer] = {"z": live.detach().clone(), "current": current.detach().clone(),
                        "pattern": pattern.detach().clone(), "value": value.detach().clone(),
                        "raw": reconstructed.detach().clone()}
                attention, v1 = block.attn(current, v1)
                raw = attention
                if layer in CLAMP_ATTN and base_modules is not None:
                    raw = base_modules["attn"][layer].clone()
                if layer in LAYERS and arm is not None:
                    shaped = raw.clone().view(raw.shape[0], raw.shape[1], heads, head_dim)
                    base_shaped = base_readers[layer]["raw"].view_as(shaped)
                    delta = arm[layer]
                    for i, query in enumerate(batch.semantic_positions):
                        for head in SELECTED[layer]:
                            shaped[i, int(query), head] = (base_shaped[i, int(query), head].float()
                                + delta[i, int(query), head].float()).to(shaped)
                    raw = shaped.reshape_as(raw)
                modules["attn"][layer] = attention.detach().clone()
                x = live + raw
                mlp = block.mlp(F.rms_norm(x, (width,)))
                modules["mlp"][layer] = mlp.detach().clone()
                if layer in CLAMP_MLP and base_modules is not None:
                    mlp = base_modules["mlp"][layer]
                x = x + mlp
            logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (width,))) / 30.0)
            state = torch.stack([x[i, length - 1].float() for i, length in enumerate(lengths)])
            final_logits = torch.stack([logits[i, length - 1].float() for i, length in enumerate(lengths)])
    finally:
        if handle is not None: handle.remove()
    return state, final_logits, modules, readers


def main():
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER,
        "atlas": ATLAS, "atlas_runner": ATLAS_RUNNER, "local": LOCAL,
        "local_runner": LOCAL_RUNNER, "capability": CAPABILITY, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("direct-skip compiler authority changed")
    prior, parent, atlas, local_result, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, PARENT, ATLAS, LOCAL, CAPABILITY, weight.SUBSPACE)]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
        if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    positions = [weight.postcue_positions(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen"
            or atlas.get("terminal") != "screen" or local_result.get("terminal") != "screen"
            or len(rows) != 30):
        raise RuntimeError("parent decision or population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "clamped_attention_layers": list(CLAMP_ATTN), "clamped_mlp_layers": list(CLAMP_MLP),
        "arms": ["empty", "exact", "compiled"], "model_forwards_max": MAX_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS, "fit_updates": 0,
        "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, F = backend.torch, backend.F
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    final_gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
        for layer in range(12, 18))
    modes, orientation_error, _wrong = overlap.residual_modes(backend, family[8], final_gain)
    q8 = torch.linalg.qr(modes, mode="reduced").Q
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    _base_hidden_output, base_hidden_capture = weight.capture_mlp8(backend, base_batch)
    _donor_hidden_output, donor_hidden_capture = weight.capture_mlp8(backend, donor_batch)
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _singular, vh = torch.linalg.svd(q8.T @ down, full_matrices=False)
    hidden_full = donor_hidden_capture["hidden"].float() - base_hidden_capture["hidden"].float()
    complement = hidden_full - weight.project(hidden_full, vh, vh.shape[0])
    down_delta = torch.nn.functional.linear(complement, down, None)
    masked_down = torch.zeros_like(down_delta)
    for i, selected in enumerate(positions): masked_down[i, list(selected)] = down_delta[i, list(selected)]

    base_state, base_logits, base_modules, base_readers = forward_program(backend, base_batch)
    direct_state, direct_logits, _modules, direct_readers = forward_program(
        backend, base_batch, base_modules=base_modules, base_readers=base_readers,
        base_hidden=base_hidden_capture["hidden"], hidden_delta=complement, positions=positions)
    zero = {layer: torch.zeros_like(base_readers[layer]["raw"].view(
        len(rows), base_readers[layer]["raw"].shape[1], int(backend.model.config.n_head), -1))
        for layer in LAYERS}
    exact, compiled = {layer: zero[layer].clone() for layer in LAYERS}, {layer: zero[layer].clone() for layer in LAYERS}
    diagnostics, max_z, max_current, max_value, max_response, max_cproj = {}, 0., 0., 0., 0., 0.
    for layer in LAYERS:
        gain = math.prod(float(backend.model.transformer.h[k].lambdas[0].detach().float())
            for k in range(9, layer + 1))
        predicted_z = base_readers[layer]["z"].float() + gain * masked_down
        predicted_current = F.rms_norm(predicted_z, (int(backend.model.config.n_embd),))
        attention = backend.model.transformer.h[layer].attn
        shape = base_readers[layer]["value"].shape
        predicted_direct = attention.c_v(predicted_current).view(shape).float()
        base_direct = attention.c_v(base_readers[layer]["current"]).view(shape).float()
        predicted_value_delta = predicted_direct - base_direct
        exact_value_delta = direct_readers[layer]["value"].float() - base_readers[layer]["value"].float()
        for i, row in enumerate(rows):
            query = int(row["base_semantic_position"])
            source_index = torch.as_tensor(local.source_partition(row)["post_cue_before_subject"], device=backend.device)
            for head in SELECTED[layer]:
                pattern = base_readers[layer]["pattern"][i, head, query].float().index_select(0, source_index)
                exact[layer][i, query, head] = torch.einsum("s,sd->d", pattern,
                    exact_value_delta[i].index_select(0, source_index)[:, head])
                compiled[layer][i, query, head] = torch.einsum("s,sd->d", pattern,
                    predicted_value_delta[i].index_select(0, source_index)[:, head])
        cproj = attention.c_proj.weight.detach().float()
        head_dim = int(backend.model.config.n_embd // backend.model.config.n_head)
        exact_write = sum(torch.nn.functional.linear(exact[layer][:, :, head],
            cproj[:, head * head_dim:(head + 1) * head_dim], None) for head in SELECTED[layer])
        compiled_write = sum(torch.nn.functional.linear(compiled[layer][:, :, head],
            cproj[:, head * head_dim:(head + 1) * head_dim], None) for head in SELECTED[layer])
        z_error = float((predicted_z - direct_readers[layer]["z"].float()).abs().max())
        current_error = float((predicted_current - direct_readers[layer]["current"].float()).abs().max())
        value_error = float((predicted_value_delta - exact_value_delta).abs().max())
        response_error = float((compiled[layer] - exact[layer]).abs().max())
        cproj_error = float((compiled_write - exact_write).abs().max())
        max_z, max_current, max_value = max(max_z, z_error), max(max_current, current_error), max(max_value, value_error)
        max_response, max_cproj = max(max_response, response_error), max(max_cproj, cproj_error)
        diagnostics[str(layer)] = {"lambda_product": gain, "pre_rms_max_abs": z_error,
            "current_max_abs": current_error, "cv_value_max_abs": value_error,
            "response_max_abs": response_error, "cproj_max_abs": cproj_error,
            "exact_response_rms": float(torch.sqrt(exact[layer].square().mean()))}

    arm_states, arm_logits = {}, {}
    for name, arm in (("empty", zero), ("exact", exact), ("compiled", compiled)):
        arm_states[name], arm_logits[name], _m, _r = forward_program(backend, base_batch,
            base_modules=base_modules, base_readers=base_readers,
            base_hidden=base_hidden_capture["hidden"], hidden_delta=complement,
            positions=positions, arm=arm)
    forwards, evaluations = 7, 7 * len(rows)
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda logits: logits[index, answers] - logits[index, foils]
    parent_effect = margin(direct_logits) - margin(base_logits)
    exact_effect = margin(arm_logits["exact"]) - margin(arm_logits["empty"])
    compiled_effect = margin(arm_logits["compiled"]) - margin(arm_logits["empty"])
    exact_q8 = (arm_states["exact"] - arm_states["empty"]) @ q8
    compiled_q8 = (arm_states["compiled"] - arm_states["empty"]) @ q8
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)
        for panel in ("A1", "A2")}
    metrics = {}
    for panel, mask in masks.items():
        metrics[panel] = {
            "compiled_exact_behavior_cosine": cosine(compiled_effect[mask], exact_effect[mask]),
            "compiled_exact_behavior_relative_rmse": float(torch.sqrt((compiled_effect[mask]-exact_effect[mask]).square().mean()) / torch.sqrt(exact_effect[mask].square().mean())),
            "compiled_exact_q8_cosine": cosine(compiled_q8[mask].reshape(-1), exact_q8[mask].reshape(-1)),
            "compiled_exact_q8_relative_rmse": float(torch.sqrt((compiled_q8[mask]-exact_q8[mask]).square().mean()) / torch.sqrt(exact_q8[mask].square().mean())),
            "direct_reader_behavior_fraction_of_full_clamp_parent": float(exact_effect[mask].abs().mean() / parent_effect[mask].abs().mean()),
            "direct_reader_q8_rms": float(torch.sqrt(exact_q8[mask].square().mean()))}
    logit_error = float((arm_logits["compiled"] - arm_logits["exact"]).abs().max())
    state_relative = float((arm_states["compiled"] - arm_states["exact"]).norm() /
        (arm_states["exact"] - arm_states["empty"]).norm())
    finite = all(math.isfinite(value) for panel in metrics.values() for value in panel.values())
    pred_a = bool(orientation_error <= 1e-6 and finite and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    pred_b = max(max_z, max_current, max_value, max_response, max_cproj) <= .001
    pred_c = all(metrics[p]["compiled_exact_behavior_cosine"] >= .999 and metrics[p]["compiled_exact_behavior_relative_rmse"] <= .01
        and metrics[p]["compiled_exact_q8_cosine"] >= .999 and metrics[p]["compiled_exact_q8_relative_rmse"] <= .01 for p in masks) and logit_error <= 1e-4 and state_relative <= 1e-4
    pred_d = all(metrics[p]["direct_reader_behavior_fraction_of_full_clamp_parent"] >= .20
        and metrics[p]["direct_reader_q8_rms"] >= .01 for p in masks)
    pred_e = set(arm_states) == {"empty", "exact", "compiled"}
    predictions = {"pred_a_authority_full_clamp_closure_finiteness_and_price": pred_a,
        "pred_b_exact_lambda_rms_cv_response_compiler": pred_b,
        "pred_c_compiled_full_vocab_arm_replays_exact": pred_c,
        "pred_d_direct_skip_is_material_in_both_panels": pred_d,
        "pred_e_zero_fit_literal_weight_program": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "iswas_mlp8_auxiliary_direct_skip_weight_compiler_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter()-started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "compiled_exact_full_logit_max_abs": logit_error, "compiled_exact_resid18_relative_norm": state_relative,
            "rows": len(rows), "complete_attention9_clamped": True,
            "attention_clamps": list(CLAMP_ATTN), "mlp_clamps": list(CLAMP_MLP)},
        "layer_diagnostics": diagnostics, "metrics": metrics, "predictions": predictions,
        "terminal": terminal, "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "layer_diagnostics", "metrics", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
