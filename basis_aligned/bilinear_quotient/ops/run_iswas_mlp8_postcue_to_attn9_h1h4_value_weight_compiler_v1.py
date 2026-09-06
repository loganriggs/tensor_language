#!/usr/bin/env python3
"""Literal MLP8 Down -> block9 RMS -> H1/H4 c_v/c_proj value compiler."""

# BQGATE: EXPERIMENT pred_a_authority_residual_rms_cv_cproj_closure_finiteness_and_price pred_b_compiled_value_response_matches_exact_causal_effect pred_c_compiled_full_vocabulary_intervention_replays pred_d_initial_value_branch_is_unchanged pred_e_zero_fit_literal_weight_program
from datetime import datetime, timezone
import hashlib
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
import run_iswas_mlp8_attn9_h1h4_query_source_factor_tensor_v1 as source
import run_iswas_mlp8_complement_attn9_head_converter_atlas_v1 as head_atlas
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_postcue_to_attn9_h1h4_value_weight_compiler_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_attn9_h1h4_query_source_factor_tensor_v2_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_attn9_h1h4_query_source_factor_tensor_v2.py"
SOURCE_INSTRUMENT = ROOT / "ops/run_iswas_mlp8_attn9_h1h4_query_source_factor_tensor_v1.py"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10.py"
ATTENTION_LIBRARY = ROOT / "ops/attention_source_destination_eval.py"
WEIGHT_INSTRUMENT = ROOT / "ops/run_iswas_shared_q8_mlp8_postcue_weight_modes_v1.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_postcue_to_attn9_h1h4_value_weight_compiler_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_postcue_to_attn9_h1h4_value_weight_compiler_v1"
EXPECTED = {"prior": "88dd0c350d19d63286dbba0a4d99e8bc3d1529133054534b8e84d20f04822b48",
    "parent": "2d575d2c6e50d0cb716e8293576815b30cca0e4f84d07549cb8042ff0d887840",
    "parent_runner": "a44db69e3b7ed6beca6f9365518ce47d1ac7c380109285e4c480de65a1881643",
    "source_instrument": "7eca9a08bd913831339b037da77c09f1a9bb398ed993e209806cdaefed565aff",
    "capability": "77e3c5b6e47dc9416133643f422c130427f90fed0746d26f211f54b681718b3d",
    "builder": "13e7954cde01b6b7d826915fc2ae02d4b9e16975150cf73aa5e0a1f906c1b757",
    "attention_library": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
    "weight_instrument": "d82035e45f8818a7de73024500592f1677f887e8be5d8a2e617c28afc6c1238c",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "overlap_runner": "15f2e1119c4df7c29aaf51c615ec92d87555d65efce029f16a89478212f29af1"}
MAX_FORWARDS, MAX_EVALUATIONS = 7, 189


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def utc_now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
def cosine(x, y):
    denominator = float(x.norm()*y.norm())
    return float((x*y).sum())/denominator if denominator else 0.0


def capture_compiler(backend, batch, base_hidden=None, complement=None, positions=None):
    raw, handles = {}, []
    original = backend.F.rms_norm
    width = int(backend.model.config.n_embd)
    def recording(value, normalized_shape, *args, **kwargs):
        output = original(value, normalized_shape, *args, **kwargs)
        shape = tuple(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else (normalized_shape,)
        if shape == (width,): raw["last_width_input"] = value.detach().clone()
        return output
    def capture_attention_input(_module, arguments):
        raw["z9"] = raw["last_width_input"].detach().clone()
        raw["current"] = arguments[0].detach().clone()
        raw["v1"] = (arguments[1].detach().clone() if len(arguments) > 1 and arguments[1] is not None
                     else None)
    backend.F.rms_norm = recording
    handles.append(backend.model.transformer.h[9].attn.register_forward_pre_hook(capture_attention_input))
    if complement is not None:
        handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
            head_atlas.converter.actuation_hook(base_hidden, complement, positions)))
    try:
        output, capture = attention_eval.capture_layer_attention(
            backend, batch, 9, call=lambda: backend.native(batch, capture=True))
    finally:
        for handle in handles: handle.remove()
        backend.F.rms_norm = original
    return output, capture, raw


def effect_metrics(torch, backend, rows, base_state, empty_state, exact_state, compiled_state, s):
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    logits = {name: das.head_logits(backend, value) for name, value in
        {"empty": empty_state, "exact": exact_state, "compiled": compiled_state}.items()}
    margins = {name: value[index, answers]-value[index, foils] for name, value in logits.items()}
    exact_effect, compiled_effect = margins["exact"]-margins["empty"], margins["compiled"]-margins["empty"]
    exact_q8, compiled_q8 = (exact_state-empty_state)@s, (compiled_state-empty_state)@s
    metrics = {}
    for panel in ("A1", "A2"):
        mask = torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)
        behavior_rmse = torch.sqrt(((compiled_effect[mask]-exact_effect[mask])**2).mean())
        q8_rmse = torch.sqrt(((compiled_q8[mask]-exact_q8[mask])**2).mean())
        metrics[panel] = {"behavior_cosine": cosine(compiled_effect[mask], exact_effect[mask]),
            "behavior_relative_rmse": float(behavior_rmse/torch.sqrt(exact_effect[mask].square().mean())),
            "q8_cosine": cosine(compiled_q8[mask].reshape(-1), exact_q8[mask].reshape(-1)),
            "q8_relative_rmse": float(q8_rmse/torch.sqrt(exact_q8[mask].square().mean()))}
    return metrics, float((logits["compiled"]-logits["exact"]).abs().max())


def main():
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER,
        "source_instrument": SOURCE_INSTRUMENT, "capability": CAPABILITY, "builder": BUILDER,
        "attention_library": ATTENTION_LIBRARY, "weight_instrument": WEIGHT_INSTRUMENT,
        "subspace": weight.SUBSPACE, "family_runner": weight.FAMILY_RUNNER,
        "overlap_runner": weight.OVERLAP_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED: raise RuntimeError("value-compiler authority changed")
    prior, parent, capability, subspace = [json.loads(path.read_text()) for path in (PRIOR, PARENT, CAPABILITY, weight.SUBSPACE)]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows() if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    positions, partitions = [weight.postcue_positions(row) for row in rows], [source.source_partition(row) for row in rows]
    if prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen" or len(rows) != 27:
        raise RuntimeError("parent terminal or population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "arms": ["empty", "exact_value", "compiled_value"],
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
    mlp8 = backend.model.transformer.h[8].mlp
    down = mlp8.Down.weight.detach().float(); _u, _singular, vh = torch.linalg.svd(s.T@down, full_matrices=False)
    full_delta = donor_hidden_capture["hidden"].float()-base_hidden_capture["hidden"].float()
    complement = full_delta-weight.project(full_delta, vh, vh.shape[0])
    base_output, base_attention, base_raw = capture_compiler(backend, base_batch)
    live_output, live_attention, live_raw = capture_compiler(
        backend, base_batch, base_hidden_capture["hidden"], complement, positions)
    changed_hidden = base_hidden_capture["hidden"].clone()
    for i, selected in enumerate(positions):
        changed_hidden[i, list(selected)] = (changed_hidden[i, list(selected)].float()+complement[i, list(selected)]).to(changed_hidden)
    with torch.no_grad(): down_delta = mlp8.Down(changed_hidden).float()-mlp8.Down(base_hidden_capture["hidden"]).float()
    predicted_z9 = base_raw["z9"].clone()
    deep_gain = backend.model.transformer.h[9].lambdas[0].detach().float()
    for i, selected in enumerate(positions): predicted_z9[i, list(selected)] += (deep_gain*down_delta[i, list(selected)]).to(predicted_z9)
    predicted_current = backend.F.rms_norm(predicted_z9, (int(backend.model.config.n_embd),))
    attention = backend.model.transformer.h[9].attn
    with torch.no_grad():
        base_direct = attention.c_v(base_raw["current"]).view_as(base_attention["value"])
        predicted_direct = attention.c_v(predicted_current).view_as(base_attention["value"])
    initial_delta = (torch.zeros_like(base_attention["value"]) if base_raw["v1"] is None else
        live_raw["v1"].view_as(base_attention["value"]).float()-base_raw["v1"].view_as(base_attention["value"]).float())
    predicted_value_delta = ((1.0-attention.lamb.detach().float())*(predicted_direct.float()-base_direct.float())
        + attention.lamb.detach().float()*initial_delta)
    exact_value_delta = live_attention["value"].float()-base_attention["value"].float()
    predicted_response, exact_response = torch.zeros_like(base_attention["head_output"], dtype=torch.float32), torch.zeros_like(base_attention["head_output"], dtype=torch.float32)
    for i, row in enumerate(rows):
        query = int(row["base_semantic_position"]); sources = list(partitions[i]["post_cue_before_subject"])
        for head in (1, 4):
            pattern = base_attention["pattern"][i, head, query, sources].float()
            predicted_response[i, query, head] = (pattern[:, None]*predicted_value_delta[i, sources, head]).sum(0)
            exact_response[i, query, head] = (pattern[:, None]*exact_value_delta[i, sources, head]).sum(0)
    zero = torch.zeros_like(predicted_response)
    outputs = {"empty": source.run_delta(backend, base_batch, base_hidden_capture["hidden"], complement, positions, base_attention, zero),
        "exact": source.run_delta(backend, base_batch, base_hidden_capture["hidden"], complement, positions, base_attention, exact_response),
        "compiled": source.run_delta(backend, base_batch, base_hidden_capture["hidden"], complement, positions, base_attention, predicted_response)}
    forwards, evaluations = 7, 7*len(rows)
    state = lambda output: head_atlas.converter.state(output, rows, torch, backend.device)
    base18, base_hidden18 = state(base_output), state(base_hidden_output)
    states = {name: state(output) for name, output in outputs.items()}
    metrics, logit_error = effect_metrics(torch, backend, rows, base18, states["empty"], states["exact"], states["compiled"], s)
    z9_error = float((predicted_z9.float()-live_raw["z9"].float()).abs().max())
    current_error = float((predicted_current.float()-live_raw["current"].float()).abs().max())
    value_error = float((predicted_value_delta-exact_value_delta).abs().max())
    response_error = float((predicted_response-exact_response).abs().max())
    cproj_weight = attention.c_proj.weight.detach().float(); head_dim = int(backend.model.config.n_embd//backend.model.config.n_head)
    compiled_write = sum(torch.nn.functional.linear(predicted_response[:, :, head],
        cproj_weight[:, head*head_dim:(head+1)*head_dim], None) for head in (1, 4))
    exact_write = sum(torch.nn.functional.linear(exact_response[:, :, head],
        cproj_weight[:, head*head_dim:(head+1)*head_dim], None) for head in (1, 4))
    cproj_error = float((compiled_write-exact_write).abs().max())
    resid_relative = float((states["compiled"]-states["exact"]).norm()/(states["exact"]-states["empty"]).norm())
    identity_error = float((base_hidden18-base18).abs().max())
    initial_error = float(initial_delta.abs().max())
    finite = all(math.isfinite(value) for row in metrics.values() for value in row.values())
    pred_a = bool(orientation_error <= 1e-6 and z9_error <= .001 and current_error <= .001
        and value_error <= .001 and response_error <= .001 and cproj_error <= .001
        and identity_error <= .001 and finite and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    pred_b = all(metrics[p]["behavior_cosine"] >= .999 and metrics[p]["behavior_relative_rmse"] <= .01
        and metrics[p]["q8_cosine"] >= .999 and metrics[p]["q8_relative_rmse"] <= .01 for p in ("A1", "A2"))
    pred_c = logit_error <= 1e-4 and resid_relative <= 1e-4
    pred_d = initial_error <= 1e-6
    pred_e = True
    predictions = {"pred_a_authority_residual_rms_cv_cproj_closure_finiteness_and_price": pred_a,
        "pred_b_compiled_value_response_matches_exact_causal_effect": pred_b,
        "pred_c_compiled_full_vocabulary_intervention_replays": pred_c,
        "pred_d_initial_value_branch_is_unchanged": pred_d,
        "pred_e_zero_fit_literal_weight_program": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "iswas_mlp8_postcue_to_attn9_h1h4_value_weight_compiler_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter()-started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"f_linear_orientation_max_abs": orientation_error, "z9_recurrence_max_abs": z9_error,
            "rms_current_max_abs": current_error, "cv_value_delta_max_abs": value_error,
            "base_pattern_response_max_abs": response_error, "cproj_compiled_vs_exact_max_abs": cproj_error,
            "compiled_exact_full_logit_max_abs": logit_error, "compiled_exact_resid18_relative_norm": resid_relative,
            "initial_value_branch_delta_max_abs": initial_error, "native_base_identity_max_abs": identity_error, "rows": len(rows)},
        "metrics": metrics, "weight_program": {"block9_deep_gain": float(deep_gain),
            "compiled_h1h4_query_write_rms": float(torch.sqrt(compiled_write.square().mean())),
            "compiled_h1h4_query_q8_rms": float(torch.sqrt((compiled_write@s).square().mean()))},
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "metrics",
        "weight_program", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
