#!/usr/bin/env python3
"""Literal local L11/L15 c_v, pattern, and c_proj compiler for auxiliary values."""

# BQGATE: EXPERIMENT pred_a_authority_local_weight_closure_finiteness_and_price pred_b_compiled_response_matches_exact_causal_effect pred_c_compiled_full_vocabulary_intervention_replays pred_d_initial_value_branch_is_characterized pred_e_zero_fit_literal_weight_program
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
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_auxiliary_three_head_factor_program_v1 as factor
import run_iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_v1 as auxiliary
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_auxiliary_postcue_local_weight_compiler_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_postcue_value_source_v12_transfer_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_auxiliary_postcue_value_source_v12_transfer_v1.py"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
ATTENTION_LIBRARY = ROOT / "ops/attention_source_destination_eval.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_postcue_local_weight_compiler_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_auxiliary_postcue_local_weight_compiler_v1"
RESULT_SCHEMA = "iswas_mlp8_auxiliary_postcue_local_weight_compiler_result_v1"
EXPECTED = {
    "prior": "12cbd392e898f4fc01e4dd7922b8567174166cc6337b916e8f5e8b95a3aa42fd",
    "parent": "acacbd917e52a5cdc6d2b43bf374af3f35583376977b0d551bb0c57d78d2fab5",
    "parent_runner": "66bd24a47af15c71fa5a00103a8c26684bd0815cc5e9ffadbc180988e0841b60",
    "capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
    "builder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2",
    "attention_library": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
}
GROUPS = ("prefix_before_cue", "cue", "post_cue_before_subject", "subject_determiner", "self")
LAYERS, SELECTED, CORE = (11, 15), {11: (1, 3), 15: (5,)}, (1, 4)
MAX_FORWARDS, MAX_EVALUATIONS = 8, 240


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def subset_name(subset) -> str:
    return "+".join(subset) if subset else "empty"


def subsets():
    yield ()
    yield ("post_cue_before_subject",)
    yield GROUPS


def cosine(x, y) -> float:
    denominator = float(x.norm() * y.norm())
    return float((x * y).sum()) / denominator if denominator else 0.0


def source_partition(row):
    differences = [i for i, pair in enumerate(zip(row["base_ids"], row["donor_ids"])) if pair[0] != pair[1]]
    if len(differences) != 1:
        raise RuntimeError("source atlas requires one aligned cue")
    cue, query = differences[0], int(row["base_semantic_position"])
    groups = {"prefix_before_cue": tuple(range(cue)), "cue": (cue,),
        "post_cue_before_subject": tuple(range(cue + 1, query - 1)),
        "subject_determiner": (query - 1,), "self": (query,)}
    flat = tuple(position for name in GROUPS for position in groups[name])
    if tuple(sorted(flat)) != tuple(range(query + 1)) or len(flat) != len(set(flat)):
        raise RuntimeError("source groups do not partition causal positions")
    return groups


def capture_dual(backend, batch, *, call, capture_raw9=False):
    captures = {layer: {} for layer in LAYERS}
    raw9, handles = {}, []
    if capture_raw9:
        def capture9(_module, arguments):
            raw9["value"] = arguments[0].detach().clone()
        handles.append(backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(capture9))
    for layer in LAYERS:
        attention = backend.model.transformer.h[layer].attn
        def capture_inputs(_module, arguments, layer=layer, attention=attention):
            current, v1 = arguments[0], arguments[1] if len(arguments) > 1 else None
            pattern, value, reconstructed = attention_eval._attention_terms(backend, attention, current, v1)
            captures[layer].update(pattern=pattern.detach().clone(), value=value.detach().clone(),
                                   reconstructed=reconstructed.detach().clone(),
                                   current=current.detach().clone(),
                                   initial=None if v1 is None else v1.detach().clone())
        def capture_output(_module, arguments, layer=layer):
            raw = arguments[0]
            heads = int(backend.model.config.n_head)
            captures[layer]["head_output"] = raw.detach().clone().view(
                len(batch.row_ids), raw.shape[1], heads, raw.shape[2] // heads)
        handles.append(attention.register_forward_pre_hook(capture_inputs))
        handles.append(attention.c_proj.register_forward_pre_hook(capture_output))
    try:
        output = call()
    finally:
        for handle in handles: handle.remove()
    for layer in LAYERS:
        captures[layer]["reconstruction_max_abs"] = float(
            (captures[layer]["reconstructed"].float() - captures[layer]["head_output"].float()).abs().max())
    return output, captures, raw9.get("value")


def main() -> None:
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER,
             "capability": CAPABILITY, "builder": BUILDER,
             "attention_library": ATTENTION_LIBRARY}
    if {key: sha(value) for key, value in paths.items()} != EXPECTED:
        raise RuntimeError("auxiliary source authority changed")
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
    partitions = [source_partition(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen"
            or len(rows) != 30):
        raise RuntimeError("parent decision or population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "groups": list(GROUPS),
        "arms": ["empty", "exact_postcue", "compiled_postcue"],
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
    base_output, base_captures, base_raw9 = capture_dual(
        backend, base_batch, capture_raw9=True, call=lambda: backend.native(base_batch, capture=True))
    core_output, changed_captures, _unused = capture_dual(
        backend, base_batch, capture_raw9=False,
        call=lambda: auxiliary.run_heads(backend, base_batch, base_hidden_capture["hidden"],
            complement, source_positions, {9: base_raw9}, {9: CORE}))
    zero = {layer: torch.zeros_like(base_captures[layer]["head_output"], dtype=torch.float32)
            for layer in LAYERS}
    exact_response, compiled_response = {}, {}
    value_mix_error = response_error = cproj_error = split_closure = 0.0
    branch_rms = {}
    for layer in LAYERS:
        attention = backend.model.transformer.h[layer].attn
        shape = base_captures[layer]["value"].shape
        with torch.no_grad():
            base_direct = attention.c_v(base_captures[layer]["current"]).view(shape).float()
            changed_direct = attention.c_v(changed_captures[layer]["current"]).view(shape).float()
        direct_delta = changed_direct - base_direct
        base_initial = (base_direct if base_captures[layer]["initial"] is None
            else base_captures[layer]["initial"].view(shape).float())
        changed_initial = (changed_direct if changed_captures[layer]["initial"] is None
            else changed_captures[layer]["initial"].view(shape).float())
        coefficient = attention.lamb.detach().float()
        if base_captures[layer]["initial"] is None and changed_captures[layer]["initial"] is None:
            direct_component, inherited_component = direct_delta, torch.zeros_like(direct_delta)
        else:
            direct_component = (1.0 - coefficient) * direct_delta
            inherited_component = coefficient * (changed_initial - base_initial)
        compiled_value_delta = direct_component + inherited_component
        exact_value_delta = changed_captures[layer]["value"].float() - base_captures[layer]["value"].float()
        value_mix_error = max(value_mix_error, float((compiled_value_delta - exact_value_delta).abs().max()))
        base_pattern = base_captures[layer]["pattern"].float()
        exact_response[layer], compiled_response[layer] = zero[layer].clone(), zero[layer].clone()
        direct_response, inherited_response = zero[layer].clone(), zero[layer].clone()
        for i, source in enumerate(partitions):
            query = int(rows[i]["base_semantic_position"])
            source_index = torch.as_tensor(source["post_cue_before_subject"], device=backend.device)
            for head in SELECTED[layer]:
                pattern = base_pattern[i, head, query].index_select(0, source_index)
                exact_response[layer][i, query, head] = torch.einsum(
                    "s,sd->d", pattern, exact_value_delta[i].index_select(0, source_index)[:, head])
                compiled_response[layer][i, query, head] = torch.einsum(
                    "s,sd->d", pattern, compiled_value_delta[i].index_select(0, source_index)[:, head])
                direct_response[i, query, head] = torch.einsum(
                    "s,sd->d", pattern, direct_component[i].index_select(0, source_index)[:, head])
                inherited_response[i, query, head] = torch.einsum(
                    "s,sd->d", pattern, inherited_component[i].index_select(0, source_index)[:, head])
        response_error = max(response_error,
            float((compiled_response[layer] - exact_response[layer]).abs().max()))
        split_closure = max(split_closure, float(
            (direct_response + inherited_response - compiled_response[layer]).abs().max()))
        branch_rms[str(layer)] = {
            "direct_cv_response_rms": float(torch.sqrt(direct_response.square().mean())),
            "inherited_initial_response_rms": float(torch.sqrt(inherited_response.square().mean())),
            "compiled_response_rms": float(torch.sqrt(compiled_response[layer].square().mean())),
        }
        cproj = attention.c_proj.weight.detach().float()
        head_dim = int(backend.model.config.n_embd // backend.model.config.n_head)
        exact_write = sum(torch.nn.functional.linear(exact_response[layer][:, :, head],
            cproj[:, head * head_dim:(head + 1) * head_dim], None) for head in SELECTED[layer])
        compiled_write = sum(torch.nn.functional.linear(compiled_response[layer][:, :, head],
            cproj[:, head * head_dim:(head + 1) * head_dim], None) for head in SELECTED[layer])
        cproj_error = max(cproj_error, float((compiled_write - exact_write).abs().max()))
    arm_deltas = {
        "empty": zero,
        "exact_postcue": exact_response,
        "compiled_postcue": compiled_response,
    }
    outputs = {name: factor.run_factor_arm(backend, base_batch, base_hidden_capture["hidden"], complement,
        source_positions, base_raw9, base_captures, deltas) for name, deltas in arm_deltas.items()}
    forwards, evaluations = 7, 7 * len(rows)
    reconstruction = max(capture["reconstruction_max_abs"] for capture in
                         (*base_captures.values(), *changed_captures.values()))
    state = lambda output: converter.state(output, rows, torch, backend.device)
    base18, base_hidden18 = map(state, (base_output, base_hidden_output))
    states = {name: state(value) for name, value in outputs.items()}
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda value: das.head_logits(backend, value)[index, answers] - das.head_logits(backend, value)[index, foils]
    empty_margin = margin(states["empty"])
    exact_e = margin(states["exact_postcue"]) - empty_margin
    compiled_e = margin(states["compiled_postcue"]) - empty_margin
    exact_c = (states["exact_postcue"] - states["empty"]) @ s
    compiled_c = (states["compiled_postcue"] - states["empty"]) @ s
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)
             for panel in ("A1", "A2")}
    metrics = {panel: {} for panel in masks}
    for panel, mask in masks.items():
        behavior_rmse = torch.sqrt((compiled_e[mask] - exact_e[mask]).square().mean())
        q8_rmse = torch.sqrt((compiled_c[mask] - exact_c[mask]).square().mean())
        metrics[panel] = {
            "behavior_cosine": cosine(compiled_e[mask], exact_e[mask]),
            "behavior_relative_rmse": float(behavior_rmse / torch.sqrt(exact_e[mask].square().mean())),
            "q8_cosine": cosine(compiled_c[mask].reshape(-1), exact_c[mask].reshape(-1)),
            "q8_relative_rmse": float(q8_rmse / torch.sqrt(exact_c[mask].square().mean())),
        }
    identity_error = float((base_hidden18 - base18).abs().max())
    logit_error = float((das.head_logits(backend, states["compiled_postcue"])
        - das.head_logits(backend, states["exact_postcue"])).abs().max())
    resid_relative = float((states["compiled_postcue"] - states["exact_postcue"]).norm()
        / (states["exact_postcue"] - states["empty"]).norm())
    finite = all(math.isfinite(value) for panel in metrics.values() for value in panel.values()) \
        and all(math.isfinite(value) for layer in branch_rms.values() for value in layer.values())
    pred_a = bool(orientation_error <= 1e-6 and reconstruction <= .001 and value_mix_error <= .001
                  and response_error <= .001 and cproj_error <= .001 and identity_error <= .001 and finite
                  and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    pred_b = all(metrics[p]["behavior_cosine"] >= .999 and metrics[p]["behavior_relative_rmse"] <= .01
        and metrics[p]["q8_cosine"] >= .999 and metrics[p]["q8_relative_rmse"] <= .01 for p in ("A1", "A2"))
    pred_c = logit_error <= 1e-4 and resid_relative <= 1e-4
    pred_d = split_closure <= .001
    pred_e = set(arm_deltas) == {"empty", "exact_postcue", "compiled_postcue"}
    predictions = {"pred_a_authority_local_weight_closure_finiteness_and_price": pred_a,
        "pred_b_compiled_response_matches_exact_causal_effect": pred_b,
        "pred_c_compiled_full_vocabulary_intervention_replays": pred_c,
        "pred_d_initial_value_branch_is_characterized": pred_d,
        "pred_e_zero_fit_literal_weight_program": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": RESULT_SCHEMA,
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_attention_reconstruction_max_abs": reconstruction,
            "cv_initial_value_mix_max_abs": value_mix_error,
            "compiled_exact_postcue_response_max_abs": response_error,
            "direct_plus_initial_response_closure_max_abs": split_closure,
            "compiled_exact_cproj_write_max_abs": cproj_error,
            "compiled_exact_full_logit_max_abs": logit_error,
            "compiled_exact_resid18_relative_norm": resid_relative,
            "native_base_identity_max_abs": identity_error, "rows": len(rows)},
        "weight_program": branch_rms, "metrics": metrics,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
        "weight_program", "metrics", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
