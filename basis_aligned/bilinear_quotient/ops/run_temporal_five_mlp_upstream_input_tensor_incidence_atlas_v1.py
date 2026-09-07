#!/usr/bin/env python3
"""Map every upstream native response into the exact five-MLP tensor program."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_self_patch_finiteness_and_price pred_b_all_compatible_local_tensor_expansions_close pred_c_known_writer_frontier_is_enriched pred_d_tensor_incidence_predicts_final_causal_modes pred_e_factor_and_target_structure_is_nontrivial
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import circuit_candidate_temporal_auxiliary_fresh_cues_v13 as temporal
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as iswas
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_five_mlp_literal_weight_tensor_program_v1 as literal
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as population

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1.json"
TENSOR_AUDIT = ROOT / "circuits/followups/temporal_five_mlp_literal_weight_tensor_program_v2_tolerance_audit_result.json"
TENSOR_RESULT = ROOT / "circuits/followups/temporal_five_mlp_literal_weight_tensor_program_v1_result.json"
TENSOR_RUNNER = ROOT / "ops/run_temporal_five_mlp_literal_weight_tensor_program_v1.py"
EARLIER_ATLAS = ROOT / "circuits/followups/temporal_iswas_upstream_full_response_mode_atlas_v1_result.json"
EARLIER_RUNNER = ROOT / "ops/run_temporal_iswas_upstream_full_response_mode_atlas_v1.py"
CONFIRMATION = ROOT / "circuits/followups/temporal_v13_iswas_v12_four_five_mlp_program_v1_result.json"
TEMPORAL_BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v13.py"
TEMPORAL_CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
ISWAS_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
ISWAS_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
WEIGHT_LIBRARY = ROOT / "ops/subspace_weight_atlas.py"
WEIGHT_RESULT = ROOT / "circuits/followups/temporal_iswas_two_mode_weight_pullback_v3_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.five_mlp_upstream_input_tensor_incidence_atlas_v1"
TARGETS = literal.SITES
TERMS = literal.TERMS
UPSTREAM_SITES = tuple(
    site
    for layer in range(18)
    for site in tuple(f"L{layer}H{head}" for head in range(9))
    + ((f"MLP{layer}",) if layer < 17 else ())
)
KNOWN_WRITERS = ("L7H8", "L9H1", "L9H4", "MLP7", "MLP9")
MAX_FORWARDS = 184
MAX_EVALUATIONS = 9936
EXPECTED = {
    "prior": "ba41e39135762ad4bdd90bac4fca42f28ca17d657f3db7e9a4ef8e9c460e1e52",
    "tensor_audit": "ae86583c3461b151de1c6b25eb586be4e32bb0c9c8670a45710c38693700c10d",
    "tensor_result": "a0ac3ab0f35d1942339120a79fad07c040c7cf3663c5dd6b711ea1317c823652",
    "tensor_runner": "0b7130cc67ce796d12fd09b22e60816b6ed24c566cddc97318b35ac7edf513ea",
    "earlier_atlas": "ff00f30785d00f2709f436a2f0bfa92a6a005d63e2abc72ad236836e0249b130",
    "earlier_runner": "8f1c2a1680163daaded60589540761f6b4903cb8160857185f0c313b146ce017",
    "confirmation": "9fc9c6aad9257ba1d75745bc1938ec2451120c1363a66ec933cae222d43c515d",
    "temporal_builder": "3f738bf2fb2d4a5425dba85eaf948d7ed888e4b891666ff77a51c8638acd2509",
    "temporal_capability": "e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1",
    "iswas_builder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2",
    "iswas_capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
    "weight_library": "2e7d3a546813a6029eca6fae455ad5abd03b429fcee92432ca6fe06e835e83f5",
    "weight_result": "c8ab608fa116342f9cbc8af4955e6087faa0f1eee9dd74dacb5c0ec168c5bf4d",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def site_parts(site):
    if site.startswith("MLP"):
        return "mlp", int(site[3:]), None
    layer, head = site[1:].split("H")
    return "attn", int(layer), int(head)


def compatible(site, target):
    kind, layer, _head = site_parts(site)
    target_layer = int(target[3:])
    return layer <= target_layer if kind == "attn" else layer < target_layer


def capture_native(backend, batch):
    cache = {"attention": {}, "mlp": {}, "io": {}}
    handles = []
    for layer, block in enumerate(backend.model.transformer.h):
        def save_attention(_module, arguments, layer=layer):
            cache["attention"][layer] = arguments[0].detach().float().cpu().clone()
        handles.append(block.attn.c_proj.register_forward_pre_hook(save_attention))
        if layer < 17:
            def save_mlp(_module, _arguments, output, layer=layer):
                cache["mlp"][layer] = output.detach().float().cpu().clone()
            handles.append(block.mlp.register_forward_hook(save_mlp))
    for target in TARGETS:
        layer = int(target[3:])
        module = backend.model.transformer.h[layer].mlp
        def save_input(_module, arguments, target=target):
            cache["io"].setdefault(target, {})["input"] = arguments[0].detach().float().clone()
        def save_output(_module, _arguments, output, target=target):
            cache["io"].setdefault(target, {})["output"] = output.detach().float().clone()
        handles.append(module.register_forward_pre_hook(save_input))
        handles.append(module.register_forward_hook(save_output))
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()
    if (set(cache["attention"]) != set(range(18))
            or set(cache["mlp"]) != set(range(17))
            or set(cache["io"]) != set(TARGETS)
            or any(set(item) != {"input", "output"} for item in cache["io"].values())):
        raise RuntimeError("incomplete native upstream/target capture")
    return output, cache


def run_patch(backend, batch, cache, sites):
    by_attention = {}
    mlps = set()
    for site in sites:
        kind, layer, head = site_parts(site)
        if kind == "attn":
            by_attention.setdefault(layer, []).append(head)
        else:
            mlps.add(layer)
    io = {}
    handles = []
    width = int(backend.model.config.n_embd // backend.model.config.n_head)
    for layer, heads in by_attention.items():
        source = cache["attention"][layer]
        def patch_attention(_module, arguments, heads=tuple(heads), source=source):
            changed = arguments[0].clone()
            for index, query in enumerate(batch.semantic_positions):
                stop_position = int(query) + 1
                for head in heads:
                    start, stop = head * width, (head + 1) * width
                    changed[index, :stop_position, start:stop] = source[
                        index, :stop_position, start:stop].to(changed)
            return (changed,) + tuple(arguments[1:])
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
            patch_attention))
    for layer in mlps:
        source = cache["mlp"][layer]
        def patch_mlp(_module, _arguments, output, source=source):
            changed = output.clone()
            for index, query in enumerate(batch.semantic_positions):
                stop_position = int(query) + 1
                changed[index, :stop_position] = source[index, :stop_position].to(changed)
            return changed
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(patch_mlp))
    for target in TARGETS:
        layer = int(target[3:])
        module = backend.model.transformer.h[layer].mlp
        def save_input(_module, arguments, target=target):
            io.setdefault(target, {})["input"] = arguments[0].detach().float().clone()
        def save_output(_module, _arguments, output, target=target):
            io.setdefault(target, {})["output"] = output.detach().float().clone()
        handles.append(module.register_forward_pre_hook(save_input))
        handles.append(module.register_forward_hook(save_output))
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(io) != set(TARGETS) or any(set(item) != {"input", "output"} for item in io.values()):
        raise RuntimeError("incomplete patched target capture")
    return output, io


def states(torch, backend, output, rows):
    return torch.stack([
        torch.as_tensor(output.captured[(row["row_id"], "resid:18")]) for row in rows
    ]).to(backend.device).float()


def query_rows(tensor, batch):
    return tensor[list(range(len(batch.row_ids))), list(batch.semantic_positions)].float()


def vector_stats(torch, value, reference):
    denominator = float(reference @ reference)
    value_norm, reference_norm = float(value.norm()), float(reference.norm())
    return {
        "signed_projection": float(value @ reference) / denominator if denominator else 0.0,
        "cosine": float(value @ reference) / (value_norm * reference_norm)
        if value_norm * reference_norm else 0.0,
        "norm_ratio": value_norm / reference_norm if reference_norm else 0.0,
    }


def ranks(values):
    order = np.argsort(np.asarray(values), kind="stable")
    result = np.empty(len(values))
    result[order] = np.arange(len(values))
    return result


def spearman(left, right):
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def numeric_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from numeric_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from numeric_values(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def main():
    paths = {
        "prior": PRIOR, "tensor_audit": TENSOR_AUDIT, "tensor_result": TENSOR_RESULT,
        "tensor_runner": TENSOR_RUNNER, "earlier_atlas": EARLIER_ATLAS,
        "earlier_runner": EARLIER_RUNNER, "confirmation": CONFIRMATION,
        "temporal_builder": TEMPORAL_BUILDER, "temporal_capability": TEMPORAL_CAPABILITY,
        "iswas_builder": ISWAS_BUILDER, "iswas_capability": ISWAS_CAPABILITY,
        "weight_library": WEIGHT_LIBRARY, "weight_result": WEIGHT_RESULT,
    }
    if {key: sha(path) for key, path in paths.items()} != EXPECTED:
        raise RuntimeError("upstream tensor-incidence authority changed")
    prior, audit, tensor_result, earlier, confirmation, tcap, icap, weights = [
        json.loads(path.read_text()) for path in (
            PRIOR, TENSOR_AUDIT, TENSOR_RESULT, EARLIER_ATLAS, CONFIRMATION,
            TEMPORAL_CAPABILITY, ISWAS_CAPABILITY, WEIGHT_RESULT)]
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": 54,
        "upstream_sites": len(UPSTREAM_SITES), "targets": list(TARGETS),
        "terms": list(TERMS), "model_forwards_max": MAX_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS, "fit_updates": 0,
        "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    temporal_rows = sum((population.capable_rows(temporal, tcap, panel, 12)
                         for panel in ("A1", "A2")), [])
    iswas_rows = sum((population.capable_rows(iswas, icap, panel)
                      for panel in ("A1", "A2")), [])
    rows = temporal_rows + iswas_rows
    authority_ok = bool(
        prior.get("candidate_id") == CANDIDATE_ID
        and audit.get("terminal") == "literal_bilinear_weight_program"
        and tensor_result.get("terminal") == "invalid"
        and earlier.get("terminal") == "screen"
        and confirmation.get("terminal") == "temporal_fresh_five_mlp_response_program"
        and tcap.get("terminal") == "manifest" and icap.get("terminal") == "screen"
        and len(temporal_rows) == 24 and len(iswas_rows) == 30
        and len(rows) == 54 and len(UPSTREAM_SITES) == 179)
    if not authority_ok:
        raise RuntimeError("upstream tensor-incidence population or decision changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    base_output, base_cache = capture_native(backend, base_batch)
    donor_output, donor_cache = capture_native(backend, donor_batch)
    aligned_shapes = all(
        base_cache[family][key].shape == donor_cache[family][key].shape
        for family in ("attention", "mlp") for key in base_cache[family])
    if not aligned_shapes:
        raise RuntimeError("base/donor upstream response shapes changed")
    self_output, self_io = run_patch(backend, base_batch, base_cache, UPSTREAM_SITES)
    base_state, donor_state, self_state = (
        states(torch, backend, output, rows)
        for output in (base_output, donor_output, self_output))
    self_error = max(
        float((self_state - base_state).abs().max()),
        *(float((self_io[target][key] - base_cache["io"][target][key]).abs().max())
          for target in TARGETS for key in ("input", "output")))

    family, _singular, _energy = literal.atlas.frontier.parent.family_builder.build_family(
        backend, json.loads(literal.atlas.frontier.parent.SUBSPACE.read_text()))
    q = family[8]
    full_gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                          for layer in literal.atlas.LAYERS)
    raw_modes, orientation_error, _wrong = literal.atlas.frontier.parent.overlap.residual_modes(
        backend, q, full_gain)
    state_basis = torch.linalg.qr(raw_modes, mode="reduced").Q
    reader_coordinates = torch.as_tensor(
        weights["mode_artifacts"]["reader_coordinates"], device=backend.device).float()
    physical_reader = state_basis @ reader_coordinates
    reader_hash_ok = tensor_sha(physical_reader) == weights[
        "mode_artifacts"]["physical_reader_covectors_sha256"]

    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    index = torch.arange(len(rows), device=backend.device)
    def margin(state):
        logits = das.head_logits(backend, state)
        return logits[index, answer] - logits[index, foil]
    base_margin, donor_margin = margin(base_state), margin(donor_state)
    full_behavior = donor_margin - base_margin
    full_modes = (donor_state - base_state) @ physical_reader
    task_indices = {
        "temporal": torch.arange(0, len(temporal_rows), device=backend.device),
        "iswas": torch.arange(len(temporal_rows), len(rows), device=backend.device),
    }
    later_gains = {
        target: math.prod(
            float(backend.model.transformer.h[layer].lambdas[0].detach().float())
            for layer in range(int(target[3:]) + 1, 18))
        for target in TARGETS
    }
    site_metrics = {}
    forwards, evaluations = 3, 3 * len(rows)
    closure_ratios = []
    closure_relative = []
    for site in UPSTREAM_SITES:
        output, io = run_patch(backend, base_batch, donor_cache, (site,))
        state = states(torch, backend, output, rows)
        final_behavior = margin(state) - base_margin
        final_modes = (state - base_state) @ physical_reader
        target_metrics = {}
        summed_tensor_modes = torch.zeros_like(full_modes)
        interaction_abs = 0.0
        total_factor_abs = 0.0
        for target in TARGETS:
            if not compatible(site, target):
                continue
            layer = int(target[3:])
            mlp = backend.model.transformer.h[layer].mlp
            x0, x1 = base_cache["io"][target]["input"], io[target]["input"]
            with torch.no_grad():
                left0, left1 = mlp.Left(x0).float(), mlp.Left(x1).float()
                right0, right1 = mlp.Right(x0).float(), mlp.Right(x1).float()
                delta_left, delta_right = left1 - left0, right1 - right0
                hidden = {
                    "left_change": delta_left * right0,
                    "right_change": left0 * delta_right,
                    "bilinear_interaction": delta_left * delta_right,
                }
                down = mlp.Down.weight.detach().float()
                factors = {
                    name: torch.nn.functional.linear(value, down, None)
                    for name, value in hidden.items()
                }
            observed = io[target]["output"] - base_cache["io"][target]["output"]
            reconstructed = sum(factors.values(), torch.zeros_like(observed))
            error = reconstructed - observed
            observed_norm = float(observed.norm())
            max_abs = float(error.abs().max())
            relative = float(error.square().sum() / observed.square().sum().clamp_min(1e-30))
            ratio = max_abs / max(observed_norm, 1e-30)
            closure_ratios.append(ratio)
            closure_relative.append(relative)
            observed_query = query_rows(observed, base_batch)
            observed_modes = later_gains[target] * (observed_query @ physical_reader)
            factor_modes = {
                name: later_gains[target] * (query_rows(value, base_batch) @ physical_reader)
                for name, value in factors.items()
            }
            summed_tensor_modes += observed_modes
            factor_norms = {name: float(value.abs().sum()) for name, value in factor_modes.items()}
            interaction_abs += factor_norms["bilinear_interaction"]
            total_factor_abs += sum(factor_norms.values())
            target_metrics[target] = {
                "local_closure": {
                    "max_abs_error": max_abs, "observed_output_norm": observed_norm,
                    "max_abs_over_output_norm": ratio,
                    "relative_squared_error": relative,
                },
                "factor_mode_l1": factor_norms,
                "factor_mode_stats": {
                    name: {
                        task: {
                            f"mode{mode + 1}": vector_stats(
                                torch, values[ids, mode], full_modes[ids, mode])
                            for mode in range(2)
                        }
                        for task, ids in task_indices.items()
                    }
                    for name, values in factor_modes.items()
                },
                "observed_direct_mode_stats": {
                    task: {
                        f"mode{mode + 1}": vector_stats(
                            torch, observed_modes[ids, mode], full_modes[ids, mode])
                        for mode in range(2)
                    }
                    for task, ids in task_indices.items()
                },
                "incidence_magnitude": sum(
                    abs(vector_stats(torch, observed_modes[ids, mode], full_modes[ids, mode])[
                        "signed_projection"])
                    for ids in task_indices.values() for mode in range(2)),
            }
        final_stats = {
            task: {
                "behavior": vector_stats(torch, final_behavior[ids], full_behavior[ids]),
                **{
                    f"mode{mode + 1}": vector_stats(
                        torch, final_modes[ids, mode], full_modes[ids, mode])
                    for mode in range(2)
                },
            }
            for task, ids in task_indices.items()
        }
        tensor_stats = {
            task: {
                f"mode{mode + 1}": vector_stats(
                    torch, summed_tensor_modes[ids, mode], full_modes[ids, mode])
                for mode in range(2)
            }
            for task, ids in task_indices.items()
        }
        site_metrics[site] = {
            "targets": target_metrics,
            "tensor_mode_stats": tensor_stats,
            "final_causal_stats": final_stats,
            "tensor_incidence_magnitude": sum(
                abs(tensor_stats[task][f"mode{mode + 1}"]["signed_projection"])
                for task in task_indices for mode in range(2)),
            "final_causal_mode_magnitude": sum(
                abs(final_stats[task][f"mode{mode + 1}"]["signed_projection"])
                for task in task_indices for mode in range(2)),
            "interaction_fraction": interaction_abs / total_factor_abs if total_factor_abs else 0.0,
        }
        forwards += 1
        evaluations += len(rows)
        del output, io, state

    tensor_values = [site_metrics[site]["tensor_incidence_magnitude"] for site in UPSTREAM_SITES]
    causal_values = [site_metrics[site]["final_causal_mode_magnitude"] for site in UPSTREAM_SITES]
    correlation = spearman(tensor_values, causal_values)
    tensor_ranking = sorted(UPSTREAM_SITES,
                            key=lambda site: site_metrics[site]["tensor_incidence_magnitude"],
                            reverse=True)
    causal_ranking = sorted(UPSTREAM_SITES,
                            key=lambda site: site_metrics[site]["final_causal_mode_magnitude"],
                            reverse=True)
    quintile = len(UPSTREAM_SITES) // 5
    ranked_causal_by_tensor = [site_metrics[site]["final_causal_mode_magnitude"]
                               for site in tensor_ranking]
    top_bottom_ratio = (
        float(np.mean(ranked_causal_by_tensor[:quintile]))
        / max(float(np.mean(ranked_causal_by_tensor[-quintile:])), 1e-30))
    known_top = [site for site in KNOWN_WRITERS if site in tensor_ranking[:20]]
    known_positive = {
        site: any(
            site_metrics[site]["final_causal_stats"][task][f"mode{mode + 1}"][
                "signed_projection"] > 0
            for task in task_indices for mode in range(2))
        for site in KNOWN_WRITERS
    }
    target_rankings = {
        target: sorted(
            (site for site in UPSTREAM_SITES if target in site_metrics[site]["targets"]),
            key=lambda site: site_metrics[site]["targets"][target]["incidence_magnitude"],
            reverse=True)
        for target in TARGETS
    }
    material_targets = [
        target for target, ranking in target_rankings.items()
        if ranking and site_metrics[ranking[0]]["targets"][target]["incidence_magnitude"] >= 0.02
    ]
    distinct_top_writers = len({target_rankings[target][0] for target in material_targets})
    top_interaction = max(site_metrics[site]["interaction_fraction"]
                          for site in tensor_ranking[:20])
    finite = all(math.isfinite(value) for value in numeric_values(site_metrics))
    pred_a = bool(
        authority_ok and aligned_shapes and reader_hash_ok and orientation_error <= 1e-6
        and self_error <= 1e-4 and finite and forwards <= MAX_FORWARDS
        and evaluations <= MAX_EVALUATIONS)
    pred_b = bool(
        closure_ratios and max(closure_ratios) <= 1e-6
        and max(closure_relative) <= 1e-8)
    pred_c = bool(
        len(known_top) >= 3 and all(known_positive[site] for site in known_top))
    pred_d = bool(correlation >= 0.35 and top_bottom_ratio >= 2.0)
    pred_e = bool(
        len(material_targets) >= 2 and distinct_top_writers >= 2
        and top_interaction >= 0.10)
    predictions = {
        "pred_a_authority_alignment_self_patch_finiteness_and_price": pred_a,
        "pred_b_all_compatible_local_tensor_expansions_close": pred_b,
        "pred_c_known_writer_frontier_is_enriched": pred_c,
        "pred_d_tensor_incidence_predicts_final_causal_modes": pred_d,
        "pred_e_factor_and_target_structure_is_nontrivial": pred_e,
    }
    terminal = (
        "invalid" if not pred_a or not pred_b
        else "upstream_tensor_incidence_screen" if all(predictions.values())
        else "causal_incidence_only" if pred_c
        else "upstream_tensor_null")
    result = {
        "schema": "temporal_five_mlp_upstream_input_tensor_incidence_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "evidence_scope": {"temporal": "fresh_v13", "iswas": "v12_replay_control"},
        "instrument": {
            "authority_ok": authority_ok, "aligned_shapes": aligned_shapes,
            "physical_reader_hash_ok": reader_hash_ok,
            "orientation_max_abs": orientation_error,
            "joint_self_patch_max_abs": self_error,
            "max_local_relative_squared_error": max(closure_relative),
            "max_local_abs_over_output_norm": max(closure_ratios),
        },
        "summary": {
            "tensor_vs_causal_spearman": correlation,
            "top_vs_bottom_quintile_causal_ratio": top_bottom_ratio,
            "known_writers_in_tensor_top20": known_top,
            "known_writer_positive_cells": known_positive,
            "material_targets": material_targets,
            "distinct_top_target_writers": distinct_top_writers,
            "top20_max_interaction_fraction": top_interaction,
        },
        "tensor_ranking": tensor_ranking, "causal_ranking": causal_ranking,
        "target_rankings": target_rankings, "site_metrics": site_metrics,
        "predictions": predictions, "terminal": terminal,
        "price": {
            "model_forwards": forwards, "example_evaluations": evaluations,
            "compatible_site_target_records": len(closure_ratios),
            "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0,
        },
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "candidate_id", "instrument", "summary", "tensor_ranking",
        "target_rankings", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
