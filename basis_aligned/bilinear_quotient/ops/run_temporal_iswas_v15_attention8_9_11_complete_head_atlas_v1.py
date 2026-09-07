#!/usr/bin/env python3
"""Standalone and conditional complete-head atlas at v15 attention 8/9/11."""
# BQGATE: EXPERIMENT pred_a_authority_population_self_reconstruction_finiteness_and_exact_price pred_b_at_least_one_standalone_head_is_material pred_c_known_heads_are_enriched pred_d_conditional_head_necessity_is_material pred_e_head_support_is_not_singleton_only
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v15_all_layer_complete_module_response_atlas_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_attention8_9_11_complete_head_atlas_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_all_layer_complete_module_response_atlas_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_v15_all_layer_complete_module_response_atlas_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_attention8_9_11_complete_head_atlas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_attention8_9_11_complete_head_atlas_v1"
EXPECTED = {
    "prior": "d02ca6a7219db08fa16b3ac9995d9348d4a437cbe27bbe83755a3acd989e3ff4",
    "parent_result": "b04b049407b9a92e585550951e07ec92374a07fff70bdce8a61aba4613663468",
    "parent_runner": "450835763a1db0ea5586265a70775b57551a6820aa7107d4f4c50ea5e2cc3958",
}
LAYERS = (8, 9, 11)
HEADS = tuple(range(9))
KNOWN = {8: {1}, 9: {1, 4}, 11: {3}}
EXACT_FORWARDS = 60


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def label(layer, head):
    return f"L{layer}H{head}"


def capture(backend, batch):
    cache, handles = {}, []
    for layer in LAYERS:
        def save(_module, arguments, layer=layer):
            cache[layer] = arguments[0].detach().clone()
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(save))
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(cache) != set(LAYERS):
        raise RuntimeError("incomplete selected-attention head capture")
    return output, cache


def patch_hook(batch, values, heads, width):
    selected = tuple(heads)
    def hook(_module, arguments):
        changed = arguments[0].clone()
        for index, query in enumerate(batch.semantic_positions):
            stop = int(query) + 1
            for head in selected:
                left, right = head * width, (head + 1) * width
                changed[index, :stop, left:right] = values[index, :stop, left:right].to(changed)
        return (changed,) + tuple(arguments[1:])
    return hook


def run_patch(backend, batch, cache, groups):
    handles = []
    width = int(backend.model.config.n_embd // backend.model.config.n_head)
    for layer, heads in groups.items():
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
            patch_hook(batch, cache[layer], heads, width)))
    try:
        return backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()


def metric_error(left, right):
    values = []
    for block in ("behavior", "final_residual"):
        for key in ("signed_projection", "relative_squared_error", "cosine", "norm_ratio"):
            values.append(abs(left[block][key] - right[block][key]))
    values.append(abs(left["behavior"]["direction_fraction"] - right["behavior"]["direction_fraction"]))
    return max(values)


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def main():
    paths = {"prior": PRIOR, "parent_result": PARENT_RESULT, "parent_runner": PARENT_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"complete-head atlas authority changed: {observed}")
    prior, parent_result = json.loads(PRIOR.read_text()), json.loads(PARENT_RESULT.read_text())
    rows = [row for row in fresh.build_rows() if row["transform_id"] in {"A1", "A2"}]
    expected_ids = parent_result["population"]["row_ids"]
    parent_attention_ranking = [
        site for site in parent_result["summary"]["module_ranking_by_behavior_abs"] if site.startswith("attn:")
    ][:3]
    if (prior.get("candidate_id") != CANDIDATE_ID
            or parent_attention_ranking != ["attn:8", "attn:9", "attn:11"]
            or [row["row_id"] for row in rows] != expected_ids):
        raise RuntimeError("parent selection rule or row order changed")
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "layers": LAYERS, "heads_per_layer": len(HEADS), "singleton_arms": 27,
        "conditional_leave_one_out_arms": 27, "whole_layer_arms": 3,
        "model_forwards_exact": EXACT_FORWARDS, "fit_updates": 0,
        "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    base_output, base_cache = capture(backend, base_batch)
    donor_output, donor_cache = capture(backend, donor_batch)
    forwards = 2
    aligned = all(base_cache[layer].shape == donor_cache[layer].shape for layer in LAYERS)
    if not aligned:
        raise RuntimeError("base/donor head response shapes changed")
    base_state = parent.states(torch, backend, base_output, rows)
    donor_state = parent.states(torch, backend, donor_output, rows)
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    index = torch.arange(len(rows), device=backend.device)

    def margins(state):
        logits = das.head_logits(backend, state)
        return logits[index, answer] - logits[index, foil]

    base_margin, donor_margin = margins(base_state), margins(donor_state)
    target_margin, target_state = donor_margin - base_margin, donor_state - base_state

    def report(output):
        state = parent.states(torch, backend, output, rows)
        delta_margin, delta_state = margins(state) - base_margin, state - base_state
        behavior = parent.vector_metrics(torch, delta_margin, target_margin)
        behavior["direction_fraction"] = float(((delta_margin * target_margin) > 0).float().mean())
        return ({"behavior": behavior, "final_residual": parent.vector_metrics(torch, delta_state, target_state)},
                delta_margin, delta_state)

    self_output = run_patch(backend, base_batch, base_cache, {layer: HEADS for layer in LAYERS})
    forwards += 1
    self_state = parent.states(torch, backend, self_output, rows)
    self_error = max(float((self_state - base_state).abs().max()),
                     float((margins(self_state) - base_margin).abs().max()))

    whole_reports, whole_deltas = {}, {}
    for layer in LAYERS:
        whole_reports[str(layer)], dm, ds = report(run_patch(
            backend, base_batch, donor_cache, {layer: HEADS}))
        whole_deltas[layer] = (dm, ds)
        forwards += 1

    singleton_reports = {}
    for layer in LAYERS:
        for head in HEADS:
            singleton_reports[label(layer, head)], _dm, _ds = report(run_patch(
                backend, base_batch, donor_cache, {layer: (head,)}))
            forwards += 1

    leave_one_out_reports, conditional_reports = {}, {}
    for layer in LAYERS:
        full_margin, full_state = whole_deltas[layer]
        for head in HEADS:
            site = label(layer, head)
            loo, loo_margin, loo_state = report(run_patch(
                backend, base_batch, donor_cache, {layer: tuple(h for h in HEADS if h != head)}))
            leave_one_out_reports[site] = loo
            conditional_margin, conditional_state = full_margin - loo_margin, full_state - loo_state
            behavior = parent.vector_metrics(torch, conditional_margin, target_margin)
            behavior["direction_fraction"] = float(((conditional_margin * target_margin) > 0).float().mean())
            conditional_reports[site] = {
                "behavior": behavior,
                "final_residual": parent.vector_metrics(torch, conditional_state, target_state),
            }
            forwards += 1

    replay_errors = {
        str(layer): metric_error(whole_reports[str(layer)], parent_result["singleton_reports"][f"attn:{layer}"])
        for layer in LAYERS
    }
    score = {
        label(layer, head): max(
            abs(singleton_reports[label(layer, head)]["behavior"]["signed_projection"]),
            abs(conditional_reports[label(layer, head)]["behavior"]["signed_projection"]),
        )
        for layer in LAYERS for head in HEADS
    }
    top_head = max(score, key=lambda site: (
        abs(singleton_reports[site]["behavior"]["signed_projection"]), score[site]))
    layer_rankings = {
        str(layer): sorted((label(layer, head) for head in HEADS), key=lambda site: score[site], reverse=True)
        for layer in LAYERS
    }
    known_enrichment = {
        str(layer): bool({label(layer, head) for head in KNOWN[layer]} & set(layer_rankings[str(layer)][:3]))
        for layer in LAYERS
    }
    standalone_material = [
        site for site, value in singleton_reports.items()
        if abs(value["behavior"]["signed_projection"]) >= .10
        and value["behavior"]["direction_fraction"] >= .75
    ]
    conditional_material = [
        site for site, value in conditional_reports.items()
        if abs(value["behavior"]["signed_projection"]) >= .10
    ]
    support = [site for site, value in score.items() if value >= .08]
    pred_a = bool(aligned and self_error <= 1e-4 and max(replay_errors.values()) <= 1e-5
                  and finite([whole_reports, singleton_reports, leave_one_out_reports,
                              conditional_reports, score]) and forwards == EXACT_FORWARDS)
    pred_b = bool(standalone_material)
    pred_c = sum(known_enrichment.values()) >= 2
    pred_d = bool(conditional_material)
    pred_e = len(support) >= 2
    predictions = {
        "pred_a_authority_population_self_reconstruction_finiteness_and_exact_price": pred_a,
        "pred_b_at_least_one_standalone_head_is_material": pred_b,
        "pred_c_known_heads_are_enriched": pred_c,
        "pred_d_conditional_head_necessity_is_material": pred_d,
        "pred_e_head_support_is_not_singleton_only": pred_e,
    }
    top_standalone = abs(singleton_reports[top_head]["behavior"]["signed_projection"])
    if not pred_a:
        terminal = "invalid"
    elif pred_b and pred_d and top_standalone >= .30:
        terminal = "localized_head"
    elif all(predictions.values()):
        terminal = "distributed_head_set"
    elif pred_b and pred_d and not pred_c:
        terminal = "known_head_construction_shift"
    elif pred_c and pred_d and not pred_b:
        terminal = "conditional_only_head_set"
    elif not pred_b and not pred_d:
        terminal = "head_singleton_and_necessity_null"
    else:
        terminal = "partial_head_structure"
    result = {
        "schema": "temporal_iswas_v15_attention8_9_11_complete_head_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"aligned_shapes": aligned,
            "base_self_max_abs_error": self_error, "all_head_parent_replay_max_abs_error": replay_errors},
        "whole_layer_reports": whole_reports, "singleton_reports": singleton_reports,
        "leave_one_out_reports": leave_one_out_reports, "conditional_reports": conditional_reports,
        "summary": {"top_head": top_head, "top_standalone_behavior_abs": top_standalone,
            "layer_rankings_by_standalone_or_conditional": layer_rankings,
            "known_head_enrichment": known_enrichment,
            "standalone_material_heads": standalone_material,
            "conditional_material_heads": conditional_material,
            "heads_with_max_contribution_ge_008": support},
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards_observed": forwards, "model_forwards_exact": EXACT_FORWARDS,
            "example_evaluations": forwards * len(rows), "fit_updates": 0,
            "model_updates": 0, "transformer_backwards": 0},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "candidate_id", "instrument", "summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
