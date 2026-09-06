#!/usr/bin/env python3
"""Exact semantic source groups for cross-task state in L9H1/H4."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_source_partition pred_b_h1h4_task_route_recurrence pred_c_contextual_carrier_task_source pred_d_source_specificity pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_l9h1_h4_source_term_factorial_v1 as source
import run_aspectual_tense_direction_matched_task_state_onset_v1 as onset


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_l9h1_h4_cross_task_source_groups_v1.json"
OUT = ROOT / "circuits/followups/aspectual_tense_l9h1_h4_cross_task_source_groups_v1_result.json"
CANDIDATE_ID = "aspectual_tense.l9h1_h4_cross_task_source_groups_v1"
PATHS = {"block9_result": ROOT / "circuits/followups/aspectual_tense_block9_task_state_decomposition_v1_result.json", "block9_instrument": ROOT / "ops/run_aspectual_tense_block9_task_state_decomposition_v1.py", "source_instrument": ROOT / "ops/run_aspectual_anchor_l9h1_h4_source_term_factorial_v1.py", "is_source": ROOT / "circuits/followups/tense_auxiliary_is_was_l9h1_h4_source_term_factorial_v1_result.json", "has_source": ROOT / "circuits/followups/aspectual_anchor_l9h1_h4_downstream_source_bank_v2_result.json"}
EXPECTED_PRIOR_SHA256 = "483c588e6b3f56e49a613304d46f7d373f654c0a8036830c63c0a507f10c14bc"
EXPECTED = {"block9_result": "60ace001114aa43204a86820f26a5b03f99adcfaf83224efb260a96c404bd3c1", "block9_instrument": "00d28d2a6626dbe418acd123346582f4150abdceceae8dcf78ab2e92c939a60d", "source_instrument": "8e890efd3520cfbece1d71f3ffb58397c732d8fc9c9446c74af9ac0380f2ca01", "is_source": "4c266158213edcda9f0c86b19064cabe6d673815167b69d9eff381ddadda9cf5", "has_source": "6d694f92d35970f4eb5eba25ca3d9aff15cdbd1949db158a8be18e827e0423a7"}
HEADS = (1, 4)
ARMS = ("h1h4_complete", "h1h4_all_sources", "h1h4_prefix_cue", "h1h4_contextual_carrier", "h1h4_self")


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    result = json.loads(PATHS["block9_result"].read_text())
    if result.get("terminal") != "null" or len(onset.paired_rows()) != 16:
        raise ExperimentError("parent result or matched pairs changed")


def positions(task, query):
    split = query - (3 if task == "has_had" else 2)
    groups = {"prefix_cue": tuple(range(split)), "contextual_carrier": tuple(range(split, query)), "self": (query,), "all_sources": tuple(range(query + 1))}
    if sorted(position for name in ("prefix_cue", "contextual_carrier", "self") for position in groups[name]) != list(range(query + 1)):
        raise ExperimentError("semantic source groups do not partition the prefix")
    return groups


def manual_logits(backend, batch):
    holder = {}
    def hook(_module, _arguments, output):
        holder["raw"] = output.detach().clone()
    handle = backend.model.lm_head.register_forward_hook(hook)
    try:
        output, capture = backend.manual_forward(batch)
    finally:
        handle.remove()
    return output, capture, 30.0 * backend.torch.tanh(holder["raw"] / 30.0)


def mediate(backend, base_task, donor_task, base_batch, donor_batch, base_capture, donor_capture, arm):
    head_dim = backend.model.config.n_embd // backend.model.config.n_head
    def patch_heads(_module, arguments):
        flattened = arguments[0]
        head_output = flattened.view(len(base_batch.row_ids), flattened.shape[1], backend.model.config.n_head, head_dim).clone()
        for index, (query, donor_query) in enumerate(zip(base_batch.semantic_positions, donor_batch.semantic_positions)):
            base_groups, donor_groups = positions(base_task, query), positions(donor_task, donor_query)
            for head in HEADS:
                if arm == "h1h4_complete":
                    head_output[index, query, head] = donor_capture["head_output"][index, donor_query, head]
                    continue
                group = arm.removeprefix("h1h4_")
                base_sum = sum((base_capture["pattern"][index, head, query, position] * base_capture["value"][index, position, head] for position in base_groups[group]), backend.torch.zeros(head_dim, device=backend.device, dtype=head_output.dtype))
                donor_sum = sum((donor_capture["pattern"][index, head, donor_query, position] * donor_capture["value"][index, position, head] for position in donor_groups[group]), backend.torch.zeros(head_dim, device=backend.device, dtype=head_output.dtype))
                head_output[index, query, head] += donor_sum - base_sum
        return (head_output.reshape_as(flattened),) + tuple(arguments[1:])
    logits_holder = {}
    def save_logits(_module, _arguments, output):
        logits_holder["raw"] = output.detach().clone()
    head_handle = backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(patch_heads)
    logit_handle = backend.model.lm_head.register_forward_hook(save_logits)
    try:
        output, _capture = backend.manual_forward(base_batch)
    finally:
        logit_handle.remove()
        head_handle.remove()
    return output, 30.0 * backend.torch.tanh(logits_holder["raw"] / 30.0)


def summarize(records):
    values = [record["recovery"] for record in records]
    return {"count": len(records), "mean_normalized_donor_recovery": statistics.fmean(values), "mean_absolute_normalized_donor_recovery": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def main():
    validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "pairs": 16, "arms": list(ARMS), "orientations": 2, "model_forwards": 12, "example_evaluations": 192, "source_group_orientation_arms": 10, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    pairs = onset.paired_rows()
    backend = source.SourceBackend.load("cuda")
    batches = {task: onset.make_batch(pairs, task) for task in ("has_had", "is_was")}
    native, captures = {}, {}
    forward_calls = evaluations = 0
    reconstruction_error = 0.0
    for task, batch in batches.items():
        _output, capture, logits = manual_logits(backend, batch)
        native[task] = onset.four_logits(logits, batch, pairs)
        captures[task] = capture
        reconstruction_error = max(reconstruction_error, float(capture["reconstruction_max_abs"]))
        forward_calls += 1
        evaluations += len(pairs)
    capability_cells = []
    for task, pair_index, offset in (("has_had", 0, 0), ("is_was", 1, 2)):
        for direction in ("present_to_past", "past_to_present"):
            indices = [index for index, pair in enumerate(pairs) if pair[pair_index]["direction_id"] == direction]
            accuracy = sum(native[task][index][offset] > native[task][index][offset + 1] for index in indices) / len(indices)
            capability_cells.append({"task": task, "direction": direction, "count": len(indices), "accuracy": accuracy, "threshold": 0.85, "passed": accuracy >= 0.85})
    records, summaries, arm_logits = [], {}, {}
    for arm in ARMS:
        for base_task, donor_task in (("has_had", "is_was"), ("is_was", "has_had")):
            _output, logits = mediate(backend, base_task, donor_task, batches[base_task], batches[donor_task], captures[base_task], captures[donor_task], arm)
            values = onset.four_logits(logits, batches[base_task], pairs)
            forward_calls += 1
            evaluations += len(pairs)
            orientation = f"{base_task}_to_{donor_task}"
            arm_logits[(arm, orientation)] = values
            group = []
            for index, patched in enumerate(values):
                base_support, donor_support, patched_support = onset.is_support(native[base_task][index]), onset.is_support(native[donor_task][index]), onset.is_support(patched)
                sign = 1.0 if donor_task == "is_was" else -1.0
                denominator = sign * (donor_support - base_support)
                if denominator <= 0.0 or not math.isfinite(denominator):
                    raise ExperimentError("native endpoint ordering changed")
                record = {"arm": arm, "orientation": orientation, "pair_index": index, "recovery": sign * (patched_support - base_support) / denominator}
                records.append(record)
                group.append(record)
            summaries.setdefault(arm, {})[orientation] = summarize(group)
    orientations = ("has_had_to_is_was", "is_was_to_has_had")
    route_error = max(abs(a - b) for orientation in orientations for pair_a, pair_b in zip(arm_logits[("h1h4_complete", orientation)], arm_logits[("h1h4_all_sources", orientation)]) for a, b in zip(pair_a, pair_b))
    targets = {"has_had_to_is_was": 0.4356718468865772, "is_was_to_has_had": 0.31288246508505524}
    retained = {arm: {orientation: summaries[arm][orientation]["mean_normalized_donor_recovery"] / summaries["h1h4_complete"][orientation]["mean_normalized_donor_recovery"] for orientation in orientations} for arm in ("h1h4_prefix_cue", "h1h4_contextual_carrier", "h1h4_self")}
    pred_a = all(cell["passed"] for cell in capability_cells) and reconstruction_error <= 1e-4 and route_error <= 1e-4 and len(records) == 160
    pred_b = all(abs(summaries["h1h4_complete"][orientation]["mean_normalized_donor_recovery"] - targets[orientation]) <= 0.05 and summaries["h1h4_complete"][orientation]["direction_fraction"] == 1.0 for orientation in orientations)
    pred_c = all(retained["h1h4_contextual_carrier"][orientation] >= 0.50 and summaries["h1h4_contextual_carrier"][orientation]["direction_fraction"] >= 0.75 for orientation in orientations)
    pred_d = all(summaries["h1h4_prefix_cue"][orientation]["mean_normalized_donor_recovery"] <= summaries["h1h4_contextual_carrier"][orientation]["mean_normalized_donor_recovery"] and abs(retained["h1h4_self"][orientation]) <= 0.25 for orientation in orientations)
    price = {"model_forwards": forward_calls, "example_evaluations": evaluations, "pairs": 16, "source_group_orientation_arms": 10, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 12, "example_evaluations": 192, "pairs": 16, "source_group_orientation_arms": 10, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = {"pred_a_authority_capability_exact_source_partition": pred_a, "pred_b_h1h4_task_route_recurrence": pred_b, "pred_c_contextual_carrier_task_source": pred_c, "pred_d_source_specificity": pred_d, "pred_e_exact_zero_fit_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_e else "invalid")
    result = {"schema": "aspectual_tense_l9h1_h4_cross_task_source_groups_result_v1", "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256, "authority_sha256": EXPECTED, "capability_cells": capability_cells, "instrument": {"source_reconstruction_max_abs_error": reconstruction_error, "all_sources_vs_complete_max_abs_logit_error": route_error}, "summaries": summaries, "retained_fraction_of_complete": retained, "predictions": predictions, "price": price, "terminal": terminal, "reason": {"screen": "contextual_carrier_bank_carries_cross_task_h1h4_state", "null": "cross_task_source_localization_or_specificity_misses", "invalid": "authority_capability_source_closure_recurrence_coverage_or_price_invalid"}[terminal], "serial_seconds": time.perf_counter() - started, "next_action": "split the dominant semantic source group into cue-conditioned pattern and value channels" if terminal == "screen" else "retain whole H1H4 task-state route without the registered source typing"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "summaries", "retained_fraction_of_complete", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
