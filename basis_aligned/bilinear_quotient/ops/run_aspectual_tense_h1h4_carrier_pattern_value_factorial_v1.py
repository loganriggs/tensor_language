#!/usr/bin/env python3
"""Exact routing-mass versus content factorial inside cross-task H1/H4 carriers."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_factorization pred_b_carrier_route_recurrence pred_c_content_dominates_routing pred_d_interaction_secondary pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import time

from circuit_fast_screen_managed_runner import atomic_create_json
import aspectual_tense_cross_task_eval as cross
import run_aspectual_anchor_l9h1_h4_source_term_factorial_v1 as source
import run_aspectual_tense_l9h1_h4_cross_task_source_groups_v1 as groups


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_h1h4_carrier_pattern_value_factorial_v1.json"
OUT = ROOT / "circuits/followups/aspectual_tense_h1h4_carrier_pattern_value_factorial_v1_result.json"
CANDIDATE_ID = "aspectual_tense.h1h4_carrier_pattern_value_factorial_v1"
PATHS = {"carrier_result": ROOT / "circuits/followups/aspectual_tense_l9h1_h4_cross_task_source_groups_v1_result.json", "carrier_instrument": ROOT / "ops/run_aspectual_tense_l9h1_h4_cross_task_source_groups_v1.py", "attention_instrument": ROOT / "ops/run_aspectual_anchor_l9h1_h4_source_term_factorial_v1.py", "cross_eval": ROOT / "ops/aspectual_tense_cross_task_eval.py"}
EXPECTED_PRIOR_SHA256 = "311654ddd54800258e5c13b80616e070f65e9f3f2563720d3500c7ebb6f791b2"
EXPECTED = {"carrier_result": "283fabed2416a3fcd5c351e111e09b77d180fb9503e596c402f400a1cebebea5", "carrier_instrument": "d9cc28ebcbde8717b6828585663d5b2307ea403c75c3c937981a7b7c00600160", "attention_instrument": "8e890efd3520cfbece1d71f3ffb58397c732d8fc9c9446c74af9ac0380f2ca01", "cross_eval": "6b8762d6d6c060b96524b16a27f58dd74e8a7f3958234fc3e56ccf35354232f1"}
FACTORS = ("routing_mass_change", "content_change", "routing_content_interaction")
HEADS = (1, 4)


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subsets():
    return tuple(subset for width in range(4) for subset in itertools.combinations(FACTORS, width))


def arm_id(subset):
    return "empty" if not subset else "+".join(subset)


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    result = json.loads(PATHS["carrier_result"].read_text())
    if result.get("terminal") != "screen" or len(cross.paired_rows()) != 16 or len(subsets()) != 8:
        raise ExperimentError("parent result, population, or factorial changed")


def aggregate(capture, index, head, query, positions):
    pattern = capture["pattern"][index, head, query]
    value = capture["value"][index, :, head]
    mass = sum((pattern[position].float() for position in positions), pattern.new_zeros((), dtype=pattern.dtype).float())
    term = sum((pattern[position] * value[position] for position in positions), value.new_zeros(value.shape[-1])).float()
    if not math.isfinite(float(mass)) or abs(float(mass)) < 1e-4:
        raise ExperimentError("carrier routing mass is ill-conditioned")
    return mass, term, term / mass


def factor_tensors(backend, base_task, donor_task, base_batch, donor_batch, base_capture, donor_capture):
    tensors, max_error, min_mass = {}, 0.0, math.inf
    for index, (query, donor_query) in enumerate(zip(base_batch.semantic_positions, donor_batch.semantic_positions)):
        for head in HEADS:
            base_pos = groups.positions(base_task, query)["contextual_carrier"]
            donor_pos = groups.positions(donor_task, donor_query)["contextual_carrier"]
            p_base, t_base, v_base = aggregate(base_capture, index, head, query, base_pos)
            p_donor, t_donor, v_donor = aggregate(donor_capture, index, head, donor_query, donor_pos)
            min_mass = min(min_mass, abs(float(p_base)), abs(float(p_donor)))
            delta_p, delta_v = p_donor - p_base, v_donor - v_base
            factors = {"routing_mass_change": delta_p * v_base, "content_change": p_base * delta_v, "routing_content_interaction": delta_p * delta_v}
            closure = sum(factors.values(), backend.torch.zeros_like(t_base))
            max_error = max(max_error, float((closure - (t_donor - t_base)).abs().max()))
            tensors[(index, head)] = factors
    return tensors, max_error, min_mass


def intervene(backend, base_batch, tensors, subset):
    head_dim = backend.model.config.n_embd // backend.model.config.n_head
    def patch_heads(_module, arguments):
        flattened = arguments[0]
        head_output = flattened.view(len(base_batch.row_ids), flattened.shape[1], backend.model.config.n_head, head_dim).clone()
        for index, query in enumerate(base_batch.semantic_positions):
            for head in HEADS:
                delta = sum((tensors[(index, head)][factor] for factor in subset), backend.torch.zeros(head_dim, device=backend.device, dtype=backend.torch.float32))
                head_output[index, query, head] = (head_output[index, query, head].float() + delta).to(head_output.dtype)
        return (head_output.reshape_as(flattened),) + tuple(arguments[1:])
    handle = backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(patch_heads)
    try:
        (output, _capture), logits = cross.capture_softcapped_logits(backend, lambda: backend.manual_forward(base_batch))
    finally:
        handle.remove()
    return output, logits


def main():
    validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "pairs": 16, "factors": list(FACTORS), "factor_subsets": 8, "orientations": 2, "model_forwards": 18, "example_evaluations": 288, "factor_subset_orientation_arms": 16, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    pairs = cross.paired_rows()
    backend = source.SourceBackend.load("cuda")
    batches = {task: cross.make_batch(pairs, task) for task in cross.TASKS}
    native, captures = {}, {}
    forward_calls = evaluations = 0
    reconstruction_error = 0.0
    for task, batch in batches.items():
        (output, capture), logits = cross.capture_softcapped_logits(backend, lambda batch=batch: backend.manual_forward(batch))
        native[task] = cross.four_logits(logits, batch, pairs)
        captures[task] = capture
        reconstruction_error = max(reconstruction_error, float(capture["reconstruction_max_abs"]))
        forward_calls += 1
        evaluations += len(pairs)
    capability_cells = cross.capability_cells(native, pairs)
    records, summaries, values = [], {}, {}
    factor_error, min_mass = 0.0, math.inf
    for recipient, donor in cross.ORIENTATIONS:
        tensors, error, observed_mass = factor_tensors(backend, recipient, donor, batches[recipient], batches[donor], captures[recipient], captures[donor])
        factor_error, min_mass = max(factor_error, error), min(min_mass, observed_mass)
        orientation = f"{recipient}_to_{donor}"
        for subset in subsets():
            _output, logits = intervene(backend, batches[recipient], tensors, subset)
            patched = cross.four_logits(logits, batches[recipient], pairs)
            forward_calls += 1
            evaluations += len(pairs)
            group_records = [cross.intervention_record(native, row, recipient, donor, index, arm=arm_id(subset)) for index, row in enumerate(patched)]
            records.extend(group_records)
            summaries.setdefault(arm_id(subset), {})[orientation] = cross.summarize(group_records)
            values[(subset, orientation)] = summaries[arm_id(subset)][orientation]["mean_normalized_donor_recovery"]
    orientations = tuple(f"{recipient}_to_{donor}" for recipient, donor in cross.ORIENTATIONS)
    shapley = {orientation: {} for orientation in orientations}
    for orientation in orientations:
        for factor in FACTORS:
            total = 0.0
            for subset in subsets():
                if factor in subset:
                    continue
                extended = tuple(item for item in FACTORS if item in set(subset) | {factor})
                total += math.factorial(len(subset)) * math.factorial(2 - len(subset)) / math.factorial(3) * (values[(extended, orientation)] - values[(subset, orientation)])
            shapley[orientation][factor] = total
    full, two = FACTORS, ("routing_mass_change", "content_change")
    targets = {"has_had_to_is_was": 0.3197612203228399, "is_was_to_has_had": 0.2197994056260137}
    two_retained = {orientation: values[(two, orientation)] / values[(full, orientation)] for orientation in orientations}
    pred_a = all(cell["passed"] for cell in capability_cells) and reconstruction_error <= 1e-4 and factor_error <= 1e-4 and min_mass >= 1e-4 and len(records) == 256
    pred_b = all(abs(values[(full, orientation)] - targets[orientation]) <= 0.05 and summaries[arm_id(full)][orientation]["direction_fraction"] == 1.0 for orientation in orientations)
    pred_c = all(shapley[orientation]["content_change"] > 0.0 and shapley[orientation]["content_change"] > shapley[orientation]["routing_mass_change"] for orientation in orientations)
    pred_d = all(abs(shapley[orientation]["routing_content_interaction"]) <= 0.25 * abs(values[(full, orientation)]) and two_retained[orientation] >= 0.80 and summaries[arm_id(two)][orientation]["direction_fraction"] >= 0.75 for orientation in orientations)
    price = {"model_forwards": forward_calls, "example_evaluations": evaluations, "pairs": 16, "factor_subset_orientation_arms": 16, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 18, "example_evaluations": 288, "pairs": 16, "factor_subset_orientation_arms": 16, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = {"pred_a_authority_capability_exact_factorization": pred_a, "pred_b_carrier_route_recurrence": pred_b, "pred_c_content_dominates_routing": pred_c, "pred_d_interaction_secondary": pred_d, "pred_e_exact_zero_fit_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_e else "invalid")
    result = {"schema": "aspectual_tense_h1h4_carrier_pattern_value_factorial_result_v1", "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256, "authority_sha256": EXPECTED, "capability_cells": capability_cells, "instrument": {"source_reconstruction_max_abs_error": reconstruction_error, "aggregate_factor_max_abs_error": factor_error, "minimum_absolute_carrier_routing_mass": min_mass}, "summaries": summaries, "factorial_shapley": shapley, "routing_plus_content_retained_fraction": two_retained, "predictions": predictions, "price": price, "terminal": terminal, "reason": {"screen": "pattern_weighted_content_dominates_carrier_task_state", "null": "content_dominance_or_interaction_prediction_misses", "invalid": "authority_capability_mass_conditioning_factor_closure_recurrence_coverage_or_price_invalid"}[terminal], "serial_seconds": time.perf_counter() - started, "next_action": "trace effective-value content into its upstream value projection" if terminal == "screen" else "retain unsplit contextual carrier source terms"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "summaries", "factorial_shapley", "routing_plus_content_retained_fraction", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
