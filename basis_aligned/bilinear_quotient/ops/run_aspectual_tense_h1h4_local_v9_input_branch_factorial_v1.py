#!/usr/bin/env python3
"""Split localized L9 carrier values into deep-residual and direct-x0 inputs."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_input_instrument pred_b_local_value_route_recurrence pred_c_deep_residual_input_dominates pred_d_rms_branch_interaction_bounded pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

from circuit_fast_screen_managed_runner import atomic_create_json
import aspectual_tense_cross_task_eval as cross
import run_aspectual_anchor_l9h1_h4_source_term_factorial_v1 as source
import run_aspectual_tense_l9h1_h4_cross_task_source_groups_v1 as groups


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_h1h4_local_v9_input_branch_factorial_v1.json"
OUT = ROOT / "circuits/followups/aspectual_tense_h1h4_local_v9_input_branch_factorial_v1_result.json"
CANDIDATE_ID = "aspectual_tense.h1h4_local_v9_input_branch_factorial_v1"
PATHS = {
    "local_value_result": ROOT / "circuits/followups/aspectual_tense_h1h4_carrier_effective_value_branch_factorial_v1_result.json",
    "local_value_instrument": ROOT / "ops/run_aspectual_tense_h1h4_carrier_effective_value_branch_factorial_v1.py",
    "attention_instrument": ROOT / "ops/run_aspectual_anchor_l9h1_h4_source_term_factorial_v1.py",
    "cross_eval": ROOT / "ops/aspectual_tense_cross_task_eval.py",
    "legacy_ablation": ROOT / "arch_features_results.json",
}
EXPECTED_PRIOR_SHA256 = "40c05ddb582ca7e981b12842fbb9d6fe16819313827afbf5e9ac4b03391f1b26"
EXPECTED = {
    "local_value_result": "9f3c04abe6bca5448d228d6fc71951b804e6a66a9a95834e59461a0d32bcf9b6",
    "local_value_instrument": "7748dca375ff965797d692422153f7a9393b0e77ca8210942532f8b18936ea84",
    "attention_instrument": "8e890efd3520cfbece1d71f3ffb58397c732d8fc9c9446c74af9ac0380f2ca01",
    "cross_eval": "6b8762d6d6c060b96524b16a27f58dd74e8a7f3958234fc3e56ccf35354232f1",
    "legacy_ablation": "72272a8161c07240dfda9a9dc582721df428b35074083706093a75ce68ec1073",
}
BRANCHES = ("deep_resid9", "direct_x0_reinjection")
HEADS = (1, 4)


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subsets():
    return tuple(subset for width in range(3) for subset in itertools.combinations(BRANCHES, width))


def arm_id(subset):
    return "empty" if not subset else "+".join(subset)


def two_branch_factorial(subset_values):
    empty = subset_values[()]
    deep = subset_values[(BRANCHES[0],)]
    reinject = subset_values[(BRANCHES[1],)]
    joint = subset_values[BRANCHES]
    interaction = joint - deep - reinject + empty
    shapley = {
        BRANCHES[0]: 0.5 * ((deep - empty) + (joint - reinject)),
        BRANCHES[1]: 0.5 * ((reinject - empty) + (joint - deep)),
    }
    return {"shapley": shapley, "interaction": interaction, "deep_retained_fraction": deep / joint, "efficiency_error": abs(sum(shapley.values()) - (joint - empty))}


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    parent = json.loads(PATHS["local_value_result"].read_text())
    if parent.get("terminal") != "screen" or len(cross.paired_rows()) != 16 or len(subsets()) != 4:
        raise ExperimentError("parent, population, or factorial changed")


def capture_state(backend, batch):
    raw = {}
    width = backend.model.config.n_embd
    head_dim = width // backend.model.config.n_head
    original_rms_norm = backend.F.rms_norm
    width_calls = 0

    def recording_rms_norm(input_tensor, normalized_shape, *args, **kwargs):
        nonlocal width_calls
        output = original_rms_norm(input_tensor, normalized_shape, *args, **kwargs)
        shape = tuple(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else (normalized_shape,)
        if shape == (width,):
            if width_calls == 0:
                raw["x0"] = output.detach().clone()
            raw["last_width_input"] = input_tensor.detach().clone()
            width_calls += 1
        return output

    def save_v9(_module, _arguments, output):
        if "last_width_input" not in raw:
            raise ExperimentError("pre-normalization L9 input was not captured")
        raw["z9"] = raw["last_width_input"].detach().clone()
        raw["v9"] = output.detach().clone().view(len(batch.row_ids), output.shape[1], backend.model.config.n_head, head_dim)

    backend.F.rms_norm = recording_rms_norm
    handle = backend.model.transformer.h[9].attn.c_v.register_forward_hook(save_v9)
    try:
        (output, capture), logits = cross.capture_softcapped_logits(backend, lambda: backend.manual_forward(batch))
    finally:
        handle.remove()
        backend.F.rms_norm = original_rms_norm
    if not {"x0", "z9", "v9"}.issubset(raw):
        raise ExperimentError("RMS/c_v capture coverage failed")
    return output, capture, logits, {key: raw[key] for key in ("x0", "z9", "v9")}


def component_values(backend, raw):
    torch = backend.torch
    block = backend.model.transformer.h[9]
    width = backend.model.config.n_embd
    reinject = block.lambdas[1].detach().float() * raw["x0"].float()
    deep = raw["z9"].float() - reinject
    z_error = float((deep + reinject - raw["z9"].float()).abs().max())
    values = {}
    with torch.no_grad():
        for subset in subsets():
            z = sum(({BRANCHES[0]: deep, BRANCHES[1]: reinject}[branch] for branch in subset), torch.zeros_like(deep))
            current = backend.F.rms_norm(z, (width,))
            values[subset] = block.attn.c_v(current).view_as(raw["v9"]).detach().clone()
    both_error = float((values[BRANCHES].float() - raw["v9"].float()).abs().max())
    return values, z_error, both_error, float(block.lambdas[1].detach())


def weighted_mean(torch, pattern, values, positions, mass):
    return sum((pattern[position] * values[position] for position in positions), torch.zeros_like(values[0])).float() / mass


def arm_tensors(backend, base_task, donor_task, base_batch, donor_batch, base_capture, donor_capture, base_values, donor_values, base_native_v9, donor_native_v9):
    tensors, local_closure = {}, 0.0
    lamb = backend.model.transformer.h[9].attn.lamb.detach().float()
    for index, (query, donor_query) in enumerate(zip(base_batch.semantic_positions, donor_batch.semantic_positions)):
        for head in HEADS:
            base_positions = groups.positions(base_task, query)["contextual_carrier"]
            donor_positions = groups.positions(donor_task, donor_query)["contextual_carrier"]
            base_pattern = base_capture["pattern"][index, head, query].float()
            donor_pattern = donor_capture["pattern"][index, head, donor_query].float()
            p_base = sum((base_pattern[p] for p in base_positions), base_pattern.new_zeros(()))
            p_donor = sum((donor_pattern[p] for p in donor_positions), donor_pattern.new_zeros(()))
            if abs(float(p_base)) < 1e-4 or abs(float(p_donor)) < 1e-4:
                raise ExperimentError("carrier routing mass is ill-conditioned")
            arms = {}
            for subset in subsets():
                base_mean = weighted_mean(backend.torch, base_pattern, base_values[subset][index, :, head].float(), base_positions, p_base)
                donor_mean = weighted_mean(backend.torch, donor_pattern, donor_values[subset][index, :, head].float(), donor_positions, p_donor)
                arms[subset] = p_base * (1.0 - lamb) * (donor_mean - base_mean)
            base_native = weighted_mean(backend.torch, base_pattern, base_native_v9[index, :, head].float(), base_positions, p_base)
            donor_native = weighted_mean(backend.torch, donor_pattern, donor_native_v9[index, :, head].float(), donor_positions, p_donor)
            native_local = p_base * (1.0 - lamb) * (donor_native - base_native)
            local_closure = max(local_closure, float((arms[BRANCHES] - native_local).abs().max()))
            tensors[(index, head)] = arms
    return tensors, local_closure


def intervene(backend, base_batch, tensors, subset):
    head_dim = backend.model.config.n_embd // backend.model.config.n_head

    def patch_heads(_module, arguments):
        flattened = arguments[0]
        heads = flattened.view(len(base_batch.row_ids), flattened.shape[1], backend.model.config.n_head, head_dim).clone()
        for index, query in enumerate(base_batch.semantic_positions):
            for head in HEADS:
                heads[index, query, head] = (heads[index, query, head].float() + tensors[(index, head)][subset]).to(heads.dtype)
        return (heads.reshape_as(flattened),) + tuple(arguments[1:])

    handle = backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(patch_heads)
    try:
        (output, _capture), logits = cross.capture_softcapped_logits(backend, lambda: backend.manual_forward(base_batch))
    finally:
        handle.remove()
    return output, logits


def main():
    validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "pairs": 16, "branches": list(BRANCHES), "branch_subsets": 4, "orientations": 2, "model_forwards": 10, "example_evaluations": 160, "c_v_component_batch_evaluations": 8, "subset_orientation_arms": 8, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
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
    native, captures, raw_values, components = {}, {}, {}, {}
    forward_calls = evaluations = component_evaluations = 0
    source_error = z_error = v9_error = 0.0
    observed_x0_lambda = None
    for task, batch in batches.items():
        _output, capture, logits, raw = capture_state(backend, batch)
        native[task] = cross.four_logits(logits, batch, pairs)
        captures[task], raw_values[task] = capture, raw
        components[task], current_z_error, current_v9_error, observed_x0_lambda = component_values(backend, raw)
        source_error = max(source_error, float(capture["reconstruction_max_abs"]))
        z_error, v9_error = max(z_error, current_z_error), max(v9_error, current_v9_error)
        forward_calls += 1
        evaluations += len(pairs)
        component_evaluations += 4
    capability_cells = cross.capability_cells(native, pairs)
    records, summaries, values = [], {}, {}
    local_closure = 0.0
    for recipient, donor in cross.ORIENTATIONS:
        tensors, closure = arm_tensors(backend, recipient, donor, batches[recipient], batches[donor], captures[recipient], captures[donor], components[recipient], components[donor], raw_values[recipient]["v9"], raw_values[donor]["v9"])
        local_closure = max(local_closure, closure)
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
    shapley, interaction, deep_retained = {}, {}, {}
    mobius_error = 0.0
    for orientation in orientations:
        accounting = two_branch_factorial({subset: values[(subset, orientation)] for subset in subsets()})
        shapley[orientation], interaction[orientation] = accounting["shapley"], accounting["interaction"]
        deep_retained[orientation] = accounting["deep_retained_fraction"]
        mobius_error = max(mobius_error, accounting["efficiency_error"])
    instrument = {"source_reconstruction_max_abs_error": source_error, "z9_component_recombination_max_abs_error": z_error, "both_component_v9_native_max_abs_error": v9_error, "local_content_tensor_closure_max_abs_error": local_closure, "mobius_efficiency_max_abs_error": mobius_error, "observed_block9_x0_lambda": observed_x0_lambda}
    targets = {"has_had_to_is_was": 0.3450739478442056, "is_was_to_has_had": 0.18414534033599952}
    max_instrument_error = max(value for key, value in instrument.items() if key.endswith("error"))
    pred_a = all(cell["passed"] for cell in capability_cells) and max_instrument_error <= 1e-4 and len(records) == 128 and all(math.isfinite(value) for value in interaction.values())
    pred_b = all(abs(values[(BRANCHES, orientation)] - targets[orientation]) <= 0.05 and summaries[arm_id(BRANCHES)][orientation]["direction_fraction"] == 1.0 for orientation in orientations)
    pred_c = all(shapley[orientation][BRANCHES[0]] > 0.0 and shapley[orientation][BRANCHES[0]] > shapley[orientation][BRANCHES[1]] and deep_retained[orientation] >= 0.75 and summaries[BRANCHES[0]][orientation]["direction_fraction"] >= 0.75 for orientation in orientations)
    pred_d = all(abs(interaction[orientation]) <= 0.25 * abs(values[(BRANCHES, orientation)]) for orientation in orientations)
    price = {"model_forwards": forward_calls, "example_evaluations": evaluations, "c_v_component_batch_evaluations": component_evaluations, "pairs": 16, "subset_orientation_arms": 8, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 10, "example_evaluations": 160, "c_v_component_batch_evaluations": 8, "pairs": 16, "subset_orientation_arms": 8, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = {"pred_a_authority_capability_exact_input_instrument": pred_a, "pred_b_local_value_route_recurrence": pred_b, "pred_c_deep_residual_input_dominates": pred_c, "pred_d_rms_branch_interaction_bounded": pred_d, "pred_e_exact_zero_fit_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_e else "invalid")
    result = {"schema": "aspectual_tense_h1h4_local_v9_input_branch_factorial_result_v1", "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256, "authority_sha256": EXPECTED, "capability_cells": capability_cells, "instrument": instrument, "summaries": summaries, "branch_shapley": shapley, "branch_interaction": interaction, "deep_retained_fraction": deep_retained, "predictions": predictions, "price": price, "terminal": terminal, "reason": {"screen": "deep_resid9_dominates_local_carrier_value_input", "null": "deep_input_dominance_or_rms_interaction_prediction_misses", "invalid": "authority_capability_capture_closure_recurrence_coverage_or_price_invalid"}[terminal], "serial_seconds": time.perf_counter() - started, "next_action": "localize the deep resid9 carrier state across block8 modules" if terminal == "screen" else "retain both exact block9 input branches"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "summaries", "branch_shapley", "branch_interaction", "deep_retained_fraction", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
