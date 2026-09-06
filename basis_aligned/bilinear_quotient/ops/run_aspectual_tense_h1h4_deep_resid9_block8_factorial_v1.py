#!/usr/bin/env python3
"""Factor deep resid9 carrier state into block8 entry, attention, and MLP."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_block_factorial pred_b_deep_route_recurrence pred_c_state_preexists_block8_updates pred_d_each_block8_update_is_nonessential pred_e_exact_zero_fit_price
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
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_h1h4_deep_resid9_block8_factorial_v1.json"
OUT = ROOT / "circuits/followups/aspectual_tense_h1h4_deep_resid9_block8_factorial_v1_result.json"
CANDIDATE_ID = "aspectual_tense.h1h4_deep_resid9_block8_factorial_v1"
PATHS = {
    "deep_input_result": ROOT / "circuits/followups/aspectual_tense_h1h4_local_v9_input_branch_factorial_v1_result.json",
    "deep_input_instrument": ROOT / "ops/run_aspectual_tense_h1h4_local_v9_input_branch_factorial_v1.py",
    "attention_instrument": ROOT / "ops/run_aspectual_anchor_l9h1_h4_source_term_factorial_v1.py",
    "cross_eval": ROOT / "ops/aspectual_tense_cross_task_eval.py",
}
EXPECTED_PRIOR_SHA256 = "ecd4933e17af2f550ef8cbcb7c8b02dc0a324e3b8fa01b0f3ba8703d5fcdb169"
EXPECTED = {
    "deep_input_result": "a9281216e53650ee8ddd9fef83b78b8198d67ae2110f9bf8664facb12fd1b35c",
    "deep_input_instrument": "bf97fd9a88fd2b793358fd1285f9bd28e5749d436055eea76fbe883c6fdc9914",
    "attention_instrument": "8e890efd3520cfbece1d71f3ffb58397c732d8fc9c9446c74af9ac0380f2ca01",
    "cross_eval": "6b8762d6d6c060b96524b16a27f58dd74e8a7f3958234fc3e56ccf35354232f1",
}
BRANCHES = ("block8_entry_z8", "attention8_update", "mlp8_update")
HEADS = (1, 4)


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subsets():
    return tuple(subset for width in range(4) for subset in itertools.combinations(BRANCHES, width))


def arm_id(subset):
    return "empty" if not subset else "+".join(subset)


def factorial_accounting(subset_values):
    n = len(BRANCHES)
    shapley = {}
    for factor in BRANCHES:
        total = 0.0
        for subset in subsets():
            if factor in subset:
                continue
            extended = tuple(item for item in BRANCHES if item in set(subset) | {factor})
            total += math.factorial(len(subset)) * math.factorial(n - len(subset) - 1) / math.factorial(n) * (subset_values[extended] - subset_values[subset])
        shapley[factor] = total
    return {"shapley": shapley, "efficiency_error": abs(sum(shapley.values()) - (subset_values[BRANCHES] - subset_values[()]))}


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    parent = json.loads(PATHS["deep_input_result"].read_text())
    if parent.get("terminal") != "null" or not parent["predictions"]["pred_c_deep_residual_input_dominates"]:
        raise ExperimentError("parent deep-input result changed")
    if len(cross.paired_rows()) != 16 or len(subsets()) != 8:
        raise ExperimentError("population or factorial changed")


def capture_components(backend, batch):
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

    def save_z(name):
        def hook(_module, _arguments, _output):
            if "last_width_input" not in raw:
                raise ExperimentError(f"{name} RMS input missing")
            raw[name] = raw["last_width_input"].detach().clone()
        return hook

    def save_output(name, reshape_heads=False):
        def hook(_module, _arguments, output):
            value = output.detach().clone()
            if reshape_heads:
                value = value.view(len(batch.row_ids), value.shape[1], backend.model.config.n_head, head_dim)
            raw[name] = value
        return hook

    backend.F.rms_norm = recording_rms_norm
    handles = [
        backend.model.transformer.h[8].attn.c_v.register_forward_hook(save_z("z8")),
        backend.model.transformer.h[8].attn.c_proj.register_forward_hook(save_output("attention8")),
        backend.model.transformer.h[8].mlp.register_forward_hook(save_output("mlp8")),
        backend.model.transformer.h[9].attn.c_v.register_forward_hook(save_z("z9")),
        backend.model.transformer.h[9].attn.c_v.register_forward_hook(save_output("native_v9", reshape_heads=True)),
    ]
    try:
        (output, capture), logits = cross.capture_softcapped_logits(backend, lambda: backend.manual_forward(batch))
    finally:
        for handle in handles:
            handle.remove()
        backend.F.rms_norm = original_rms_norm
    required = {"x0", "z8", "attention8", "mlp8", "z9", "native_v9"}
    if not required.issubset(raw):
        raise ExperimentError("block component capture coverage failed")
    return output, capture, logits, {key: raw[key] for key in required}


def component_values(backend, raw):
    torch = backend.torch
    block9 = backend.model.transformer.h[9]
    width = backend.model.config.n_embd
    components = {BRANCHES[0]: raw["z8"].float(), BRANCHES[1]: raw["attention8"].float(), BRANCHES[2]: raw["mlp8"].float()}
    x9 = sum(components.values(), torch.zeros_like(raw["z8"].float()))
    deep9 = raw["z9"].float() - block9.lambdas[1].detach().float() * raw["x0"].float()
    deep_error = float((block9.lambdas[0].detach().float() * x9 - deep9).abs().max())
    values = {}
    with torch.no_grad():
        for subset in subsets():
            z = block9.lambdas[0].detach().float() * sum((components[branch] for branch in subset), torch.zeros_like(x9))
            values[subset] = block9.attn.c_v(backend.F.rms_norm(z, (width,))).view_as(raw["native_v9"]).detach().clone()
        direct_deep = block9.attn.c_v(backend.F.rms_norm(deep9, (width,))).view_as(raw["native_v9"])
    joint_error = float((values[BRANCHES].float() - direct_deep.float()).abs().max())
    return values, deep_error, joint_error, float(block9.lambdas[0].detach())


def weighted_mean(torch, pattern, values, positions, mass):
    return sum((pattern[position] * values[position] for position in positions), torch.zeros_like(values[0])).float() / mass


def arm_tensors(backend, base_task, donor_task, base_batch, donor_batch, base_capture, donor_capture, base_values, donor_values):
    tensors = {}
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
            tensors[(index, head)] = arms
    return tensors


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
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "pairs": 16, "components": list(BRANCHES), "component_subsets": 8, "orientations": 2, "model_forwards": 18, "example_evaluations": 288, "c_v_component_batch_evaluations": 16, "subset_orientation_arms": 16, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
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
    native, captures, components = {}, {}, {}
    forward_calls = evaluations = component_evaluations = 0
    source_error = deep_error = joint_error = 0.0
    observed_deep_lambda = None
    for task, batch in batches.items():
        _output, capture, logits, raw = capture_components(backend, batch)
        native[task] = cross.four_logits(logits, batch, pairs)
        captures[task] = capture
        components[task], current_deep_error, current_joint_error, observed_deep_lambda = component_values(backend, raw)
        source_error = max(source_error, float(capture["reconstruction_max_abs"]))
        deep_error, joint_error = max(deep_error, current_deep_error), max(joint_error, current_joint_error)
        forward_calls += 1
        evaluations += len(pairs)
        component_evaluations += 8
    capability_cells = cross.capability_cells(native, pairs)
    records, summaries, values = [], {}, {}
    for recipient, donor in cross.ORIENTATIONS:
        tensors = arm_tensors(backend, recipient, donor, batches[recipient], batches[donor], captures[recipient], captures[donor], components[recipient], components[donor])
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
    shapley, z8_retained, leave_one_retained = {}, {}, {}
    mobius_error = 0.0
    z8 = (BRANCHES[0],)
    omit_mlp = (BRANCHES[0], BRANCHES[1])
    omit_attn = (BRANCHES[0], BRANCHES[2])
    for orientation in orientations:
        accounting = factorial_accounting({subset: values[(subset, orientation)] for subset in subsets()})
        shapley[orientation] = accounting["shapley"]
        mobius_error = max(mobius_error, accounting["efficiency_error"])
        z8_retained[orientation] = values[(z8, orientation)] / values[(BRANCHES, orientation)]
        leave_one_retained[orientation] = {"omit_mlp8": values[(omit_mlp, orientation)] / values[(BRANCHES, orientation)], "omit_attention8": values[(omit_attn, orientation)] / values[(BRANCHES, orientation)]}
    instrument = {"source_reconstruction_max_abs_error": source_error, "deep9_component_recombination_max_abs_error": deep_error, "joint_deep_v9_max_abs_error": joint_error, "mobius_efficiency_max_abs_error": mobius_error, "observed_block9_deep_lambda": observed_deep_lambda}
    targets = {"has_had_to_is_was": 0.34396155730965494, "is_was_to_has_had": 0.184366512527354}
    pred_a = all(cell["passed"] for cell in capability_cells) and max(value for key, value in instrument.items() if key.endswith("error")) <= 1e-4 and len(records) == 256 and all(math.isfinite(value) for values_ in shapley.values() for value in values_.values())
    pred_b = all(abs(values[(BRANCHES, orientation)] - targets[orientation]) <= 0.05 and summaries[arm_id(BRANCHES)][orientation]["direction_fraction"] == 1.0 for orientation in orientations)
    pred_c = all(shapley[orientation][BRANCHES[0]] > 0.0 and shapley[orientation][BRANCHES[0]] > shapley[orientation][BRANCHES[1]] and shapley[orientation][BRANCHES[0]] > shapley[orientation][BRANCHES[2]] and z8_retained[orientation] >= 0.75 and summaries[BRANCHES[0]][orientation]["direction_fraction"] >= 0.75 for orientation in orientations)
    pred_d = all(leave_one_retained[orientation][name] >= 0.75 for orientation in orientations for name in leave_one_retained[orientation]) and all(summaries[arm][orientation]["direction_fraction"] >= 0.75 for orientation in orientations for arm in (arm_id(omit_mlp), arm_id(omit_attn)))
    price = {"model_forwards": forward_calls, "example_evaluations": evaluations, "c_v_component_batch_evaluations": component_evaluations, "pairs": 16, "subset_orientation_arms": 16, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 18, "example_evaluations": 288, "c_v_component_batch_evaluations": 16, "pairs": 16, "subset_orientation_arms": 16, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = {"pred_a_authority_capability_exact_block_factorial": pred_a, "pred_b_deep_route_recurrence": pred_b, "pred_c_state_preexists_block8_updates": pred_c, "pred_d_each_block8_update_is_nonessential": pred_d, "pred_e_exact_zero_fit_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_e else "invalid")
    result = {"schema": "aspectual_tense_h1h4_deep_resid9_block8_factorial_result_v1", "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256, "authority_sha256": EXPECTED, "capability_cells": capability_cells, "instrument": instrument, "summaries": summaries, "component_shapley": shapley, "z8_retained_fraction": z8_retained, "leave_one_update_retained_fraction": leave_one_retained, "predictions": predictions, "price": price, "terminal": terminal, "reason": {"screen": "deep_carrier_state_predates_block8_updates", "null": "preexisting_state_or_nonessential_update_prediction_misses", "invalid": "authority_capability_capture_closure_recurrence_coverage_or_price_invalid"}[terminal], "serial_seconds": time.perf_counter() - started, "next_action": "run a coarser carrier-state chronology screen across earlier block boundaries" if terminal == "screen" else "retain the necessary block8 update and localize it"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "summaries", "component_shapley", "z8_retained_fraction", "leave_one_update_retained_fraction", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
