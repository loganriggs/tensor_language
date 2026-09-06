#!/usr/bin/env python3
"""Split localized H1/H4 carrier content into local-L9 and carried-L0 value branches."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_branch_closure pred_b_parent_content_recurrence pred_c_local_l9_branch_dominates pred_d_branch_interaction_bounded pred_e_exact_zero_fit_price
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
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_h1h4_carrier_effective_value_branch_factorial_v1.json"
OUT = ROOT / "circuits/followups/aspectual_tense_h1h4_carrier_effective_value_branch_factorial_v1_result.json"
CANDIDATE_ID = "aspectual_tense.h1h4_carrier_effective_value_branch_factorial_v1"
PATHS = {
    "parent_result": ROOT / "circuits/followups/aspectual_tense_h1h4_carrier_pattern_value_factorial_v1_result.json",
    "parent_instrument": ROOT / "ops/run_aspectual_tense_h1h4_carrier_pattern_value_factorial_v1.py",
    "attention_instrument": ROOT / "ops/run_aspectual_anchor_l9h1_h4_source_term_factorial_v1.py",
    "cross_eval": ROOT / "ops/aspectual_tense_cross_task_eval.py",
}
EXPECTED_PRIOR_SHA256 = "687c3f841b18c5aaa31fd87c6dd55a11b2578c7e36aae3604386732992e95632"
EXPECTED = {
    "parent_result": "fbba3090f5727481425d43baaf2d8f981738b986140769071d706a757e20b570",
    "parent_instrument": "737d84daa4e3390a8ffb49e93c3967d097a7eec4fb4d6ac135ff2f4e2c5bae5e",
    "attention_instrument": "8e890efd3520cfbece1d71f3ffb58397c732d8fc9c9446c74af9ac0380f2ca01",
    "cross_eval": "6b8762d6d6c060b96524b16a27f58dd74e8a7f3958234fc3e56ccf35354232f1",
}
BRANCHES = ("local_l9_value_change", "carried_l0_v1_change")
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
    """Return the unique two-factor Möbius/Shapley accounting."""
    empty = subset_values[()]
    local = subset_values[(BRANCHES[0],)]
    carried = subset_values[(BRANCHES[1],)]
    joint = subset_values[BRANCHES]
    interaction = joint - local - carried + empty
    shapley = {
        BRANCHES[0]: 0.5 * ((local - empty) + (joint - carried)),
        BRANCHES[1]: 0.5 * ((carried - empty) + (joint - local)),
    }
    return {
        "shapley": shapley,
        "interaction": interaction,
        "local_retained_fraction": local / joint,
        "efficiency_error": abs(sum(shapley.values()) - (joint - empty)),
    }


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    parent = json.loads(PATHS["parent_result"].read_text())
    if parent.get("terminal") != "null" or not parent["predictions"]["pred_c_content_dominates_routing"]:
        raise ExperimentError("parent content result changed")
    if len(cross.paired_rows()) != 16 or len(subsets()) != 4:
        raise ExperimentError("population or branch factorial changed")


def capture_with_raw_values(backend, batch):
    raw = {}
    head_dim = backend.model.config.n_embd // backend.model.config.n_head

    def save(name):
        def hook(_module, _arguments, output):
            raw[name] = output.detach().clone().view(
                len(batch.row_ids), output.shape[1], backend.model.config.n_head, head_dim
            )
        return hook

    handles = [
        backend.model.transformer.h[0].attn.c_v.register_forward_hook(save("v1")),
        backend.model.transformer.h[9].attn.c_v.register_forward_hook(save("v9")),
    ]
    try:
        (output, capture), logits = cross.capture_softcapped_logits(
            backend, lambda: backend.manual_forward(batch)
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(raw) != {"v1", "v9"}:
        raise ExperimentError("raw value hook coverage failed")
    return output, capture, logits, raw


def weighted_mean(torch, pattern, values, positions, mass):
    return sum(
        (pattern[position] * values[position] for position in positions),
        torch.zeros_like(values[0]),
    ).float() / mass


def branch_tensors(backend, base_task, donor_task, base_batch, donor_batch, base_capture, donor_capture, base_raw, donor_raw):
    tensors = {}
    branch_error = effective_error = math_error = 0.0
    lamb = backend.model.transformer.h[9].attn.lamb.detach().float()
    for capture, raw in ((base_capture, base_raw), (donor_capture, donor_raw)):
        reconstructed = (1.0 - lamb) * raw["v9"].float() + lamb * raw["v1"].float()
        effective_error = max(effective_error, float((reconstructed - capture["value"].float()).abs().max()))
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
            base_local = weighted_mean(backend.torch, base_pattern, (1.0 - lamb) * base_raw["v9"][index, :, head].float(), base_positions, p_base)
            donor_local = weighted_mean(backend.torch, donor_pattern, (1.0 - lamb) * donor_raw["v9"][index, :, head].float(), donor_positions, p_donor)
            base_carried = weighted_mean(backend.torch, base_pattern, lamb * base_raw["v1"][index, :, head].float(), base_positions, p_base)
            donor_carried = weighted_mean(backend.torch, donor_pattern, lamb * donor_raw["v1"][index, :, head].float(), donor_positions, p_donor)
            branch = {
                "local_l9_value_change": p_base * (donor_local - base_local),
                "carried_l0_v1_change": p_base * (donor_carried - base_carried),
            }
            base_effective = weighted_mean(backend.torch, base_pattern, base_capture["value"][index, :, head].float(), base_positions, p_base)
            donor_effective = weighted_mean(backend.torch, donor_pattern, donor_capture["value"][index, :, head].float(), donor_positions, p_donor)
            parent_content = p_base * (donor_effective - base_effective)
            closure = branch[BRANCHES[0]] + branch[BRANCHES[1]]
            branch_error = max(branch_error, float((closure - parent_content).abs().max()))
            math_error = max(math_error, float((closure - sum(branch.values(), backend.torch.zeros_like(parent_content))).abs().max()))
            tensors[(index, head)] = branch
    return tensors, {"effective_value_recombination_max_abs_error": effective_error, "content_branch_closure_max_abs_error": branch_error, "branch_sum_internal_max_abs_error": math_error, "observed_lambda9": float(lamb)}


def intervene(backend, base_batch, tensors, subset):
    head_dim = backend.model.config.n_embd // backend.model.config.n_head

    def patch_heads(_module, arguments):
        flattened = arguments[0]
        heads = flattened.view(len(base_batch.row_ids), flattened.shape[1], backend.model.config.n_head, head_dim).clone()
        for index, query in enumerate(base_batch.semantic_positions):
            for head in HEADS:
                delta = sum((tensors[(index, head)][branch] for branch in subset), backend.torch.zeros(head_dim, device=backend.device, dtype=backend.torch.float32))
                heads[index, query, head] = (heads[index, query, head].float() + delta).to(heads.dtype)
        return (heads.reshape_as(flattened),) + tuple(arguments[1:])

    handle = backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(patch_heads)
    try:
        (output, _capture), logits = cross.capture_softcapped_logits(backend, lambda: backend.manual_forward(base_batch))
    finally:
        handle.remove()
    return output, logits


def main():
    validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "pairs": 16, "branches": list(BRANCHES), "branch_subsets": 4, "orientations": 2, "model_forwards": 10, "example_evaluations": 160, "subset_orientation_arms": 8, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
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
    native, captures, raw_values = {}, {}, {}
    forward_calls = evaluations = 0
    source_error = 0.0
    for task, batch in batches.items():
        _output, capture, logits, raw = capture_with_raw_values(backend, batch)
        native[task] = cross.four_logits(logits, batch, pairs)
        captures[task], raw_values[task] = capture, raw
        source_error = max(source_error, float(capture["reconstruction_max_abs"]))
        forward_calls += 1
        evaluations += len(pairs)
    capability_cells = cross.capability_cells(native, pairs)
    records, summaries, values = [], {}, {}
    instrument = {"source_reconstruction_max_abs_error": source_error, "effective_value_recombination_max_abs_error": 0.0, "content_branch_closure_max_abs_error": 0.0, "branch_sum_internal_max_abs_error": 0.0, "observed_lambda9": None}
    for recipient, donor in cross.ORIENTATIONS:
        tensors, checks = branch_tensors(backend, recipient, donor, batches[recipient], batches[donor], captures[recipient], captures[donor], raw_values[recipient], raw_values[donor])
        for key, value in checks.items():
            if key == "observed_lambda9":
                instrument[key] = value
            else:
                instrument[key] = max(float(instrument[key]), value)
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
    both = BRANCHES
    shapley, interaction, local_retained = {}, {}, {}
    mobius_efficiency_error = 0.0
    for orientation in orientations:
        accounting = two_branch_factorial({subset: values[(subset, orientation)] for subset in subsets()})
        shapley[orientation] = accounting["shapley"]
        interaction[orientation] = accounting["interaction"]
        local_retained[orientation] = accounting["local_retained_fraction"]
        mobius_efficiency_error = max(mobius_efficiency_error, accounting["efficiency_error"])
    instrument["mobius_efficiency_max_abs_error"] = mobius_efficiency_error
    targets = {"has_had_to_is_was": 0.3728653283328377, "is_was_to_has_had": 0.1813208145439096}
    max_instrument_error = max(float(instrument[key]) for key in instrument if key.endswith("error"))
    pred_a = all(cell["passed"] for cell in capability_cells) and max_instrument_error <= 1e-4 and len(records) == 128 and all(math.isfinite(value) for value in interaction.values())
    pred_b = all(abs(values[(both, orientation)] - targets[orientation]) <= 0.05 and summaries[arm_id(both)][orientation]["direction_fraction"] == 1.0 for orientation in orientations)
    pred_c = all(shapley[orientation][BRANCHES[0]] > 0.0 and shapley[orientation][BRANCHES[0]] > shapley[orientation][BRANCHES[1]] and local_retained[orientation] >= 0.60 and summaries[BRANCHES[0]][orientation]["direction_fraction"] >= 0.75 for orientation in orientations)
    pred_d = all(abs(interaction[orientation]) <= 0.25 * abs(values[(both, orientation)]) for orientation in orientations)
    price = {"model_forwards": forward_calls, "example_evaluations": evaluations, "pairs": 16, "subset_orientation_arms": 8, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 10, "example_evaluations": 160, "pairs": 16, "subset_orientation_arms": 8, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = {"pred_a_authority_capability_exact_branch_closure": pred_a, "pred_b_parent_content_recurrence": pred_b, "pred_c_local_l9_branch_dominates": pred_c, "pred_d_branch_interaction_bounded": pred_d, "pred_e_exact_zero_fit_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_e else "invalid")
    result = {"schema": "aspectual_tense_h1h4_carrier_effective_value_branch_factorial_result_v1", "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256, "authority_sha256": EXPECTED, "capability_cells": capability_cells, "instrument": instrument, "summaries": summaries, "branch_shapley": shapley, "branch_interaction": interaction, "local_retained_fraction": local_retained, "predictions": predictions, "price": price, "terminal": terminal, "reason": {"screen": "local_l9_value_dominates_carrier_content", "null": "local_dominance_or_branch_interaction_prediction_misses", "invalid": "authority_capability_hook_closure_recurrence_coverage_or_price_invalid"}[terminal], "serial_seconds": time.perf_counter() - started, "next_action": "trace local layer-9 value change into its block-9 input factors" if terminal == "screen" else "retain both exact effective-value branches"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "summaries", "branch_shapley", "branch_interaction", "local_retained_fraction", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
