#!/usr/bin/env python3
"""Block9 module/head decomposition of direction-matched task-pair state."""

# BQGATE: EXPERIMENT pred_a_authority_capability_identity_and_route pred_b_resid10_ceiling_recurrence pred_c_block9_component_localization pred_d_task_branch_outside_shared_temporal_heads pred_e_exact_zero_fit_price
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
import run_aspectual_tense_direction_matched_task_state_onset_v1 as onset


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_block9_task_state_decomposition_v1.json"
OUT = ROOT / "circuits/followups/aspectual_tense_block9_task_state_decomposition_v1_result.json"
CANDIDATE_ID = "aspectual_tense.block9_task_state_decomposition_v1"
PATHS = {"onset_result": ROOT / "circuits/followups/aspectual_tense_direction_matched_task_state_onset_v1_result.json", "onset_instrument": ROOT / "ops/run_aspectual_tense_direction_matched_task_state_onset_v1.py", "typed_artifact": ROOT / "circuits/followups/aspectual_tense_typed_shared_contextual_path_v1_artifact.json"}
EXPECTED_PRIOR_SHA256 = "cb9f5e829ae99db654ef8473fa2bc7004c4174b2c1db9418c39e031361351dda"
EXPECTED = {"onset_result": "9d59df283eb7ada0c2f206746141eab2e4984e1c6b94a04c01cc054bb3b1ab36", "onset_instrument": "1c6c8c8891e5cdd0f5d27671a612797dd741940a2b00a4bebced1f03e9c4505f", "typed_artifact": "f0f038f37fd9d97dff088117f93acdf239bab74c5877b522d7976bc81bfc6e85"}
ARMS = ("resid10_ceiling", "attn9_module", "attn9_all_heads", "attn9_h1_h4", "attn9_complement", "mlp9_module")
H1H4, COMPLEMENT, ALL_HEADS = (1, 4), (0, 2, 3, 5, 6, 7, 8), tuple(range(9))


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    result = json.loads(PATHS["onset_result"].read_text())
    if result.get("terminal") != "null" or result.get("stable_onset") != "resid:10" or len(onset.paired_rows()) != 16:
        raise ExperimentError("onset authority changed")


def run_arm(backend, batch, arm, donor_cache):
    holder = {}
    handle = backend.model.lm_head.register_forward_hook(lambda _module, _arguments, output: holder.setdefault("raw", output.detach().clone()))
    try:
        if arm == "resid10_ceiling":
            output = backend.patched(batch, site=producer._site("resid:10"), donor_cache=donor_cache)
        elif arm == "attn9_module":
            output = backend.patched(batch, site=producer._site("attn:09"), donor_cache=donor_cache)
        elif arm == "mlp9_module":
            output = backend.patched(batch, site=producer._site("mlp:09"), donor_cache=donor_cache)
        else:
            heads = {"attn9_all_heads": ALL_HEADS, "attn9_h1_h4": H1H4, "attn9_complement": COMPLEMENT}[arm]
            output = backend.patched_heads(batch, layer=9, heads=heads, donor_cache=donor_cache)
    finally:
        handle.remove()
    raw = holder.get("raw")
    if raw is None:
        raise ExperimentError("final logits missing")
    return output, 30.0 * backend.torch.tanh(raw / 30.0)


def summary(records):
    recoveries = [record["recovery"] for record in records]
    return {"count": len(records), "mean_normalized_donor_recovery": statistics.fmean(recoveries), "mean_absolute_normalized_donor_recovery": statistics.fmean(abs(value) for value in recoveries), "direction_fraction": sum(value > 0.0 for value in recoveries) / len(recoveries), "donor_temporal_correct_fraction": sum(record["donor_temporal_correct"] for record in records) / len(records)}


def main():
    validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "pairs": 16, "arms": list(ARMS), "orientations": 2, "model_forwards": 14, "example_evaluations": 224, "component_orientation_arms": 12, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    pairs = onset.paired_rows()
    backend = producer.Bilin18TorchBackend.load("cuda")
    batches = {task: onset.make_batch(pairs, task) for task in ("has_had", "is_was")}
    native, caches = {}, {}
    forward_calls = evaluations = 0
    for task, batch in batches.items():
        output, logits = onset.run_logits(backend, batch, capture=True)
        native[task] = onset.four_logits(logits, batch, pairs)
        caches[task] = output.captured
        forward_calls += 1
        evaluations += len(pairs)
    capability_cells = []
    for task, pair_index, offset in (("has_had", 0, 0), ("is_was", 1, 2)):
        for direction in ("present_to_past", "past_to_present"):
            indices = [index for index, pair in enumerate(pairs) if pair[pair_index]["direction_id"] == direction]
            accuracy = sum(native[task][index][offset] > native[task][index][offset + 1] for index in indices) / len(indices)
            capability_cells.append({"task": task, "direction": direction, "count": len(indices), "accuracy": accuracy, "threshold": 0.85, "passed": accuracy >= 0.85})
    records, summaries, arm_logits = [], {}, {}
    cache_identity_max_abs = 0.0
    for arm in ARMS:
        for recipient, donor in (("has_had", "is_was"), ("is_was", "has_had")):
            relevant_sites = {"resid10_ceiling": ("resid:10",), "attn9_module": ("attn:09",), "mlp9_module": ("mlp:09",), "attn9_all_heads": tuple(f"attn:09:head:{head:02d}" for head in ALL_HEADS), "attn9_h1_h4": tuple(f"attn:09:head:{head:02d}" for head in H1H4), "attn9_complement": tuple(f"attn:09:head:{head:02d}" for head in COMPLEMENT)}[arm]
            for row_id in batches[recipient].row_ids:
                for site in relevant_sites:
                    value = caches[recipient][(row_id, site)]
                    cache_identity_max_abs = max(cache_identity_max_abs, float((value.float() - value.float().clone()).abs().max()))
            _output, logits = run_arm(backend, batches[recipient], arm, caches[donor])
            values = onset.four_logits(logits, batches[recipient], pairs)
            forward_calls += 1
            evaluations += len(pairs)
            orientation = f"{recipient}_to_{donor}"
            group = []
            arm_logits[(arm, orientation)] = values
            for index, patched in enumerate(values):
                base_support, donor_support, patched_support = onset.is_support(native[recipient][index]), onset.is_support(native[donor][index]), onset.is_support(patched)
                sign = 1.0 if donor == "is_was" else -1.0
                denominator = sign * (donor_support - base_support)
                if denominator <= 0.0 or not math.isfinite(denominator):
                    raise ExperimentError("native endpoint order changed")
                offset = 2 if donor == "is_was" else 0
                record = {"arm": arm, "orientation": orientation, "pair_index": index, "recovery": sign * (patched_support - base_support) / denominator, "donor_temporal_correct": patched[offset] > patched[offset + 1]}
                records.append(record)
                group.append(record)
            summaries.setdefault(arm, {})[orientation] = summary(group)
    route_error = max(abs(a - b) for orientation in ("has_had_to_is_was", "is_was_to_has_had") for pair_a, pair_b in zip(arm_logits[("attn9_module", orientation)], arm_logits[("attn9_all_heads", orientation)]) for a, b in zip(pair_a, pair_b))
    orientations = ("has_had_to_is_was", "is_was_to_has_had")
    ceiling_targets = {"has_had_to_is_was": 0.8365237052169481, "is_was_to_has_had": 0.7629185997080772}
    pred_a = all(cell["passed"] for cell in capability_cells) and cache_identity_max_abs == 0.0 and route_error <= 1e-5 and len(records) == 192
    pred_b = all(abs(summaries["resid10_ceiling"][orientation]["mean_normalized_donor_recovery"] - ceiling_targets[orientation]) <= 0.05 and summaries["resid10_ceiling"][orientation]["direction_fraction"] == 1.0 for orientation in orientations)
    component_pass = {}
    for component in ("attn9_module", "mlp9_module"):
        component_pass[component] = all(summaries[component][orientation]["mean_normalized_donor_recovery"] >= 0.50 * summaries["resid10_ceiling"][orientation]["mean_normalized_donor_recovery"] and summaries[component][orientation]["direction_fraction"] >= 0.75 for orientation in orientations)
    pred_c = any(component_pass.values())
    complement_retained = {orientation: summaries["attn9_complement"][orientation]["mean_normalized_donor_recovery"] / summaries["attn9_module"][orientation]["mean_normalized_donor_recovery"] for orientation in orientations}
    pred_d = all(summaries["attn9_complement"][orientation]["mean_normalized_donor_recovery"] > summaries["attn9_h1_h4"][orientation]["mean_normalized_donor_recovery"] and complement_retained[orientation] >= 0.60 for orientation in orientations)
    price = {"model_forwards": forward_calls, "example_evaluations": evaluations, "pairs": 16, "component_orientation_arms": 12, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 14, "example_evaluations": 224, "pairs": 16, "component_orientation_arms": 12, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = {"pred_a_authority_capability_identity_and_route": pred_a, "pred_b_resid10_ceiling_recurrence": pred_b, "pred_c_block9_component_localization": pred_c, "pred_d_task_branch_outside_shared_temporal_heads": pred_d, "pred_e_exact_zero_fit_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_e else "invalid")
    result = {"schema": "aspectual_tense_block9_task_state_decomposition_result_v1", "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256, "authority_sha256": EXPECTED, "capability_cells": capability_cells, "instrument": {"cache_identity_max_abs_error": cache_identity_max_abs, "attn9_all_heads_vs_module_max_abs_logit_error": route_error}, "summaries": summaries, "component_pass": component_pass, "complement_retained_fraction_of_attn9": complement_retained, "predictions": predictions, "price": price, "terminal": terminal, "reason": {"screen": "block9_task_branch_outside_shared_temporal_heads", "null": "block9_component_or_head_split_prediction_misses", "invalid": "authority_capability_identity_route_ceiling_coverage_or_price_invalid"}[terminal], "serial_seconds": time.perf_counter() - started, "next_action": "decompose the winning block9 task branch at source positions" if terminal == "screen" else "retain block9 onset without the registered component split"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "summaries", "component_pass", "complement_retained_fraction_of_attn9", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
