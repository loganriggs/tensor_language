#!/usr/bin/env python3
"""Split the fresh causal A11 reader edge into exact head-output endpoints."""

# BQGATE: EXPERIMENT pred_a_authority_source_replay_capture_hook_coverage_finiteness_and_exact_price pred_b_all_nine_head_removal_reproduces_the_parent_A11_reader_loss pred_c_prospective_H3_is_a_stable_majority_A11_endpoint pred_d_fit_ordered_prefix_of_at_most_three_heads_recovers_A11_loss_on_holdout pred_e_no_v17_head_outcome_was_used_to_choose_rows_source_boundary_or_H3
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v17 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
import run_temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v17_a11_head_endpoint_ordered_cumulative_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v17.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v17_a11_head_endpoint_ordered_cumulative_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v17_a11_head_endpoint_ordered_cumulative_v1"
EXPECTED = {
    "prior": "7b53830fa452211ac2f445466141d2ba76896f0e5c8e08f25bb3075894152e25",
    "parent_result": "4883354bf39fe6bfe9ffd5c2f97475be749a0aec78d8aa13985739bcf4b93062",
    "parent_runner": "a34217de464b988f10be94b60d7526146b9f86c4ad717502e4dfe852e1f2b0d7",
    "builder": "7e59800324e5b6a4a9c5af564cbe8fb5cf642cfd6e702e1e97c32bc0cc54fe9e",
}
HEADS = tuple(range(9))
ENDPOINT_ARMS = (("no_heads_removed", ()), ("all_heads_removed", HEADS)) + tuple(
    (f"remove_H{head}", (head,)) for head in HEADS
) + tuple((f"keep_only_H{head}", tuple(item for item in HEADS if item != head))
          for head in HEADS)
PRICE = {"checkpoint_loads": 1, "model_forwards": 31, "sequence_evaluations": 496,
         "scored_token_positions": 992, "endpoint_arms": 20, "cumulative_arms": 8,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
BARS = {"replay": 1e-5, "output": 1e-5, "parent_loss_agreement": .03,
        "h3_first": .03, "h3_last": .03, "h3_share": .50,
        "compact_max": 3, "compact_fraction": .80, "direction": .875}
PREDICTION_KEYS = (
    "pred_a_authority_source_replay_capture_hook_coverage_finiteness_and_exact_price",
    "pred_b_all_nine_head_removal_reproduces_the_parent_A11_reader_loss",
    "pred_c_prospective_H3_is_a_stable_majority_A11_endpoint",
    "pred_d_fit_ordered_prefix_of_at_most_three_heads_recovers_A11_loss_on_holdout",
    "pred_e_no_v17_head_outcome_was_used_to_choose_rows_source_boundary_or_H3",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def replace_removed_heads(current, absent, removed, position_rows, n_heads=9):
    removed = tuple(int(head) for head in removed)
    if current.shape != absent.shape or current.ndim != 3 or current.shape[-1] % n_heads:
        raise ValueError("A11 head tensors changed geometry")
    if len(removed) != len(set(removed)) or any(head < 0 or head >= n_heads for head in removed):
        raise ValueError("removed heads must be unique valid indices")
    if len(position_rows) != current.shape[0]:
        raise ValueError("position rows changed coverage")
    width, changed = current.shape[-1] // n_heads, current.clone()
    for row, positions in enumerate(position_rows):
        if len(positions) != len(set(positions)) or any(
            position < 0 or position >= current.shape[1] for position in positions
        ):
            raise ValueError("head replacement position is invalid")
        for head in removed:
            sl = slice(head * width, (head + 1) * width)
            changed[row, list(positions), sl] = absent[row, list(positions), sl].to(changed)
    return changed


def capture_a11(forward, model):
    saved, calls = {}, {"input": 0, "output": 0}
    def pre(_module, arguments):
        calls["input"] += 1
        saved["input"] = arguments[0].detach().clone()
    def post(_module, _arguments, output):
        calls["output"] += 1
        saved["output"] = output.detach().clone()
    pre_handle = model.transformer.h[11].attn.c_proj.register_forward_pre_hook(pre)
    post_handle = model.transformer.h[11].attn.c_proj.register_forward_hook(post)
    try:
        result = forward()
    finally:
        pre_handle.remove(); post_handle.remove()
    if calls != {"input": 1, "output": 1} or set(saved) != {"input", "output"}:
        raise RuntimeError(f"A11 capture coverage changed: {calls}")
    return result, saved, calls


def run_arm(backend, batch, base_heads, donor_heads, source_positions,
            absent_a11, positions, removed):
    calls, saved_output = {"a11": 0}, {}
    def a11_pre(_module, arguments):
        calls["a11"] += 1
        changed = replace_removed_heads(arguments[0], absent_a11, removed, positions)
        return (changed,) + tuple(arguments[1:])
    def a11_post(_module, _arguments, output):
        saved_output["value"] = output.detach().clone()
    pre_handle = backend.model.transformer.h[11].attn.c_proj.register_forward_pre_hook(a11_pre)
    post_handle = backend.model.transformer.h[11].attn.c_proj.register_forward_hook(a11_post)
    try:
        output, source_calls = parent.with_source_heads(
            lambda: parent.full_forward(backend, batch), backend.model, batch,
            base_heads, donor_heads, source_positions)
    finally:
        pre_handle.remove(); post_handle.remove()
    if calls["a11"] != 1 or set(saved_output) != {"value"}:
        raise RuntimeError(f"A11 arm coverage changed: {calls}")
    return output, {"a11": calls["a11"], "source": source_calls}, saved_output["value"]


def phase_mask(rows, phase):
    return np.asarray([row["group_number"] < 8 if phase == "FIT"
                       else row["group_number"] >= 8 for row in rows])


def direction(value, reference):
    return float(np.mean(np.asarray(value) * np.asarray(reference) > 0))


def main():
    paths = {"prior": PRIOR, "parent_result": PARENT_RESULT,
             "parent_runner": PARENT_RUNNER, "builder": BUILDER}
    observed = {name: sha(path) for name, path in paths.items()}
    parent_result = json.loads(PARENT_RESULT.read_text())
    rows = [row for row in fresh.build_rows() if row["transform_id"] == "A2"]
    authority_ok = bool(observed == EXPECTED and len(rows) == 16
        and parent_result.get("terminal") == "partial_directed_reader_interface"
        and parent_result.get("predictions", {}).get(
            "pred_b_A11_reader_loss_is_stable_and_complete_A11_output_rescues") is True)
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "rows": len(rows), "heads": HEADS,
           "endpoint_arms": [label for label, _removed in ENDPOINT_ARMS], "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok:
        raise RuntimeError("v17 A11 head endpoint authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda")
    batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    positions = tuple(tuple(range(int(query) + 1)) for query in batch.semantic_positions)
    source_positions = parent.postcue_rows(batch, donor_batch)
    calls = {}
    with backend.torch.no_grad():
        native_bundle, base_heads, calls["native_source_capture"] = parent.capture_source_heads(
            lambda: capture_a11(lambda: parent.full_forward(backend, batch), backend.model),
            backend.model, batch)
        native, absent_a11, calls["native_a11_capture"] = native_bundle
        donor, donor_heads, calls["donor_source_capture"] = parent.capture_source_heads(
            lambda: parent.full_forward(backend, donor_batch), backend.model, donor_batch)
        writer_bundle, calls["writer_source_patch"] = parent.with_source_heads(
            lambda: capture_a11(lambda: parent.full_forward(backend, batch), backend.model),
            backend.model, batch, base_heads, donor_heads, source_positions)
        writer, present_a11, calls["writer_a11_capture"] = writer_bundle

        outputs, a11_outputs = {}, {}
        for label, removed in ENDPOINT_ARMS:
            outputs[label], calls[label], a11_outputs[label] = run_arm(
                backend, batch, base_heads, donor_heads, source_positions,
                absent_a11["input"], positions, removed)

        native_margin = parent.toward_donor_margin(native)
        writer_margin = parent.toward_donor_margin(writer)
        reference = writer_margin - native_margin
        margins = {label: parent.toward_donor_margin(output) for label, output in outputs.items()}
        recovery = {}
        endpoint = {f"H{head}": {} for head in HEADS}
        for phase in ("FIT", "HOLDOUT"):
            chosen = phase_mask(rows, phase)
            recovery[phase] = {label: parent.metrics(margin[chosen] - native_margin[chosen],
                                                     reference[chosen])
                               for label, margin in margins.items()}
            full_loss_vector = writer_margin[chosen] - margins["all_heads_removed"][chosen]
            for head in HEADS:
                first_vector = writer_margin[chosen] - margins[f"remove_H{head}"][chosen]
                last_vector = margins[f"keep_only_H{head}"][chosen] - margins["all_heads_removed"][chosen]
                first = recovery[phase]["no_heads_removed"]["signed_recovery"] - recovery[phase][f"remove_H{head}"]["signed_recovery"]
                last = recovery[phase][f"keep_only_H{head}"]["signed_recovery"] - recovery[phase]["all_heads_removed"]["signed_recovery"]
                endpoint[f"H{head}"][phase] = {"first_loss": first, "last_loss": last,
                    "endpoint_attribution": .5 * (first + last),
                    "first_direction": direction(first_vector, full_loss_vector),
                    "last_direction": direction(last_vector, full_loss_vector)}
        fit_order = tuple(sorted(HEADS, key=lambda head:
            (-endpoint[f"H{head}"]["FIT"]["endpoint_attribution"], head)))
        prefix_outputs = {}
        for length in range(1, 9):
            label = f"fit_order_prefix_{length}"
            prefix_outputs[label], calls[label], a11_outputs[label] = run_arm(
                backend, batch, base_heads, donor_heads, source_positions,
                absent_a11["input"], positions, fit_order[:length])

    prefix_margin = {label: parent.toward_donor_margin(output)
                     for label, output in prefix_outputs.items()}
    prefixes = {phase: {} for phase in ("FIT", "HOLDOUT")}
    for phase in prefixes:
        chosen = phase_mask(rows, phase)
        full_loss = writer_margin[chosen] - margins["all_heads_removed"][chosen]
        full_scalar = 1.0 - recovery[phase]["all_heads_removed"]["signed_recovery"]
        for length in range(1, 9):
            label = f"fit_order_prefix_{length}"
            report = parent.metrics(prefix_margin[label][chosen] - native_margin[chosen], reference[chosen])
            loss = 1.0 - report["signed_recovery"]
            prefixes[phase][label] = {**report, "loss": loss,
                "a11_loss_fraction": loss / full_scalar if full_scalar > 1e-12 else 0.0,
                "loss_direction": direction(writer_margin[chosen] - prefix_margin[label][chosen],
                                             full_loss)}

    replay_error = float(np.max(np.abs(margins["no_heads_removed"] - writer_margin)))
    output_errors = []
    for row, row_positions in enumerate(positions):
        if row_positions:
            index = list(row_positions)
            output_errors.append(float((a11_outputs["all_heads_removed"][row, index].float()
                - absent_a11["output"][row, index].float()).abs().max()))
    output_error = max(output_errors, default=0.0)
    a11_loss = {phase: 1.0 - recovery[phase]["all_heads_removed"]["signed_recovery"]
                for phase in ("FIT", "HOLDOUT")}
    loss_agreement = {phase: abs(a11_loss[phase] - parent_result["summaries"][phase]["A11"]["loss"])
                      for phase in ("FIT", "HOLDOUT")}
    hook_ok = finite(calls) and all(
        item.get("a11") == 1 and set(item.get("source", {}).values()) == {1}
        for key, item in calls.items() if key.startswith(("remove_", "keep_", "fit_order_", "no_", "all_")))
    A = bool(authority_ok and replay_error <= BARS["replay"] and hook_ok
             and finite({"recovery": recovery, "endpoint": endpoint, "prefixes": prefixes})
             and PRICE["model_forwards"] * len(rows) == PRICE["sequence_evaluations"])
    B = bool(output_error <= BARS["output"]
             and all(loss_agreement[phase] <= BARS["parent_loss_agreement"]
                     for phase in ("FIT", "HOLDOUT")))
    C = all(endpoint["H3"][phase]["first_loss"] >= BARS["h3_first"]
        and endpoint["H3"][phase]["last_loss"] >= BARS["h3_last"]
        and endpoint["H3"][phase]["first_direction"] >= BARS["direction"]
        and endpoint["H3"][phase]["last_direction"] >= BARS["direction"]
        and endpoint["H3"][phase]["endpoint_attribution"] / max(a11_loss[phase], 1e-30)
            >= BARS["h3_share"] for phase in ("FIT", "HOLDOUT"))
    compact_lengths = [length for length in range(1, BARS["compact_max"] + 1)
        if all(prefixes[phase][f"fit_order_prefix_{length}"]["a11_loss_fraction"]
                   >= BARS["compact_fraction"]
               and prefixes[phase][f"fit_order_prefix_{length}"]["loss_direction"]
                   >= BARS["direction"] for phase in ("FIT", "HOLDOUT"))]
    D, E = bool(compact_lengths), True
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A or not B else
        "h3_compact_A11_endpoint" if C and D else
        "compact_non_H3_A11_endpoint" if D else "distributed_A11_endpoint")
    result = {"schema": "temporal_iswas_v17_a11_head_endpoint_ordered_cumulative_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "fit_head_order": fit_order, "compact_prefix_lengths": compact_lengths,
        "instrument": {"writer_replay_max_abs_margin_error": replay_error,
                       "all_head_output_max_abs_error": output_error, "calls": calls},
        "a11_loss": a11_loss, "parent_loss_agreement": loss_agreement,
        "endpoint_attribution": endpoint, "endpoint_reports": recovery,
        "prefix_reports": prefixes, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("fit_head_order", "compact_prefix_lengths",
        "instrument", "a11_loss", "parent_loss_agreement", "endpoint_attribution",
        "prefix_reports", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
