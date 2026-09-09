#!/usr/bin/env python3
"""Exact all-prefix Q1/K1/Q2/K2/value and token-source atlas for v23."""

# BQGATE: EXPERIMENT pred_a_authority_population_partition_finiteness_enumeration_and_exact_price pred_b_factor_source_closure_and_prior_union_replay pred_c_changed_temporal_tokens_dominate_source_game pred_d_proper_selective_factor_subset_exists pred_e_factor_credit_is_distributed_and_split_stable
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import attention_source_destination_eval as attention_eval
import attention_source_factor_primitive as source
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1 as shared


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_four_head_input_factor_source_atlas_v1.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v23_capability_v1_result.json"
PARENT = ROOT / "circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23.py"
ATTENTION = ROOT / "ops/attention_source_destination_eval.py"
SOURCE = ROOT / "ops/attention_source_factor_primitive.py"
SHARED = ROOT / "ops/run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1.py"
DAS = ROOT / "ops/circuit_das_subspace.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v23_four_head_input_factor_source_atlas_v1_result.json"
EXPECTED = {
    "prior": "22394a2f5a7d8c7b4a0c30bbb93e752a7c0d520120824472ed2cb9049e671a26",
    "capability": "35e6cb352c276a4e1b3643cc2f5795bde667c06b8d7987352b198ccce10a4856",
    "parent": "db850d5e9b86f76cb4381a12fc83f91cd2aac3544029fabd18bad138d34fed92",
    "builder": "a4830fd110b8cd854a5f28bfae776f697a15d4f791355990e02ea030fdca4c05",
    "attention": "273298e57a1b2fc5d0be0a50f2976a727b07fdff03f227299c2578e20314f708",
    "source": "1c965deb1969cd317185f4d068006d7b9052837d1603898e0d5372076b48d0a8",
    "shared": "9ab2a9edb60f4e3e4befebf11f55225659559a2a504075567218eab9bf903d06",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
ROWS_SHA256 = "46e9e493a4b2b42ce0c121b30978f08208537300363fa91902184c8f8d6565c7"
ROUTES = ((8, 1), (9, 1), (9, 4), (11, 3))
ROUTES_BY_LAYER = {
    layer: tuple(head for route_layer, head in ROUTES if route_layer == layer)
    for layer in sorted({layer for layer, _head in ROUTES})
}
PANELS = ("A1", "A2", "P", "C")
BARS = {
    "closure": 1e-4, "parent_replay": 2e-3,
    "target_recovery": .50, "target_cosine": .90, "target_direction": .90,
    "control": .15, "changed_allocation": .25, "factor_allocation": .05,
    "distributed_factor_count": 3,
}
PRICE = {
    "checkpoint_loads": 1, "model_forwards_exact": 45,
    "sequence_evaluations_exact": 2880, "transformer_backwards": 0,
    "model_updates": 0, "fit_parameters": 0,
}
PREDICTION_KEYS = (
    "pred_a_authority_population_partition_finiteness_enumeration_and_exact_price",
    "pred_b_factor_source_closure_and_prior_union_replay",
    "pred_c_changed_temporal_tokens_dominate_source_game",
    "pred_d_proper_selective_factor_subset_exists",
    "pred_e_factor_credit_is_distributed_and_split_stable",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, (int, float)) or isinstance(value, bool) or math.isfinite(float(value))


def shapley(values, names):
    values = np.asarray(values, dtype=np.float64)
    count, allocations = len(names), np.zeros(len(names), dtype=np.float64)
    for index in range(count):
        bit = 1 << index
        for mask in range(1 << count):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            allocations[index] += weight * (values[mask | bit] - values[mask])
    return {
        "allocations": {name: float(allocations[index]) for index, name in enumerate(names)},
        "efficiency_residual": float(abs(allocations.sum() - (values[-1] - values[0]))),
    }


def selective(report):
    target = report["target"]
    return bool(
        target["signed_projection"] >= BARS["target_recovery"]
        and target["cosine"] >= BARS["target_cosine"]
        and target["direction_fraction"] >= BARS["target_direction"]
        and all(report["controls"][panel] <= BARS["control"] for panel in ("P", "C"))
    )


def assemble_layer_head_outputs(base_head_output, replacements):
    """Copy one layer response and replace every declared head without key overwrite."""
    changed = base_head_output.float().clone()
    for head, replacement in replacements.items():
        if tuple(replacement.shape) != tuple(changed[:, :, int(head)].shape):
            raise ValueError("replacement head response has the wrong shape")
        changed[:, :, int(head)] = replacement
    return changed


def main():
    paths = {
        "prior": PRIOR, "capability": CAPABILITY, "parent": PARENT, "builder": BUILDER,
        "attention": ATTENTION, "source": SOURCE, "shared": SHARED, "das": DAS,
        "producer": PRODUCER,
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    capability, parent = json.loads(CAPABILITY.read_text()), json.loads(PARENT.read_text())
    rows = fresh.build_rows()
    counts = {panel: sum(row["family"] == panel for row in rows) for panel in PANELS}
    capable = set(capability["jointly_capable_row_ids"]["A1"] + capability["jointly_capable_row_ids"]["A2"])
    target_rows = [index for index, row in enumerate(rows) if row["row_id"] in capable]
    partitions = source.batch_token_role_partitions(rows, 9, __import__("torch"), device="cpu")
    partition_ok = bool(
        tuple(partitions.shape) == (64, 3, 9)
        and not bool((partitions.sum(1) > 1).any())
        and all(int(partitions[index].sum()) == int(row["base_semantic_position"]) + 1
                for index, row in enumerate(rows))
    )
    authority_ok = bool(
        observed == EXPECTED and fresh.authority_sha256() == ROWS_SHA256
        and counts == {panel: 16 for panel in PANELS} and len(target_rows) == 30
        and capability.get("terminal") == "screen" and all(capability.get("predictions", {}).values())
        and parent.get("terminal") == "confirmed_selective_four_head_writer_program"
        and tuple(source.SOURCE_FACTORS) == ("q", "k", "q2", "k2", "u")
        and tuple(source.SOURCE_GROUPS) == ("changed", "unchanged_prefix", "matched_suffix")
        and partition_ok
    )
    dryrun = {
        "candidate_id": "cross_task.temporal_iswas.v23_four_head_input_factor_source_atlas_v1",
        "dryrun": True, "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "authority_ok": authority_ok, "counts": counts, "target_rows": len(target_rows),
        "routes": [f"L{layer}H{head}" for layer, head in ROUTES],
        "factor_arms": 32, "source_arms": 8, "source_proper_extra_arms": 6,
        "capture_forwards": 6,
        "partition_ok": partition_ok, "bars": BARS, "price": PRICE,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if not authority_ok:
        raise RuntimeError("v23 input factor/source authority invalid")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_logits, base_lengths, _, _, _, _ = shared.capture_native(backend, base_batch, False)
    donor_logits, donor_lengths, _, _, _, _ = shared.capture_native(backend, donor_batch, False)
    forwards = 2
    captures = {"base": {}, "donor": {}}
    reconstruction_error = 0.0
    with torch.no_grad():
        for side, batch in (("base", base_batch), ("donor", donor_batch)):
            for layer, _head in ROUTES:
                if layer in captures[side]:
                    continue
                _output, capture = attention_eval.capture_layer_attention(
                    backend, batch, layer, include_qk_factors=True)
                captures[side][layer] = capture
                reconstruction_error = max(reconstruction_error, float(capture["reconstruction_max_abs"]))
                forwards += 1
    tokens, _lengths = backend._tensor_batch(base_batch)
    groups = source.batch_token_role_partitions(rows, tokens.shape[1], torch, device=backend.device)

    def factors(side, layer, head):
        capture = captures[side][layer]
        return {
            "q": capture["q"][:, :, head].float(), "k": capture["k"][:, :, head].float(),
            "q2": capture["q2"][:, :, head].float(), "k2": capture["k2"][:, :, head].float(),
            "u": capture["value"][:, :, head].float(),
        }

    closure_error = reconstruction_error

    def cache_for_factor(mask):
        nonlocal closure_error
        cache = {}
        for layer, heads in ROUTES_BY_LAYER.items():
            base_capture, donor_capture = captures["base"][layer], captures["donor"][layer]
            replacements = {}
            for head in heads:
                grouped = source.mixed_grouped_query_source_writes(
                    factors("base", layer, head), factors("donor", layer, head),
                    source.factor_names(mask), groups, torch)
                mixed = grouped.sum(2)
                expected = base_capture["head_output"][:, :, head] if mask == 0 \
                    else donor_capture["head_output"][:, :, head] if mask == 31 else None
                if expected is not None:
                    for index, endpoint in enumerate(base_batch.semantic_positions):
                        closure_error = max(closure_error, float(
                            (mixed[index, :int(endpoint) + 1]
                             - expected[index, :int(endpoint) + 1].float()).abs().max()))
                replacements[head] = mixed
            changed = assemble_layer_head_outputs(base_capture["head_output"], replacements)
            cache[f"attn:{layer}"] = changed.reshape(changed.shape[0], changed.shape[1], -1)
        return cache

    def cache_for_roles(role_mask):
        cache = {}
        for layer, heads in ROUTES_BY_LAYER.items():
            base_capture = captures["base"][layer]
            selected = torch.as_tensor(
                [bool(role_mask & (1 << index)) for index in range(len(source.SOURCE_GROUPS))],
                device=backend.device)[None, None, :, None]
            replacements = {}
            for head in heads:
                native = source.mixed_grouped_query_source_writes(
                    factors("base", layer, head), factors("donor", layer, head), (), groups, torch)
                donor = source.mixed_grouped_query_source_writes(
                    factors("base", layer, head), factors("donor", layer, head),
                    source.SOURCE_FACTORS, groups, torch)
                replacements[head] = torch.where(selected, donor, native).sum(2)
            changed = assemble_layer_head_outputs(base_capture["head_output"], replacements)
            cache[f"attn:{layer}"] = changed.reshape(changed.shape[0], changed.shape[1], -1)
        return cache

    base_margin = shared.margin(torch, base_logits, base_lengths, rows, backend.device)
    donor_margin = shared.margin(torch, donor_logits, donor_lengths, rows, backend.device)
    native_delta = donor_margin - base_margin
    target_index = torch.as_tensor(target_rows, device=backend.device)
    control_index = {panel: torch.as_tensor(
        [index for index, row in enumerate(rows) if row["family"] == panel], device=backend.device)
        for panel in ("P", "C")}
    half_index = {name: torch.as_tensor(
        [index for index in target_rows if int(rows[index]["group_number"]) % 4 in residues],
        device=backend.device)
        for name, residues in (("first", (0, 1)), ("second", (2, 3)))}

    def report(logits, lengths):
        delta = shared.margin(torch, logits, lengths, rows, backend.device) - base_margin
        result = {
            "target": shared.metrics(torch, delta[target_index], native_delta[target_index]),
            "controls": {panel: shared.rms_ratio(delta[index], native_delta[target_index])
                         for panel, index in control_index.items()},
            "halves": {name: shared.metrics(torch, delta[index], native_delta[index])
                       for name, index in half_index.items()},
        }
        return result

    zero_report = report(base_logits, base_lengths)
    factor_reports = {0: zero_report}
    with torch.no_grad():
        for mask in range(1, 32):
            logits, lengths, _final, _x = shared.run_patch(
                backend, base_batch, cache_for_factor(mask),
                routes=tuple(f"L{layer}H{head}" for layer, head in ROUTES))
            factor_reports[mask] = report(logits, lengths)
            forwards += 1
        source_reports = {0: zero_report, 7: factor_reports[31]}
        for role_mask in range(1, 7):
            logits, lengths, _final, _x = shared.run_patch(
                backend, base_batch, cache_for_roles(role_mask),
                routes=tuple(f"L{layer}H{head}" for layer, head in ROUTES))
            source_reports[role_mask] = report(logits, lengths)
            forwards += 1

    factor_values = [factor_reports[mask]["target"]["signed_projection"] for mask in range(32)]
    source_values = [source_reports[mask]["target"]["signed_projection"] for mask in range(8)]
    factor_game = shapley(factor_values, source.SOURCE_FACTORS)
    source_game = shapley(source_values, source.SOURCE_GROUPS)
    half_factor_games = {
        half: shapley([factor_reports[mask]["halves"][half]["signed_projection"]
                       for mask in range(32)], source.SOURCE_FACTORS)
        for half in ("first", "second")}
    half_source_games = {
        half: shapley([source_reports[mask]["halves"][half]["signed_projection"]
                       for mask in range(8)], source.SOURCE_GROUPS)
        for half in ("first", "second")}
    full = factor_reports[31]
    parent_union = parent["union_report"]
    replay_values = [
        abs(full["target"][name] - parent_union["target"]["behavior"][name])
        for name in ("signed_projection", "cosine", "direction_fraction", "relative_residual", "norm_ratio")
    ] + [abs(full["controls"][panel] - parent_union["controls"][panel]["behavior_leak_ratio"])
         for panel in ("P", "C")]
    replay_error = max(replay_values)
    proper_selective = [mask for mask in range(1, 31) if selective(factor_reports[mask])]
    changed = source_game["allocations"]["changed"]
    changed_singleton = source_reports[1]
    A = bool(
        authority_ok and forwards == PRICE["model_forwards_exact"]
        and forwards * len(rows) == PRICE["sequence_evaluations_exact"]
        and len(factor_reports) == 32 and len(source_reports) == 8
        and finite([factor_reports, source_reports, factor_game, source_game])
    )
    B = bool(closure_error <= BARS["closure"] and replay_error <= BARS["parent_replay"])
    C = bool(
        changed >= BARS["changed_allocation"]
        and changed == max(source_game["allocations"].values())
        and all(half_source_games[half]["allocations"]["changed"] > 0 for half in ("first", "second"))
        and selective(changed_singleton)
    )
    stable_factors = [name for name in source.SOURCE_FACTORS
                      if factor_game["allocations"][name] >= BARS["factor_allocation"]
                      and all(half_factor_games[half]["allocations"][name] > 0
                              for half in ("first", "second"))]
    D = bool(proper_selective)
    E = len(stable_factors) >= BARS["distributed_factor_count"]
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = "invalid_instrument" if not (A and B) else (
        "changed_source_selective_proper_factor_program" if C and D else
        "valid_input_factor_source_atlas")
    result = {
        "schema": "temporal_iswas_v23_four_head_input_factor_source_atlas_result_v1",
        "candidate_id": dryrun["candidate_id"], "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "population": {"counts": counts, "target_rows": len(target_rows)},
        "routes": dryrun["routes"], "source_groups": list(source.SOURCE_GROUPS),
        "factor_names": list(source.SOURCE_FACTORS), "closure_max_abs": closure_error,
        "parent_replay_max_abs": replay_error,
        "factor_reports": {str(mask): value for mask, value in factor_reports.items()},
        "source_reports": {str(mask): value for mask, value in source_reports.items()},
        "factor_game": factor_game, "source_game": source_game,
        "half_factor_games": half_factor_games, "half_source_games": half_source_games,
        "proper_selective_factor_masks": proper_selective,
        "proper_selective_factor_names": [list(source.factor_names(mask)) for mask in proper_selective],
        "stable_distributed_factors": stable_factors, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE, "model_forwards": forwards,
        "v24_accessed": False, "fit_parameters": 0,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "closure_max_abs", "parent_replay_max_abs", "factor_game", "source_game",
        "proper_selective_factor_names", "stable_distributed_factors", "predictions",
        "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
