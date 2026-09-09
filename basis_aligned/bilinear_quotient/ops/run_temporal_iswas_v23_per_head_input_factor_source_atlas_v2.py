#!/usr/bin/env python3
"""Conditional per-head Q1/K1/Q2/K2/value and token-source atlas for v23."""

# BQGATE: EXPERIMENT pred_a_authority_dependency_captures_closures_finiteness_enumeration_and_exact_price pred_b_every_full_head_replays_immutable_parent_singleton pred_c_changed_temporal_tokens_dominate_at_least_three_heads pred_d_at_least_two_heads_have_a_proper_selective_factor_subset pred_e_at_least_one_head_pair_has_split_stable_different_factor_profiles
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

from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v23_four_head_input_factor_source_atlas_v1 as union


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_per_head_input_factor_source_atlas_v2.json"
AGGREGATE = ROOT / "circuits/followups/temporal_iswas_v23_four_head_input_factor_source_atlas_v1_result.json"
PARENT = union.PARENT
CAPABILITY = union.CAPABILITY
BUILDER = union.BUILDER
ATTENTION = union.ATTENTION
SOURCE = union.SOURCE
SHARED = union.SHARED
DAS = union.DAS
PRODUCER = union.PRODUCER
UNION_SCRIPT = Path(union.__file__).resolve()
OUT = ROOT / "circuits/followups/temporal_iswas_v23_per_head_input_factor_source_atlas_v2_result.json"
EXPECTED = {
    "prior": "5e09a8430af3dcd66100215eb4dc21b78a36b480074704463d6be639d78fffa6",
    "parent": "db850d5e9b86f76cb4381a12fc83f91cd2aac3544029fabd18bad138d34fed92",
    "capability": "35e6cb352c276a4e1b3643cc2f5795bde667c06b8d7987352b198ccce10a4856",
    "builder": "a4830fd110b8cd854a5f28bfae776f697a15d4f791355990e02ea030fdca4c05",
    "attention": "273298e57a1b2fc5d0be0a50f2976a727b07fdff03f227299c2578e20314f708",
    "source": "1c965deb1969cd317185f4d068006d7b9052837d1603898e0d5372076b48d0a8",
    "shared": "9ab2a9edb60f4e3e4befebf11f55225659559a2a504075567218eab9bf903d06",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
    "union_script": "5ed0b87af30a4728ecc50f60930b7d664301ed0f7e7acab8f181d88077c908cc",
}
ROWS_SHA256 = union.ROWS_SHA256
ROUTES = union.ROUTES
PANELS = union.PANELS
BARS = {
    "closure": 1e-4, "parent_replay": 2e-3, "target_recovery": .80,
    "target_cosine": .90, "target_direction": .90, "control": .25,
    "changed_allocation": .40, "changed_head_count": 3,
    "proper_head_count": 2, "profile_distance": .35,
    "half_profile_distance": .20,
}
PRICE = {
    "checkpoint_loads": 1, "model_forwards_exact": 156,
    "sequence_evaluations_exact": 9984, "transformer_backwards": 0,
    "model_updates": 0, "fit_parameters": 0,
}
PREDICTION_KEYS = (
    "pred_a_authority_dependency_captures_closures_finiteness_enumeration_and_exact_price",
    "pred_b_every_full_head_replays_immutable_parent_singleton",
    "pred_c_changed_temporal_tokens_dominate_at_least_three_heads",
    "pred_d_at_least_two_heads_have_a_proper_selective_factor_subset",
    "pred_e_at_least_one_head_pair_has_split_stable_different_factor_profiles",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def dependency_status():
    if not AGGREGATE.exists():
        return {"status": "pending"}
    result = json.loads(AGGREGATE.read_text())
    predictions = result.get("predictions", {})
    valid = bool(
        result.get("candidate_id") ==
        "cross_task.temporal_iswas.v23_four_head_input_factor_source_atlas_v1"
        and predictions.get("pred_a_authority_population_partition_finiteness_enumeration_and_exact_price") is True
        and predictions.get("pred_b_factor_source_closure_and_prior_union_replay") is True
        and result.get("terminal") != "invalid_instrument"
        and result.get("v24_accessed") is False
    )
    return {
        "status": "valid" if valid else "invalid",
        "sha256": sha256(AGGREGATE), "terminal": result.get("terminal"),
        "predictions": {key: predictions.get(key) for key in sorted(predictions)},
    }


def normalized_profile(allocations, names):
    vector = np.asarray([allocations[name] for name in names], dtype=np.float64)
    scale = float(np.abs(vector).sum())
    return vector / max(scale, 1e-30)


def pair_profile_distances(games, half_games, names):
    report = {}
    for left, right in itertools.combinations(sorted(games), 2):
        key = f"{left}__{right}"
        report[key] = {
            "overall_l1": float(np.abs(
                normalized_profile(games[left]["allocations"], names)
                - normalized_profile(games[right]["allocations"], names)).sum()),
            "halves": {
                half: float(np.abs(
                    normalized_profile(half_games[left][half]["allocations"], names)
                    - normalized_profile(half_games[right][half]["allocations"], names)).sum())
                for half in ("first", "second")
            },
        }
    return report


def selective(report):
    return bool(
        report["target"]["signed_projection"] >= BARS["target_recovery"]
        and report["target"]["cosine"] >= BARS["target_cosine"]
        and report["target"]["direction_fraction"] >= BARS["target_direction"]
        and all(report["controls"][panel] <= BARS["control"] for panel in ("P", "C"))
    )


def main():
    paths = {
        "prior": PRIOR, "parent": PARENT, "capability": CAPABILITY, "builder": BUILDER,
        "attention": ATTENTION, "source": SOURCE, "shared": SHARED, "das": DAS,
        "producer": PRODUCER, "union_script": UNION_SCRIPT,
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    dependency = dependency_status()
    capability = json.loads(CAPABILITY.read_text())
    parent = json.loads(PARENT.read_text())
    rows = union.fresh.build_rows()
    counts = {panel: sum(row["family"] == panel for row in rows) for panel in PANELS}
    capable = set(capability["jointly_capable_row_ids"]["A1"] + capability["jointly_capable_row_ids"]["A2"])
    target_rows = [index for index, row in enumerate(rows) if row["row_id"] in capable]
    static_authority = bool(
        observed == EXPECTED and union.fresh.authority_sha256() == ROWS_SHA256
        and counts == {panel: 16 for panel in PANELS} and len(target_rows) == 30
        and parent.get("terminal") == "confirmed_selective_four_head_writer_program"
        and set(parent.get("singleton_reports", {})) == {f"L{layer}H{head}" for layer, head in ROUTES}
        and tuple(union.source.SOURCE_FACTORS) == ("q", "k", "q2", "k2", "u")
        and tuple(union.source.SOURCE_GROUPS) == ("changed", "unchanged_prefix", "matched_suffix")
    )
    dryrun = {
        "candidate_id": "cross_task.temporal_iswas.v23_per_head_input_factor_source_atlas_v2",
        "dryrun": True, "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "static_authority_ok": static_authority, "dependency": dependency,
        "counts": counts, "target_rows": len(target_rows),
        "routes": [f"L{layer}H{head}" for layer, head in ROUTES],
        "factor_arms_per_head": 32, "source_arms_per_head": 8,
        "capture_forwards": 6, "bars": BARS, "price": PRICE,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if not static_authority or dependency.get("status") != "valid":
        raise RuntimeError(f"v23 per-head authority/dependency invalid: {dependency}")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = now(), time.perf_counter()
    backend = union.producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    base_batch = union.das._batch(backend, rows, side="base")
    donor_batch = union.das._batch(backend, rows, side="donor")
    base_logits, base_lengths, _, _, _, _ = union.shared.capture_native(backend, base_batch, False)
    donor_logits, donor_lengths, _, _, _, _ = union.shared.capture_native(backend, donor_batch, False)
    forwards = 2
    captures = {"base": {}, "donor": {}}
    closure_error = 0.0
    with torch.no_grad():
        for side, batch in (("base", base_batch), ("donor", donor_batch)):
            for layer in sorted(union.ROUTES_BY_LAYER):
                _output, capture = union.attention_eval.capture_layer_attention(
                    backend, batch, layer, include_qk_factors=True)
                captures[side][layer] = capture
                closure_error = max(closure_error, float(capture["reconstruction_max_abs"]))
                forwards += 1
    tokens, _lengths = backend._tensor_batch(base_batch)
    groups = union.source.batch_token_role_partitions(rows, tokens.shape[1], torch, device=backend.device)

    def factors(side, layer, head):
        capture = captures[side][layer]
        return {name: capture[name][:, :, head].float() for name in ("q", "k", "q2", "k2")}

    def complete_factors(side, layer, head):
        result = factors(side, layer, head)
        result["u"] = captures[side][layer]["value"][:, :, head].float()
        return result

    def make_cache(layer, head, *, factor_mask=None, role_mask=None):
        nonlocal closure_error
        base_capture = captures["base"][layer]
        native = complete_factors("base", layer, head)
        donor = complete_factors("donor", layer, head)
        if factor_mask is not None:
            grouped = union.source.mixed_grouped_query_source_writes(
                native, donor, union.source.factor_names(factor_mask), groups, torch)
            mixed = grouped.sum(2)
            expected = base_capture["head_output"][:, :, head] if factor_mask == 0 else (
                captures["donor"][layer]["head_output"][:, :, head] if factor_mask == 31 else None)
        else:
            native_terms = union.source.mixed_grouped_query_source_writes(
                native, donor, (), groups, torch)
            donor_terms = union.source.mixed_grouped_query_source_writes(
                native, donor, union.source.SOURCE_FACTORS, groups, torch)
            selected = torch.as_tensor(
                [bool(role_mask & (1 << index)) for index in range(3)],
                device=backend.device)[None, None, :, None]
            mixed = torch.where(selected, donor_terms, native_terms).sum(2)
            expected = base_capture["head_output"][:, :, head] if role_mask == 0 else (
                captures["donor"][layer]["head_output"][:, :, head] if role_mask == 7 else None)
        if expected is not None:
            for index, endpoint in enumerate(base_batch.semantic_positions):
                closure_error = max(closure_error, float((
                    mixed[index, :int(endpoint) + 1] - expected[index, :int(endpoint) + 1].float()
                ).abs().max()))
        changed = union.assemble_layer_head_outputs(base_capture["head_output"], {head: mixed})
        return {f"attn:{layer}": changed.reshape(changed.shape[0], changed.shape[1], -1)}

    base_margin = union.shared.margin(torch, base_logits, base_lengths, rows, backend.device)
    donor_margin = union.shared.margin(torch, donor_logits, donor_lengths, rows, backend.device)
    native_delta = donor_margin - base_margin
    target_index = torch.as_tensor(target_rows, device=backend.device)
    control_index = {panel: torch.as_tensor(
        [index for index, row in enumerate(rows) if row["family"] == panel], device=backend.device)
        for panel in ("P", "C")}
    half_index = {name: torch.as_tensor(
        [index for index in target_rows if int(rows[index]["group_number"]) % 4 in residues],
        device=backend.device)
        for name, residues in (("first", (0, 1)), ("second", (2, 3)))}

    def report(delta, reference, control_scale):
        return {
            "target": union.shared.metrics(torch, delta[target_index], reference[target_index]),
            "controls": {panel: union.shared.rms_ratio(delta[index], control_scale)
                         for panel, index in control_index.items()},
            "halves": {name: union.shared.metrics(torch, delta[index], reference[index])
                       for name, index in half_index.items()},
        }

    native_scale = native_delta[target_index]
    factor_deltas, source_deltas = {}, {}
    with torch.no_grad():
        for layer, head in ROUTES:
            route = f"L{layer}H{head}"
            make_cache(layer, head, factor_mask=0)
            factor_deltas[route] = {0: torch.zeros_like(base_margin)}
            for mask in range(1, 32):
                logits, lengths, _final, _x = union.shared.run_patch(
                    backend, base_batch, make_cache(layer, head, factor_mask=mask), routes=(route,))
                factor_deltas[route][mask] = (
                    union.shared.margin(torch, logits, lengths, rows, backend.device) - base_margin)
                forwards += 1
            source_deltas[route] = {0: torch.zeros_like(base_margin), 7: factor_deltas[route][31]}
            for mask in range(1, 7):
                logits, lengths, _final, _x = union.shared.run_patch(
                    backend, base_batch, make_cache(layer, head, role_mask=mask), routes=(route,))
                source_deltas[route][mask] = (
                    union.shared.margin(torch, logits, lengths, rows, backend.device) - base_margin)
                forwards += 1

    reports, factor_games, source_games = {}, {}, {}
    half_factor_games, half_source_games = {}, {}
    proper_masks = {}
    replay_errors = {}
    for route in sorted(factor_deltas):
        full = factor_deltas[route][31]
        head_scale = full[target_index]
        factor_relative = {mask: report(delta, full, head_scale)
                           for mask, delta in factor_deltas[route].items()}
        source_relative = {mask: report(delta, full, head_scale)
                           for mask, delta in source_deltas[route].items()}
        reports[route] = {
            "full_vs_native": report(full, native_delta, native_scale),
            "factor_relative": factor_relative, "source_relative": source_relative,
        }
        factor_games[route] = union.shapley(
            [factor_relative[mask]["target"]["signed_projection"] for mask in range(32)],
            union.source.SOURCE_FACTORS)
        source_games[route] = union.shapley(
            [source_relative[mask]["target"]["signed_projection"] for mask in range(8)],
            union.source.SOURCE_GROUPS)
        half_factor_games[route] = {half: union.shapley(
            [factor_relative[mask]["halves"][half]["signed_projection"] for mask in range(32)],
            union.source.SOURCE_FACTORS) for half in ("first", "second")}
        half_source_games[route] = {half: union.shapley(
            [source_relative[mask]["halves"][half]["signed_projection"] for mask in range(8)],
            union.source.SOURCE_GROUPS) for half in ("first", "second")}
        passing = [mask for mask in range(1, 31) if selective(factor_relative[mask])]
        proper_masks[route] = {
            "all": passing,
            "canonical": min(passing, key=lambda mask: (mask.bit_count(), mask)) if passing else None,
            "canonical_names": list(union.source.factor_names(
                min(passing, key=lambda mask: (mask.bit_count(), mask)))) if passing else None,
        }
        actual = reports[route]["full_vs_native"]
        expected = parent["singleton_reports"][route]
        differences = [abs(actual["target"][key] - expected["target"]["behavior"][key])
                       for key in ("signed_projection", "cosine", "direction_fraction",
                                   "relative_residual", "norm_ratio")]
        differences += [abs(actual["controls"][panel]
                            - expected["controls"][panel]["behavior_leak_ratio"])
                        for panel in ("P", "C")]
        replay_errors[route] = max(differences)

    distances = pair_profile_distances(
        factor_games, half_factor_games, union.source.SOURCE_FACTORS)
    changed_heads = [route for route in sorted(source_games)
                     if source_games[route]["allocations"]["changed"] >= BARS["changed_allocation"]
                     and source_games[route]["allocations"]["changed"] ==
                         max(source_games[route]["allocations"].values())
                     and all(half_source_games[route][half]["allocations"]["changed"] > 0
                             for half in ("first", "second"))]
    proper_heads = [route for route in sorted(proper_masks) if proper_masks[route]["all"]]
    distinct_pairs = [pair for pair, values in distances.items()
                      if values["overall_l1"] >= BARS["profile_distance"]
                      and all(values["halves"][half] >= BARS["half_profile_distance"]
                              for half in ("first", "second"))]
    finite_payload = [reports, factor_games, source_games, half_factor_games,
                      half_source_games, distances, replay_errors]
    A = bool(
        static_authority and dependency.get("status") == "valid"
        and forwards == PRICE["model_forwards_exact"]
        and forwards * len(rows) == PRICE["sequence_evaluations_exact"]
        and union.finite(finite_payload)
    )
    B = bool(closure_error <= BARS["closure"]
             and max(replay_errors.values()) <= BARS["parent_replay"])
    C = len(changed_heads) >= BARS["changed_head_count"]
    D = len(proper_heads) >= BARS["proper_head_count"]
    E = bool(distinct_pairs)
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = "invalid_instrument" if not (A and B) else (
        "per_head_specialized_input_programs" if E else "shared_input_program_screen")
    result = {
        "schema": "temporal_iswas_v23_per_head_input_factor_source_atlas_result_v2",
        "candidate_id": dryrun["candidate_id"], "started_utc": started_utc,
        "finished_utc": now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": observed, "aggregate_dependency": dependency,
        "population": {"counts": counts, "target_rows": len(target_rows)},
        "routes": dryrun["routes"], "factor_names": list(union.source.SOURCE_FACTORS),
        "source_groups": list(union.source.SOURCE_GROUPS), "closure_max_abs": closure_error,
        "parent_singleton_replay_max_abs": replay_errors,
        "reports": reports, "factor_games": factor_games, "source_games": source_games,
        "half_factor_games": half_factor_games, "half_source_games": half_source_games,
        "factor_profile_pair_distances": distances, "proper_factor_masks": proper_masks,
        "changed_dominant_heads": changed_heads, "proper_selective_heads": proper_heads,
        "split_stable_distinct_profile_pairs": distinct_pairs, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE, "model_forwards": forwards,
        "v24_accessed": False, "fit_parameters": 0,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "closure_max_abs", "parent_singleton_replay_max_abs", "factor_games",
        "source_games", "proper_factor_masks", "changed_dominant_heads",
        "split_stable_distinct_profile_pairs", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__":
    main()
