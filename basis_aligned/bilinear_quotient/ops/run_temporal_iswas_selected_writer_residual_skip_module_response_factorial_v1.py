#!/usr/bin/env python3
"""Factor block-10 residual carriage from downstream module responses for is-was."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_writer_replay_state_module_capture_finiteness_and_exact_price pred_b_joint_state_and_module_clamp_replays_the_complete_writer_effect pred_c_residual_identity_path_is_a_majority_carrier pred_d_distributed_module_response_is_a_majority_carrier pred_e_two_factor_decomposition_is_additive_and_temporally_selective
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
import run_temporal_iswas_selected_writer_downstream_module_mediation_atlas_v2 as atlas


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_selected_writer_residual_skip_module_response_factorial_v1.json"
ROUTE_RESULT = ROOT / "circuits/followups/temporal_iswas_l11h3_writer_route_closure_ladder_v1_result.json"
ATLAS_RESULT = ROOT / "circuits/followups/temporal_iswas_selected_writer_downstream_module_mediation_atlas_v2_result.json"
ATLAS_RUNNER = ROOT / "ops/run_temporal_iswas_selected_writer_downstream_module_mediation_atlas_v2.py"
GREEDY_RESULT = atlas.GREEDY_RESULT
OUT = ROOT / "circuits/followups/temporal_iswas_selected_writer_residual_skip_module_response_factorial_v1_result.json"
EXPECTED = {
    "prior": "02ba82aff09a369a5065ff69532cf8a6a142e6af8dd6b3ef5089fb5bc168fae3",
    "route_result": "a5401d739775526fa1802c09d7cf633d6ae1253625a32671aabbb822a0754e6c",
    "atlas_result": "88537c2561098239472072b277ab29bef868cc1d648d672024f306fed814990b",
    "atlas_runner": "719394cd2fd134a02afe0365b6603bff4ae3882b5ddd5f2b586930e41b6f3902",
    "greedy_result": "6a2ae6598ae95269f7d9e24e13196155342f690d331c78efed5115a8cf84bda2",
}
ARMS = ("R1M0", "R0M1", "R1M1")
PRICE = {"checkpoint_loads": 1, "model_forwards": 6,
         "sequence_evaluations": 768, "scored_token_positions": 1536,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_pairing_writer_replay_state_module_capture_finiteness_and_exact_price",
    "pred_b_joint_state_and_module_clamp_replays_the_complete_writer_effect",
    "pred_c_residual_identity_path_is_a_majority_carrier",
    "pred_d_distributed_module_response_is_a_majority_carrier",
    "pred_e_two_factor_decomposition_is_additive_and_temporally_selective",
)
BARS = {"replay": 1e-5, "joint_recovery_min": .99, "joint_recovery_max": 1.01,
        "joint_cosine": .9999, "joint_residual": .01, "joint_direction": 1.0,
        "carrier_recovery": .5, "carrier_cosine": .9, "carrier_direction": .9,
        "additivity": .25, "collateral": .01}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def capture_state_and_modules(backend, tokens, *, head_captures=None,
                              selected_labels=(), pairs=None, position_rows=None):
    """Capture the raw block-10 residual input alongside all complete module writes."""
    saved = {}
    calls = 0

    def capture_state(_module, arguments):
        nonlocal calls
        calls += 1
        if len(arguments) != 3 or not hasattr(arguments[0], "detach"):
            raise RuntimeError("block-10 input signature changed")
        saved["residual"] = arguments[0].detach().clone()

    handle = backend.model.transformer.h[10].register_forward_pre_hook(capture_state)
    try:
        logits, writers, l11, modules, module_calls = atlas.capture_modules(
            backend, tokens, head_captures=head_captures,
            selected_labels=selected_labels, pairs=pairs, position_rows=position_rows)
    finally:
        handle.remove()
    if calls != 1 or set(saved) != {"residual"}:
        raise RuntimeError("block-10 residual capture coverage changed")
    return logits, writers, l11, saved["residual"], modules, calls, module_calls


def run_factorial_arm(backend, tokens, residual_replacement, module_replacements,
                      position_rows):
    """Install one residual state and all 16 complete downstream module responses."""
    residual_calls = 0
    module_calls = {name: 0 for name in atlas.MODULES}
    handles = []

    def state_hook(_module, arguments):
        nonlocal residual_calls
        residual_calls += 1
        if len(arguments) != 3:
            raise RuntimeError("block-10 input signature changed")
        changed = atlas.replace_positions(arguments[0], residual_replacement, position_rows)
        return (changed,) + tuple(arguments[1:])

    def module_hook(name):
        def replace(_module, _inputs, output):
            module_calls[name] += 1
            return atlas.replace_positions(output, module_replacements[name], position_rows)
        return replace

    handles.append(backend.model.transformer.h[10].register_forward_pre_hook(state_hook))
    targets = atlas.module_targets(backend.model)
    for name in atlas.MODULES:
        handles.append(targets[name].register_forward_hook(module_hook(name)))
    try:
        logits, _captures = atlas.mediation.parent._forward(backend, tokens)
    finally:
        for handle in handles:
            handle.remove()
    if residual_calls != 1 or set(module_calls.values()) != {1}:
        raise RuntimeError("factorial state/module patch coverage changed")
    return logits, residual_calls, module_calls


def joint_qualified(row):
    return bool(BARS["joint_recovery_min"] <= row["signed_recovery"]
        <= BARS["joint_recovery_max"] and row["cosine"] >= BARS["joint_cosine"]
        and row["relative_residual"] <= BARS["joint_residual"]
        and row["direction_agreement"] >= BARS["joint_direction"])


def carrier_qualified(row):
    return bool(row["signed_recovery"] >= BARS["carrier_recovery"]
        and row["cosine"] >= BARS["carrier_cosine"]
        and row["direction_agreement"] >= BARS["carrier_direction"])


def relative_additivity_error(joint, residual, modules):
    denominator = max(float(np.linalg.norm(joint)), 1e-30)
    return float(np.linalg.norm(joint - residual - modules) / denominator)


def main():
    paths = {"prior": PRIOR, "route_result": ROUTE_RESULT,
             "atlas_result": ATLAS_RESULT, "atlas_runner": ATLAS_RUNNER,
             "greedy_result": GREEDY_RESULT}
    observed = {name: sha(path) for name, path in paths.items()}
    prior = json.loads(PRIOR.read_text())
    dry = {"candidate_id": prior["candidate_id"], "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "authority_ok": observed == EXPECTED, "arms": list(ARMS),
           "modules": list(atlas.MODULES), "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    route = json.loads(ROUTE_RESULT.read_text())
    atlas_result = json.loads(ATLAS_RESULT.read_text())
    greedy = json.loads(GREEDY_RESULT.read_text())
    selected_heads = tuple(greedy["selected_prefixes"]["iswas"]["heads"])
    authority_ok = bool(observed == EXPECTED and prior["price"] == PRICE
        and prior["locked_writer"]["heads"] == list(selected_heads)
        and prior["locked_writer"]["source_region"] == "postcue"
        and selected_heads == ("L07H07", "L09H04")
        and route["terminal"] == "l11h3_bypass_dominant"
        and tuple(route["predictions"].values()) == (True, True, False, False, True)
        and atlas_result["terminal"] == "distributed_or_unstable_bypass"
        and atlas_result["selected_modules"] == []
        and tuple(atlas_result["predictions"].values()) ==
            (True, True, False, False, False))
    if not authority_ok:
        raise RuntimeError("residual/module factorial authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    mediation, loc, parent = atlas.mediation, atlas.mediation.parent.loc, atlas.mediation.parent
    backend = loc.producer.Bilin18TorchBackend.load("cuda")
    rows = loc.original.build_rows()
    endpoints, lookup = loc.parent.endpoint_bank(rows)
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    torch = backend.torch
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints],
                          dtype=torch.long, device=backend.device)
    role, other = "iswas", "temporal"
    pairs = loc.parent.pair_indices(endpoints, lookup, role)
    other_pairs = loc.parent.pair_indices(endpoints, lookup, other)
    queries = [endpoint[f"{role}_position"] - 1 for _row, _cell, endpoint in endpoints]
    regions = [loc.source_regions(endpoint["ids"], endpoints[pairs[index]][2]["ids"],
                                  queries[index])
               for index, (_row, _cell, endpoint) in enumerate(endpoints)]
    source_rows = [item[parent.LOCKED[role]] for item in regions]
    position_rows = [list(range(query + 1)) for query in queries]

    with torch.no_grad():
        (native_logits, head_captures, native_l11, native_state, native_modules,
         native_state_calls, native_module_calls) = capture_state_and_modules(backend, tokens)
        (writer_logits, _unused, _writer_l11, writer_state, writer_modules,
         writer_state_calls, writer_module_calls) = capture_state_and_modules(
            backend, tokens, head_captures=head_captures, selected_labels=selected_heads,
            pairs=pairs, position_rows=source_rows)
        reference_logits, _ = mediation.source._forward(
            backend, tokens, donor_v=native_l11["v"], pairs=pairs,
            value_positions=source_rows)
        arm_logits, arm_state_calls, arm_module_calls = {}, {}, {}
        specifications = {
            "R1M0": (writer_state, native_modules),
            "R0M1": (native_state, writer_modules),
            "R1M1": (writer_state, writer_modules),
        }
        for arm in ARMS:
            arm_logits[arm], arm_state_calls[arm], arm_module_calls[arm] = \
                run_factorial_arm(backend, tokens, *specifications[arm], position_rows)

    native = {name: loc.margins(native_logits, endpoints, name) for name in parent.LOCKED}
    writer_effect = loc.margins(writer_logits, endpoints, role) - native[role]
    writer_other = loc.margins(writer_logits, endpoints, other) - native[other]
    reference_effect = loc.margins(reference_logits, endpoints, role) - native[role]
    gold = native[role][pairs] - native[role]
    other_gold = native[other][other_pairs] - native[other]
    effects = {arm: loc.margins(logits, endpoints, role) - native[role]
               for arm, logits in arm_logits.items()}
    other_effects = {arm: loc.margins(logits, endpoints, other) - native[other]
                     for arm, logits in arm_logits.items()}
    greedy_reports = greedy["panels"]["original"]["reports"]
    reports, additivity, replay_errors = [], {}, []
    for phase in ("FIT", "HOLDOUT"):
        chosen = np.asarray([row["phase"] == phase for row, _cell, _endpoint in endpoints])
        replay = mediation.accounting.effect_metrics(
            writer_effect[chosen], reference_effect[chosen], writer_other[chosen])
        replay_gold = mediation.accounting.effect_metrics(
            writer_effect[chosen], gold[chosen], writer_other[chosen])
        old = next(row for row in greedy_reports if row["role"] == role
            and row["phase"] == phase and row["template_id"] == "ALL"
            and row["arm"] == greedy["selected_prefixes"][role]["arm"])
        for field in ("signed_recovery", "cosine", "relative_residual",
                      "direction_agreement", "non_target_to_target_gold_norm"):
            replay_errors.append(abs(replay[field] - old[field]))
        replay_errors.append(abs(replay_gold["non_target_to_target_gold_norm"]
                                 - old["command_gold_non_target_norm_ratio"]))
        temporal_norm = max(float(np.linalg.norm(other_gold[chosen])), 1e-30)
        for arm in ARMS:
            metric = mediation.accounting.effect_metrics(
                effects[arm][chosen], writer_effect[chosen], other_effects[arm][chosen])
            reports.append({"arm": arm, "phase": phase, **metric,
                "temporal_command_gold_collateral":
                    float(np.linalg.norm(other_effects[arm][chosen]) / temporal_norm)})
        additivity[phase] = relative_additivity_error(
            effects["R1M1"][chosen], effects["R1M0"][chosen], effects["R0M1"][chosen])

    by_phase = {(row["arm"], row["phase"]): row for row in reports}
    A = bool(authority_ok and max(replay_errors) <= BARS["replay"]
        and native_state_calls == writer_state_calls == 1
        and set(native_module_calls.values()) == {1}
        and set(writer_module_calls.values()) == {1}
        and set(arm_state_calls.values()) == {1}
        and all(set(calls.values()) == {1} for calls in arm_module_calls.values())
        and tuple(native_state.shape) == tuple(writer_state.shape)
        and all(tuple(native_modules[name].shape) == tuple(writer_modules[name].shape)
                for name in atlas.MODULES)
        and finite(reports) and finite(additivity) and PRICE == prior["price"])
    B = all(joint_qualified(by_phase[("R1M1", phase)]) for phase in ("FIT", "HOLDOUT"))
    C = all(carrier_qualified(by_phase[("R1M0", phase)]) for phase in ("FIT", "HOLDOUT"))
    D = all(carrier_qualified(by_phase[("R0M1", phase)]) for phase in ("FIT", "HOLDOUT"))
    E = bool(all(additivity[phase] <= BARS["additivity"] for phase in ("FIT", "HOLDOUT"))
        and all(row["temporal_command_gold_collateral"] <= BARS["collateral"]
                for row in reports))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid" if not (A and B) else
                "redundant_dual_carriers" if C and D else
                "identity_skip_dominant" if C else
                "module_response_dominant" if D else
                "interaction_dominant")
    result = {
        "schema": "temporal_iswas_selected_writer_residual_skip_module_response_factorial_result_v1",
        "candidate_id": prior["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": observed, "selected_writer_heads": list(selected_heads),
        "reports": reports, "additivity_relative_l2": additivity,
        "writer_replay_max_abs_error": max(replay_errors),
        "capture_shapes": {"block10_residual": list(native_state.shape),
            "modules": {name: list(value.shape) for name, value in native_modules.items()}},
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"reports": reports, "additivity_relative_l2": additivity,
        "writer_replay_max_abs_error": max(replay_errors), "predictions": predictions,
        "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
