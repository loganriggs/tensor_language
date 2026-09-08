#!/usr/bin/env python3
"""V2: trace block-10 identity through live P7 modules with corrected writer replay."""

# BQGATE: EXPERIMENT pred_a_authority_capture_all_subset_noop_full_writer_replay_finiteness_and_exact_price pred_b_exact_mobius_reconstruction_and_shapley_efficiency pred_c_at_least_one_p7_module_has_stable_live_route_credit pred_d_at_least_one_p7_pair_has_stable_interaction pred_e_every_source_conditioned_subset_is_temporally_selective
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
import run_temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1 as necessity


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_identity_p7_live_module_effect_game_ood_v2.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_iswas_p7_residual_two_stream_ood_composition_v1_result.json"
NECESSITY_RESULT = ROOT / "circuits/followups/temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1_result.json"
NECESSITY_RUNNER = ROOT / "ops/run_temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_identity_p7_live_module_effect_game_ood_v2_result.json"
EXPECTED = {
    "prior": "7088f860028616928d775fe4de55019896f5e68c529540fcaccd1745d28b6fa6",
    "parent_result": "55498ec9110fbddc8ae9a86eb4bab63efeb92540322aaa8783cec2d8c7bd55a7",
    "necessity_result": "20eff67e079f46beefe0cf6ec14f16af7674c359fb460c8d73357175761154e4",
    "necessity_runner": "384a79a38294416204d5bde0e24fcee7008e391a6712de82ec59c627f7f1318a",
}
FILES = {"prior": PRIOR, "parent_result": PARENT_RESULT,
         "necessity_result": NECESSITY_RESULT, "necessity_runner": NECESSITY_RUNNER}
P7 = necessity.P7
N_MASKS = 1 << len(P7)
PRICE = {"checkpoint_loads": 1, "model_forwards": 258,
         "sequence_evaluations": 33024, "scored_token_positions": 66048,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
BARS = {"replay": 1e-5, "closure": 1e-8, "credit": .03, "collateral": .01}
PREDICTION_KEYS = (
    "pred_a_authority_capture_all_subset_noop_full_writer_replay_finiteness_and_exact_price",
    "pred_b_exact_mobius_reconstruction_and_shapley_efficiency",
    "pred_c_at_least_one_p7_module_has_stable_live_route_credit",
    "pred_d_at_least_one_p7_pair_has_stable_interaction",
    "pred_e_every_source_conditioned_subset_is_temporally_selective",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def members(mask):
    if not isinstance(mask, int) or mask < 0 or mask >= N_MASKS:
        raise ValueError("mask outside fixed P7 lattice")
    return tuple(name for bit, name in enumerate(P7) if mask & (1 << bit))


def mobius(values):
    """Vector-valued subset-lattice Mobius transform; axis zero is the bit mask."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != N_MASKS or not np.isfinite(values).all():
        raise ValueError("values must be finite with one row per fixed-P7 mask")
    coefficients = values.copy()
    for bit in range(len(P7)):
        flag = 1 << bit
        for mask in range(N_MASKS):
            if mask & flag:
                coefficients[mask] -= coefficients[mask ^ flag]
    return coefficients


def reconstruct(coefficients):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim < 1 or coefficients.shape[0] != N_MASKS:
        raise ValueError("coefficients must have one row per fixed-P7 mask")
    values = coefficients.copy()
    for bit in range(len(P7)):
        flag = 1 << bit
        for mask in range(N_MASKS):
            if mask & flag:
                values[mask] += values[mask ^ flag]
    return values


def shapley(coefficients):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    output = np.zeros((len(P7),) + coefficients.shape[1:], dtype=np.float64)
    for mask in range(1, N_MASKS):
        order = mask.bit_count()
        for bit in range(len(P7)):
            if mask & (1 << bit):
                output[bit] += coefficients[mask] / order
    return output


def pair_interactions(coefficients):
    """Grabisch-Roubens pair index, including all higher-order dividends."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    output = {}
    for left in range(len(P7)):
        for right in range(left + 1, len(P7)):
            value = np.zeros(coefficients.shape[1:], dtype=np.float64)
            flags = (1 << left) | (1 << right)
            for mask in range(N_MASKS):
                if mask & flags == flags:
                    value += coefficients[mask] / (mask.bit_count() - 1)
            output[(P7[left], P7[right])] = value
    return output


def signed_credit(vector, reference):
    vector, reference = np.asarray(vector), np.asarray(reference)
    denominator = max(float(np.dot(reference, reference)), 1e-30)
    return float(np.dot(vector, reference) / denominator)


def run_game_arm(backend, tokens, source_entry, native_modules, position_rows, enabled):
    enabled = tuple(enabled)
    if len(enabled) != len(set(enabled)) or not set(enabled).issubset(P7):
        raise ValueError("enabled modules must be a unique fixed-P7 subset")
    disabled = tuple(name for name in P7 if name not in enabled)
    entry_calls = 0
    module_calls = {name: 0 for name in disabled}
    handles = []

    def entry_hook(_module, arguments):
        nonlocal entry_calls
        entry_calls += 1
        if len(arguments) != 3:
            raise RuntimeError("block-10 input signature changed")
        changed = necessity.parent.factorial.atlas.replace_positions(
            arguments[0], source_entry, position_rows)
        return (changed,) + tuple(arguments[1:])

    def clamp(name):
        def hook(_module, _inputs, output):
            module_calls[name] += 1
            return necessity.parent.factorial.atlas.replace_positions(
                output, native_modules[name], position_rows)
        return hook

    handles.append(backend.model.transformer.h[10].register_forward_pre_hook(entry_hook))
    targets = necessity.parent.factorial.atlas.module_targets(backend.model)
    for name in disabled:
        handles.append(targets[name].register_forward_hook(clamp(name)))
    try:
        logits, _captures = necessity.parent.factorial.atlas.mediation.parent._forward(
            backend, tokens)
    finally:
        for handle in handles:
            handle.remove()
    if entry_calls != 1 or (module_calls and set(module_calls.values()) != {1}):
        raise RuntimeError("effect-game hook coverage changed")
    return logits, {"entry": entry_calls, "modules": module_calls}


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior = json.loads(PRIOR.read_text())
    parent_result = json.loads(PARENT_RESULT.read_text())
    necessity_result = json.loads(NECESSITY_RESULT.read_text())
    authority_ok = bool(observed == EXPECTED and prior["price"] == PRICE
        and parent_result["terminal"] == "stable_two_stream_interface"
        and tuple(parent_result["P7"]) == P7
        and necessity_result["terminal"] == "nonminimal_sufficient_bundle"
        and necessity_result["stable_necessary_modules"] == [])
    dry = {"candidate_id": prior["candidate_id"], "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "authority_ok": authority_ok, "population": "ood", "P7": list(P7),
           "n_masks": N_MASKS, "source_states": ["absent", "present"], "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if not authority_ok:
        raise RuntimeError("P7 live-module effect-game authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    loc = necessity.parent.factorial.atlas.mediation.parent.loc
    model_parent = necessity.parent.factorial.atlas.mediation.parent
    backend = loc.producer.Bilin18TorchBackend.load("cuda")
    rows = loc.ood.build_rows()
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
    source_rows = [item[model_parent.LOCKED[role]] for item in regions]
    position_rows = [list(range(query + 1)) for query in queries]
    selected_heads = tuple(json.loads(necessity.parent.factorial.GREEDY_RESULT.read_text())[
        "selected_prefixes"][role]["heads"])

    with torch.no_grad():
        (native_logits, head_captures, _native_l11, native_entry, native_modules,
         native_entry_calls, native_module_calls) = necessity.parent.factorial.capture_state_and_modules(
            backend, tokens)
        (writer_logits, _unused, _writer_l11, writer_entry, _writer_modules,
         writer_entry_calls, writer_module_calls) = necessity.parent.factorial.capture_state_and_modules(
            backend, tokens, head_captures=head_captures, selected_labels=selected_heads,
            pairs=pairs, position_rows=source_rows)
        native_margin = {name: loc.margins(native_logits, endpoints, name)
                         for name in model_parent.LOCKED}
        game = {False: {role: [], other: []}, True: {role: [], other: []}}
        calls = {False: [], True: []}
        for source_present, source_entry in ((False, native_entry), (True, writer_entry)):
            for mask in range(N_MASKS):
                logits, arm_calls = run_game_arm(
                    backend, tokens, source_entry, native_modules, position_rows, members(mask))
                calls[source_present].append(arm_calls)
                for name in (role, other):
                    game[source_present][name].append(loc.margins(logits, endpoints, name))

    for source_present in (False, True):
        for name in (role, other):
            game[source_present][name] = np.asarray(game[source_present][name])
    noop_error = max(float(np.max(np.abs(game[False][name] - native_margin[name][None, :])))
                       for name in (role, other))
    effects = game[True][role] - game[False][role]
    other_effects = game[True][other] - game[False][other]
    writer_effect = loc.margins(writer_logits, endpoints, role) - native_margin[role]
    coefficients = mobius(effects)
    reconstructed = reconstruct(coefficients)
    phi = shapley(coefficients)
    pair_values = pair_interactions(coefficients)
    closure_error = float(np.max(np.abs(reconstructed - effects)))
    efficiency_error = float(np.max(np.abs(phi.sum(axis=0)
        - (effects[N_MASKS - 1] - effects[0]))))
    parent_by = {(row["phase"], row["arm"]): row for row in parent_result["reports"]}
    reports, module_credits, pair_credits, coalition_recovery = [], {}, {}, {}
    other_gold = native_margin[other][other_pairs] - native_margin[other]
    max_collateral = 0.0
    replay_errors = []
    for phase in ("FIT", "HOLDOUT"):
        chosen = np.asarray([row["phase"] == phase for row, _cell, _endpoint in endpoints])
        reference = effects[N_MASKS - 1][chosen]
        direct = effects[0][chosen]
        module_credits[phase] = {
            name: signed_credit(phi[bit][chosen], reference)
            for bit, name in enumerate(P7)}
        pair_credits[phase] = {
            f"{left}+{right}": signed_credit(value[chosen], reference)
            for (left, right), value in pair_values.items()}
        coalition_recovery[phase] = [signed_credit(effects[mask][chosen], reference)
                                     for mask in range(N_MASKS)]
        temporal_norm = max(float(np.linalg.norm(other_gold[chosen])), 1e-30)
        collateral = [float(np.linalg.norm(other_effects[mask][chosen]) / temporal_norm)
                      for mask in range(N_MASKS)]
        max_collateral = max(max_collateral, max(collateral))
        full_metric = necessity.parent.factorial.atlas.mediation.accounting.effect_metrics(
            reference, writer_effect[chosen], other_effects[N_MASKS - 1][chosen])
        direct_metric = necessity.parent.factorial.atlas.mediation.accounting.effect_metrics(
            direct, writer_effect[chosen], other_effects[0][chosen])
        reports.extend((
            {"phase": phase, "arm": "all_live", **full_metric,
             "temporal_command_gold_collateral": collateral[N_MASKS - 1]},
            {"phase": phase, "arm": "all_p7_native_clamped", **direct_metric,
             "temporal_command_gold_collateral": collateral[0]},
        ))
        parent_row = parent_by[(phase, "writer")]
        for field in ("signed_recovery", "cosine", "relative_residual",
                      "direction_agreement", "non_target_to_target_gold_norm"):
            replay_errors.append(abs(full_metric[field] - parent_row[field]))
    replay_error = max(replay_errors)

    stable_modules = [name for name in P7
        if abs(module_credits["FIT"][name]) >= BARS["credit"]
        and abs(module_credits["HOLDOUT"][name]) >= BARS["credit"]
        and module_credits["FIT"][name] * module_credits["HOLDOUT"][name] > 0]
    stable_pairs = [name for name in pair_credits["FIT"]
        if abs(pair_credits["FIT"][name]) >= BARS["credit"]
        and abs(pair_credits["HOLDOUT"][name]) >= BARS["credit"]
        and pair_credits["FIT"][name] * pair_credits["HOLDOUT"][name] > 0]
    calls_ok = bool(native_entry_calls == writer_entry_calls == 1
        and set(native_module_calls.values()) == {1}
        and set(writer_module_calls.values()) == {1}
        and all(record["entry"] == 1
                and (not record["modules"] or set(record["modules"].values()) == {1})
                for side in calls.values() for record in side))
    A = bool(authority_ok and calls_ok and len(endpoints) * PRICE["model_forwards"]
        == PRICE["sequence_evaluations"] and noop_error <= BARS["replay"]
        and replay_error <= BARS["replay"] and finite({"reports": reports,
            "module_credits": module_credits, "pair_credits": pair_credits,
            "coalition_recovery": coalition_recovery}))
    B = closure_error <= BARS["closure"] and efficiency_error <= BARS["closure"]
    C, D = bool(stable_modules), bool(stable_pairs)
    E = max_collateral <= BARS["collateral"]
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A or not B else
        "nonselective" if not E else "direct_background_identity" if not C else
        "additive_live_module_routes" if not D else
        "redundant_or_complementary_live_routes")
    result = {"schema": "temporal_iswas_identity_p7_live_module_effect_game_ood_result_v2",
        "candidate_id": prior["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "selected_writer_heads": list(selected_heads), "P7": list(P7),
        "reports": reports, "module_shapley_signed_credit": module_credits,
        "pair_interaction_signed_credit": pair_credits,
        "stable_route_modules": stable_modules, "stable_interaction_pairs": stable_pairs,
        "coalition_signed_recovery_by_mask": coalition_recovery,
        "noop_max_abs_margin_error": noop_error,
        "full_writer_replay_max_abs_error": replay_error,
        "mobius_reconstruction_max_abs_error": closure_error,
        "shapley_efficiency_max_abs_error": efficiency_error,
        "maximum_temporal_command_gold_collateral": max_collateral,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({"module_shapley_signed_credit": module_credits,
        "stable_route_modules": stable_modules, "stable_interaction_pairs": stable_pairs,
        "noop_max_abs_margin_error": noop_error,
        "full_writer_replay_max_abs_error": replay_error,
        "mobius_reconstruction_max_abs_error": closure_error,
        "shapley_efficiency_max_abs_error": efficiency_error,
        "maximum_temporal_command_gold_collateral": max_collateral,
        "predictions": predictions, "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
