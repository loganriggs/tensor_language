#!/usr/bin/env python3
"""Validate fixed P7 response plus residual identity as an OOD two-stream circuit."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_capture_full_arm_replay_finiteness_and_exact_price pred_b_original_selected_p7_response_transfers_ood_without_reselection pred_c_p7_response_composes_with_identity_to_replay_the_writer_ood pred_d_both_physical_streams_are_needed_in_the_composed_interface pred_e_two_stream_composition_is_nearly_additive_and_temporally_selective
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
import run_temporal_iswas_downstream_module_singleton_ordered_greedy_v1 as cumulative
import run_temporal_iswas_selected_writer_residual_skip_module_response_factorial_v1 as factorial


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_p7_residual_two_stream_ood_composition_v1.json"
CUMULATIVE_RESULT = ROOT / "circuits/followups/temporal_iswas_downstream_module_singleton_ordered_greedy_v1_result.json"
DIRECT_V2_RESULT = ROOT / "circuits/followups/temporal_iswas_selected_writer_block10_direct_residual_add_remove_v2_scoped_logit_audit_result.json"
FACTORIAL_RESULT = ROOT / "circuits/followups/temporal_iswas_selected_writer_residual_skip_module_response_factorial_v1_result.json"
FACTORIAL_RUNNER = ROOT / "ops/run_temporal_iswas_selected_writer_residual_skip_module_response_factorial_v1.py"
CUMULATIVE_RUNNER = ROOT / "ops/run_temporal_iswas_downstream_module_singleton_ordered_greedy_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_p7_residual_two_stream_ood_composition_v1_result.json"
EXPECTED = {
    "prior": "d3ac8e980cce0be1b194ea6c7293c6f3e00f494b39d273113c1049f899ae6448",
    "cumulative_result": "ca53dd95504e174b3301ce9e4cf87edcf112ec9685a6f5b82643057224338d69",
    "direct_v2_result": "7778b2f3db00ff98f445202dc8dcb7bf8cb52aff631f6f609ad41a47ec1b7e67",
    "factorial_result": "8e67ec87e123cd7c452e8e28b34dd5f8d9b310e4f79ffe69b6382f680c33607b",
    "factorial_runner": "9e3c9773bb96330f2b2166f5fe2c1087d4ec10b8199f7f4219c527e48ef89c0e",
    "cumulative_runner": "13734de17a6e0c7fa47be1fe3d45804e94e59d8743562e2075e2f0aa8a420d39",
}
FILES = {"prior": PRIOR, "cumulative_result": CUMULATIVE_RESULT,
         "direct_v2_result": DIRECT_V2_RESULT, "factorial_result": FACTORIAL_RESULT,
         "factorial_runner": FACTORIAL_RUNNER, "cumulative_runner": CUMULATIVE_RUNNER}
P7 = ("A11", "M11", "M12", "M15", "M13", "M16", "M10")
ARMS = ("writer", "identity_full", "response_full", "P7_response", "P7_plus_identity")
PRICE = {"checkpoint_loads": 1, "model_forwards": 6,
         "sequence_evaluations": 768, "scored_token_positions": 1536,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_pairing_capture_full_arm_replay_finiteness_and_exact_price",
    "pred_b_original_selected_p7_response_transfers_ood_without_reselection",
    "pred_c_p7_response_composes_with_identity_to_replay_the_writer_ood",
    "pred_d_both_physical_streams_are_needed_in_the_composed_interface",
    "pred_e_two_stream_composition_is_nearly_additive_and_temporally_selective",
)
BARS = {"replay": 1e-5, "p7_recovery_min": .4, "p7_recovery_max": 1.75,
        "p7_cosine": .85, "p7_residual": .75, "p7_direction": .8,
        "joint_recovery_min": .9, "joint_recovery_max": 1.1,
        "joint_cosine": .98, "joint_residual": .2, "joint_direction": .9,
        "partial_min": .2, "partial_max": .8, "contribution_min": .2,
        "additivity": .15, "collateral": .01}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def run_p7_arm(backend, tokens, entry, module_replacements, position_rows):
    entry_calls = 0
    module_calls = {name: 0 for name in P7}
    handles = []

    def entry_hook(_module, arguments):
        nonlocal entry_calls
        entry_calls += 1
        if len(arguments) != 3:
            raise RuntimeError("block-10 input signature changed")
        changed = factorial.atlas.replace_positions(arguments[0], entry, position_rows)
        return (changed,) + tuple(arguments[1:])

    def module_hook(name):
        def replace(_module, _inputs, output):
            module_calls[name] += 1
            return factorial.atlas.replace_positions(
                output, module_replacements[name], position_rows)
        return replace

    handles.append(backend.model.transformer.h[10].register_forward_pre_hook(entry_hook))
    targets = factorial.atlas.module_targets(backend.model)
    for name in P7:
        handles.append(targets[name].register_forward_hook(module_hook(name)))
    try:
        logits, _captures = factorial.atlas.mediation.parent._forward(backend, tokens)
    finally:
        for handle in handles:
            handle.remove()
    if entry_calls != 1 or set(module_calls.values()) != {1}:
        raise RuntimeError("P7 state/module hook coverage changed")
    return logits, entry_calls, module_calls


def metric_rows(effects, other_effects, writer_effect, other_gold, endpoints):
    rows = []
    for phase in ("FIT", "HOLDOUT"):
        chosen = np.asarray([row["phase"] == phase for row, _cell, _endpoint in endpoints])
        temporal_norm = max(float(np.linalg.norm(other_gold[chosen])), 1e-30)
        for arm in ARMS:
            metric = factorial.atlas.mediation.accounting.effect_metrics(
                effects[arm][chosen], writer_effect[chosen], other_effects[arm][chosen])
            rows.append({"phase": phase, "arm": arm, **metric,
                "temporal_command_gold_collateral":
                    float(np.linalg.norm(other_effects[arm][chosen]) / temporal_norm)})
        for stream, contribution in (
                ("identity", effects["P7_plus_identity"] - effects["P7_response"]),
                ("P7_response", effects["P7_plus_identity"] - effects["identity_full"])):
            metric = factorial.atlas.mediation.accounting.effect_metrics(
                contribution[chosen], writer_effect[chosen],
                np.zeros_like(contribution[chosen]))
            rows.append({"phase": phase, "arm": f"{stream}_joint_contribution", **metric})
    return rows


def replay_error(reports, direct_v2):
    old = {(row["phase"], row["arm"]): row for row in direct_v2["panels"]["ood"]["reports"]}
    mapped = {"writer": "writer", "identity_full": "R1M0_dynamic",
              "response_full": "R0M1_dynamic"}
    errors = []
    for row in reports:
        if row["arm"] not in mapped:
            continue
        reference = old[(row["phase"], mapped[row["arm"]])]
        for field in ("signed_recovery", "cosine", "relative_residual",
                      "direction_agreement", "non_target_to_target_gold_norm",
                      "temporal_command_gold_collateral"):
            errors.append(abs(row[field] - reference[field]))
    return max(errors)


def qualified(row, kind):
    if kind == "p7":
        return bool(BARS["p7_recovery_min"] <= row["signed_recovery"]
            <= BARS["p7_recovery_max"] and row["cosine"] >= BARS["p7_cosine"]
            and row["relative_residual"] <= BARS["p7_residual"]
            and row["direction_agreement"] >= BARS["p7_direction"])
    return bool(BARS["joint_recovery_min"] <= row["signed_recovery"]
        <= BARS["joint_recovery_max"] and row["cosine"] >= BARS["joint_cosine"]
        and row["relative_residual"] <= BARS["joint_residual"]
        and row["direction_agreement"] >= BARS["joint_direction"])


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior = json.loads(PRIOR.read_text())
    cumulative_result = json.loads(CUMULATIVE_RESULT.read_text())
    direct_v2 = json.loads(DIRECT_V2_RESULT.read_text())
    factorial_result = json.loads(FACTORIAL_RESULT.read_text())
    authority_ok = bool(observed == EXPECTED and prior["price"] == PRICE
        and tuple(cumulative_result["selected_prefix"]["modules"]) == P7
        and cumulative_result["selected_arm"] == "P7"
        and cumulative_result["terminal"] == "compact_distributed_module_prefix"
        and direct_v2["terminal"] == "ood_instability"
        and tuple(direct_v2["predictions"].values()) == (True, True, True, False, False)
        and factorial_result["terminal"] == "identity_skip_dominant")
    dry = {"candidate_id": prior["candidate_id"], "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "authority_ok": authority_ok, "population": "ood", "P7": list(P7),
           "arms": list(ARMS), "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if not authority_ok:
        raise RuntimeError("P7 residual OOD authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    loc, parent = factorial.atlas.mediation.parent.loc, factorial.atlas.mediation.parent
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
    source_rows = [item[parent.LOCKED[role]] for item in regions]
    position_rows = [list(range(query + 1)) for query in queries]
    selected_heads = tuple(json.loads(factorial.GREEDY_RESULT.read_text())[
        "selected_prefixes"][role]["heads"])

    with torch.no_grad():
        (native_logits, head_captures, _native_l11, native_entry, native_modules,
         native_entry_calls, native_module_calls) = factorial.capture_state_and_modules(
            backend, tokens)
        (writer_logits, _unused, _writer_l11, writer_entry, writer_modules,
         writer_entry_calls, writer_module_calls) = factorial.capture_state_and_modules(
            backend, tokens, head_captures=head_captures, selected_labels=selected_heads,
            pairs=pairs, position_rows=source_rows)
        identity_logits, identity_entry_calls, identity_module_calls = factorial.run_factorial_arm(
            backend, tokens, writer_entry, native_modules, position_rows)
        response_logits, response_entry_calls, response_module_calls = factorial.run_factorial_arm(
            backend, tokens, native_entry, writer_modules, position_rows)
        p7_logits, p7_entry_calls, p7_module_calls = run_p7_arm(
            backend, tokens, native_entry, writer_modules, position_rows)
        joint_logits, joint_entry_calls, joint_module_calls = run_p7_arm(
            backend, tokens, writer_entry, writer_modules, position_rows)

    logits = {"writer": writer_logits, "identity_full": identity_logits,
              "response_full": response_logits, "P7_response": p7_logits,
              "P7_plus_identity": joint_logits}
    native = {name: loc.margins(native_logits, endpoints, name) for name in parent.LOCKED}
    effects = {name: loc.margins(value, endpoints, role) - native[role]
               for name, value in logits.items()}
    other_effects = {name: loc.margins(value, endpoints, other) - native[other]
                     for name, value in logits.items()}
    writer_effect = effects["writer"]
    other_gold = native[other][other_pairs] - native[other]
    reports = metric_rows(effects, other_effects, writer_effect, other_gold, endpoints)
    replay = replay_error(reports, direct_v2)
    by = {(row["phase"], row["arm"]): row for row in reports}
    additivity = {}
    for phase in ("FIT", "HOLDOUT"):
        chosen = np.asarray([row["phase"] == phase for row, _cell, _endpoint in endpoints])
        additivity[phase] = float(np.linalg.norm(effects["P7_plus_identity"][chosen]
            - effects["identity_full"][chosen] - effects["P7_response"][chosen])
            / max(float(np.linalg.norm(writer_effect[chosen])), 1e-30))

    A = bool(authority_ok and len(endpoints) * 6 == PRICE["sequence_evaluations"]
        and replay <= BARS["replay"] and native_entry_calls == writer_entry_calls == 1
        and set(native_module_calls.values()) == {1}
        and set(writer_module_calls.values()) == {1}
        and identity_entry_calls == response_entry_calls == p7_entry_calls == joint_entry_calls == 1
        and set(identity_module_calls.values()) == {1}
        and set(response_module_calls.values()) == {1}
        and set(p7_module_calls.values()) == set(joint_module_calls.values()) == {1}
        and finite({"reports": reports, "additivity": additivity}))
    B = all(qualified(by[(phase, "P7_response")], "p7") for phase in ("FIT", "HOLDOUT"))
    C = all(qualified(by[(phase, "P7_plus_identity")], "joint")
            for phase in ("FIT", "HOLDOUT"))
    D = all(BARS["partial_min"] <= by[(phase, arm)]["signed_recovery"] <= BARS["partial_max"]
            for phase in ("FIT", "HOLDOUT") for arm in ("identity_full", "P7_response"))
    D = bool(D and all(by[(phase, f"{stream}_joint_contribution")]["signed_recovery"]
        >= BARS["contribution_min"] for phase in ("FIT", "HOLDOUT")
        for stream in ("identity", "P7_response")))
    E = bool(all(value <= BARS["additivity"] for value in additivity.values())
        and all(row.get("temporal_command_gold_collateral", 0.0) <= BARS["collateral"]
                for row in reports))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid" if not A else "p7_ood_null" if not B else
        "noncomposable" if not C else "redundant_single_stream" if not D else
        "nonadditive_or_nonselective" if not E else "stable_two_stream_interface")
    result = {"schema": "temporal_iswas_p7_residual_two_stream_ood_composition_result_v1",
        "candidate_id": prior["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "selected_writer_heads": list(selected_heads), "P7": list(P7),
        "reports": reports, "two_stream_additivity_relative_l2": additivity,
        "parent_replay_max_abs_error": replay, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({"reports": reports, "two_stream_additivity_relative_l2": additivity,
        "parent_replay_max_abs_error": replay, "predictions": predictions,
        "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
