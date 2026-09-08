#!/usr/bin/env python3
"""Measure fixed-P7 module necessity inside the exact OOD identity background."""

# BQGATE: EXPERIMENT pred_a_authority_capture_full_replay_finiteness_and_exact_price pred_b_at_least_two_p7_modules_are_stably_necessary pred_c_attention11_is_stably_necessary_and_licenses_head_splitting pred_d_at_least_one_p7_mlp_is_stably_necessary_and_licenses_weight_factor_splitting pred_e_every_leave_one_out_arm_is_temporally_selective
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
import run_temporal_iswas_p7_residual_two_stream_ood_composition_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_iswas_p7_residual_two_stream_ood_composition_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_p7_residual_two_stream_ood_composition_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1_result.json"
EXPECTED = {
    "prior": "282a7e1d48cf2d3c99d73064cdb8e30c69e7fd0952989192d903e939fd741f7e",
    "parent_result": "55498ec9110fbddc8ae9a86eb4bab63efeb92540322aaa8783cec2d8c7bd55a7",
    "parent_runner": "0e3f6de1b273d372006652e37c085ddf15d5b120795225ba873de82a402a19ec",
}
FILES = {"prior": PRIOR, "parent_result": PARENT_RESULT, "parent_runner": PARENT_RUNNER}
P7 = parent.P7
ARMS = ("full",) + tuple(f"without_{name}" for name in P7)
PRICE = {"checkpoint_loads": 1, "model_forwards": 10,
         "sequence_evaluations": 1280, "scored_token_positions": 2560,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
BARS = {"replay": 1e-5, "necessity": .03, "full_recovery_min": .9,
        "full_recovery_max": 1.1, "full_cosine": .98,
        "full_residual": .2, "full_direction": .9, "collateral": .01}
PREDICTION_KEYS = (
    "pred_a_authority_capture_full_replay_finiteness_and_exact_price",
    "pred_b_at_least_two_p7_modules_are_stably_necessary",
    "pred_c_attention11_is_stably_necessary_and_licenses_head_splitting",
    "pred_d_at_least_one_p7_mlp_is_stably_necessary_and_licenses_weight_factor_splitting",
    "pred_e_every_leave_one_out_arm_is_temporally_selective",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def run_subset_arm(backend, tokens, entry, module_replacements, position_rows, subset):
    subset = tuple(subset)
    if not set(subset).issubset(P7) or len(subset) != len(set(subset)):
        raise ValueError("subset must contain unique fixed-P7 members")
    entry_calls = 0
    module_calls = {name: 0 for name in subset}
    handles = []

    def entry_hook(_module, arguments):
        nonlocal entry_calls
        entry_calls += 1
        if len(arguments) != 3:
            raise RuntimeError("block-10 input signature changed")
        changed = parent.factorial.atlas.replace_positions(arguments[0], entry, position_rows)
        return (changed,) + tuple(arguments[1:])

    def module_hook(name):
        def replace(_module, _inputs, output):
            module_calls[name] += 1
            return parent.factorial.atlas.replace_positions(
                output, module_replacements[name], position_rows)
        return replace

    handles.append(backend.model.transformer.h[10].register_forward_pre_hook(entry_hook))
    targets = parent.factorial.atlas.module_targets(backend.model)
    for name in subset:
        handles.append(targets[name].register_forward_hook(module_hook(name)))
    try:
        logits, _captures = parent.factorial.atlas.mediation.parent._forward(backend, tokens)
    finally:
        for handle in handles:
            handle.remove()
    if entry_calls != 1 or (module_calls and set(module_calls.values()) != {1}):
        raise RuntimeError("P7 subset state/module hook coverage changed")
    return logits, entry_calls, module_calls


def full_qualified(row):
    return bool(BARS["full_recovery_min"] <= row["signed_recovery"]
        <= BARS["full_recovery_max"] and row["cosine"] >= BARS["full_cosine"]
        and row["relative_residual"] <= BARS["full_residual"]
        and row["direction_agreement"] >= BARS["full_direction"])


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior = json.loads(PRIOR.read_text())
    parent_result = json.loads(PARENT_RESULT.read_text())
    authority_ok = bool(observed == EXPECTED and prior["price"] == PRICE
        and parent_result["terminal"] == "stable_two_stream_interface"
        and tuple(parent_result["P7"]) == P7
        and tuple(parent_result["predictions"].values()) == (True,) * 5)
    dry = {"candidate_id": prior["candidate_id"], "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "authority_ok": authority_ok, "population": "ood", "P7": list(P7),
           "arms": list(ARMS), "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if not authority_ok:
        raise RuntimeError("P7 LOO authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    loc = parent.factorial.atlas.mediation.parent.loc
    model_parent = parent.factorial.atlas.mediation.parent
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
    selected_heads = tuple(json.loads(parent.factorial.GREEDY_RESULT.read_text())[
        "selected_prefixes"][role]["heads"])

    with torch.no_grad():
        (native_logits, head_captures, _native_l11, _native_entry, _native_modules,
         native_entry_calls, native_module_calls) = parent.factorial.capture_state_and_modules(
            backend, tokens)
        (_writer_logits, _unused, _writer_l11, writer_entry, writer_modules,
         writer_entry_calls, writer_module_calls) = parent.factorial.capture_state_and_modules(
            backend, tokens, head_captures=head_captures, selected_labels=selected_heads,
            pairs=pairs, position_rows=source_rows)
        logits = {}
        arm_calls = {}
        logits["full"], entry_calls, module_calls = run_subset_arm(
            backend, tokens, writer_entry, writer_modules, position_rows, P7)
        arm_calls["full"] = {"entry": entry_calls, "modules": module_calls}
        for omitted in P7:
            arm = f"without_{omitted}"
            subset = tuple(name for name in P7 if name != omitted)
            logits[arm], entry_calls, module_calls = run_subset_arm(
                backend, tokens, writer_entry, writer_modules, position_rows, subset)
            arm_calls[arm] = {"entry": entry_calls, "modules": module_calls}

    native = {name: loc.margins(native_logits, endpoints, name) for name in model_parent.LOCKED}
    effects = {name: loc.margins(value, endpoints, role) - native[role]
               for name, value in logits.items()}
    other_effects = {name: loc.margins(value, endpoints, other) - native[other]
                     for name, value in logits.items()}
    parent_writer = next(row for row in parent_result["reports"]
                         if row["phase"] == "FIT" and row["arm"] == "writer")
    del parent_writer
    # The writer effect is the parent full arm because that arm replayed writer exactly.
    writer_effect = effects["full"]
    other_gold = native[other][other_pairs] - native[other]
    reports = []
    for phase in ("FIT", "HOLDOUT"):
        chosen = np.asarray([row["phase"] == phase for row, _cell, _endpoint in endpoints])
        temporal_norm = max(float(np.linalg.norm(other_gold[chosen])), 1e-30)
        for arm in ARMS:
            metric = parent.factorial.atlas.mediation.accounting.effect_metrics(
                effects[arm][chosen], writer_effect[chosen], other_effects[arm][chosen])
            reports.append({"phase": phase, "arm": arm, **metric,
                "temporal_command_gold_collateral":
                    float(np.linalg.norm(other_effects[arm][chosen]) / temporal_norm)})
    by = {(row["phase"], row["arm"]): row for row in reports}
    parent_by = {(row["phase"], row["arm"]): row for row in parent_result["reports"]}
    replay_errors = []
    for phase in ("FIT", "HOLDOUT"):
        for field in ("signed_recovery", "cosine", "relative_residual",
                      "direction_agreement", "non_target_to_target_gold_norm",
                      "temporal_command_gold_collateral"):
            replay_errors.append(abs(by[(phase, "full")][field]
                - parent_by[(phase, "P7_plus_identity")][field]))
    replay = max(replay_errors)
    necessity = {name: {phase: by[(phase, "full")]["signed_recovery"]
        - by[(phase, f"without_{name}")]["signed_recovery"]
        for phase in ("FIT", "HOLDOUT")} for name in P7}
    stable = [name for name in P7 if all(necessity[name][phase] >= BARS["necessity"]
                                        for phase in ("FIT", "HOLDOUT"))]

    calls_ok = bool(native_entry_calls == writer_entry_calls == 1
        and set(native_module_calls.values()) == {1}
        and set(writer_module_calls.values()) == {1}
        and all(record["entry"] == 1 and set(record["modules"].values()) == {1}
                for record in arm_calls.values()))
    A = bool(authority_ok and len(endpoints) * 10 == PRICE["sequence_evaluations"]
        and replay <= BARS["replay"] and calls_ok
        and all(full_qualified(by[(phase, "full")]) for phase in ("FIT", "HOLDOUT"))
        and finite({"reports": reports, "necessity": necessity}))
    B = len(stable) >= 2
    C = "A11" in stable
    D = any(name.startswith("M") for name in stable)
    E = all(row["temporal_command_gold_collateral"] <= BARS["collateral"]
            for row in reports)
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A else
        "nonselective" if not E else
        "nonminimal_sufficient_bundle" if not B else
        "mlp_only_necessity" if not C and D else
        "attention_only_necessity" if C and not D else
        "stable_cross_module_necessity")
    result = {"schema": "temporal_iswas_p7_identity_background_module_leave_one_out_ood_result_v1",
        "candidate_id": prior["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "selected_writer_heads": list(selected_heads), "P7": list(P7),
        "reports": reports, "necessity_drop": necessity,
        "stable_necessary_modules": stable, "parent_replay_max_abs_error": replay,
        "hook_calls": arm_calls, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({"necessity_drop": necessity, "stable_necessary_modules": stable,
        "parent_replay_max_abs_error": replay, "predictions": predictions,
        "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
