#!/usr/bin/env python3
"""Conditionally factor necessary M11 into exact activation-conditioned products."""

# BQGATE: EXPERIMENT pred_a_conditional_authority_capture_factor_closure_hook_coverage_finiteness_and_exact_price pred_b_all_three_factors_replay_complete_m11_and_parent_joint_program pred_c_bilinear_interaction_is_stably_necessary pred_d_at_least_one_linear_factor_is_stably_material pred_e_all_factor_arms_preserve_temporal_selectivity
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_p7_identity_m11_exact_product_factorial_ood_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_iswas_p7_residual_two_stream_ood_composition_v1_result.json"
NECESSITY_RUNNER = ROOT / "ops/run_temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1.py"
NECESSITY_RESULT = ROOT / "circuits/followups/temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1_result.json"
BINDING = ROOT / "circuits/bindings/temporal_iswas_p7_identity_m11_exact_product_factorial_ood_v1.json"
OUT = ROOT / "circuits/followups/temporal_iswas_p7_identity_m11_exact_product_factorial_ood_v1_result.json"
EXPECTED = {
    "prior": "1aac98581f8ae575cfad2ca7d7a3daf7de536b2c5aef090566b22b7fd65d75d0",
    "parent_result": "55498ec9110fbddc8ae9a86eb4bab63efeb92540322aaa8783cec2d8c7bd55a7",
    "necessity_runner": "384a79a38294416204d5bde0e24fcee7008e391a6712de82ec59c627f7f1318a",
}
FILES = {"prior": PRIOR, "parent_result": PARENT_RESULT,
         "necessity_runner": NECESSITY_RUNNER}
FACTORS = ("left", "right", "interaction")
ARMS = ("none", "complete_M11", "left", "right", "interaction",
        "left_right", "left_interaction", "right_interaction", "all_three")
PRICE = {"checkpoint_loads": 1, "model_forwards": 11,
         "sequence_evaluations": 1408, "scored_token_positions": 2816,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_conditional_authority_capture_factor_closure_hook_coverage_finiteness_and_exact_price",
    "pred_b_all_three_factors_replay_complete_m11_and_parent_joint_program",
    "pred_c_bilinear_interaction_is_stably_necessary",
    "pred_d_at_least_one_linear_factor_is_stably_material",
    "pred_e_all_factor_arms_preserve_temporal_selectivity",
)
BARS = {"replay": 1e-5, "closure": 2e-6,
        "full_recovery_min": .9, "full_recovery_max": 1.1,
        "full_cosine": .98, "full_residual": .2, "full_direction": .9,
        "material": .03, "interaction": .03, "collateral": .01}
SUBSETS = {"none": (), "left": ("left",), "right": ("right",),
    "interaction": ("interaction",), "left_right": ("left", "right"),
    "left_interaction": ("left", "interaction"),
    "right_interaction": ("right", "interaction"), "all_three": FACTORS}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_product_factors(live_left, live_right, writer_left, writer_right):
    shapes = {tuple(x.shape) for x in (live_left, live_right, writer_left, writer_right)}
    if len(shapes) != 1 or live_left.ndim != 3:
        raise ValueError("factor tensors must share [batch, token, hidden] shape")
    delta_left = writer_left.float() - live_left.float()
    delta_right = writer_right.float() - live_right.float()
    return {
        "left": delta_left * live_right.float(),
        "right": live_left.float() * delta_right,
        "interaction": delta_left * delta_right,
    }


def compose_hidden(live_left, live_right, factors, subset):
    subset = tuple(subset)
    if len(set(subset)) != len(subset) or not set(subset).issubset(FACTORS):
        raise ValueError("invalid factor subset")
    hidden = live_left.float() * live_right.float()
    for name in subset:
        hidden = hidden + factors[name]
    return hidden


def closure_relative_error(live_left, live_right, writer_left, writer_right):
    factors = exact_product_factors(live_left, live_right, writer_left, writer_right)
    predicted = compose_hidden(live_left, live_right, factors, FACTORS)
    exact = writer_left.float() * writer_right.float()
    denominator = max(float(exact.norm()), 1e-30)
    return float((predicted - exact).norm()) / denominator


def eligibility(binding, result):
    return bool(binding.get("necessity_result_sha256") == sha(NECESSITY_RESULT)
        and binding.get("necessity_runner_sha256") == EXPECTED["necessity_runner"]
        and result.get("predictions", {}).get(
            "pred_a_authority_capture_full_replay_finiteness_and_exact_price") is True
        and result.get("predictions", {}).get(
            "pred_d_at_least_one_p7_mlp_is_stably_necessary_and_licenses_weight_factor_splitting") is True
        and result.get("predictions", {}).get(
            "pred_e_every_leave_one_out_arm_is_temporally_selective") is True
        and "M11" in result.get("stable_necessary_modules", ()))


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def capture_with_m11_factors(two, backend, tokens, **kwargs):
    saved = {}
    calls = {"left": 0, "right": 0}
    mlp = backend.model.transformer.h[11].mlp

    def capture(name):
        def hook(_module, _inputs, output):
            calls[name] += 1
            saved[name] = output.detach().clone()
        return hook

    handles = [mlp.Left.register_forward_hook(capture("left")),
               mlp.Right.register_forward_hook(capture("right"))]
    try:
        state = two.factorial.capture_state_and_modules(backend, tokens, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    if calls != {"left": 1, "right": 1} or set(saved) != set(FACTORS) - {"interaction"}:
        raise RuntimeError("M11 factor capture coverage changed")
    return state, saved, calls


def run_factor_arm(two, backend, tokens, entry, writer_factors, writer_modules,
                   position_rows, arm):
    if arm not in ARMS:
        raise ValueError("unknown M11 factor arm")
    subset = None if arm == "complete_M11" else SUBSETS[arm]
    mlp = backend.model.transformer.h[11].mlp
    entry_calls = 0
    calls = {"left": 0, "right": 0, "down": 0, "m11": 0}
    live = {}
    closure = []
    m11_output = None
    other_p7 = tuple(name for name in two.P7 if name != "M11")
    module_calls = {name: 0 for name in other_p7}
    handles = []

    def entry_hook(_module, arguments):
        nonlocal entry_calls
        entry_calls += 1
        if len(arguments) != 3:
            raise RuntimeError("block-10 input signature changed")
        changed = two.factorial.atlas.replace_positions(arguments[0], entry, position_rows)
        return (changed,) + tuple(arguments[1:])

    def save_lr(name):
        def hook(_module, _inputs, output):
            calls[name] += 1
            live[name] = output.detach().clone()
        return hook

    def patch_down(_module, arguments):
        calls["down"] += 1
        if set(live) != {"left", "right"}:
            raise RuntimeError("M11 Left/Right unavailable before Down")
        factors = exact_product_factors(live["left"], live["right"],
                                        writer_factors["left"], writer_factors["right"])
        closure.append(closure_relative_error(live["left"], live["right"],
                                              writer_factors["left"], writer_factors["right"]))
        if subset is None or not subset:
            return None
        predicted = compose_hidden(live["left"], live["right"], factors, subset)
        changed = arguments[0].clone()
        for row, positions in enumerate(position_rows):
            if positions:
                changed[row, positions] = predicted[row, positions].to(changed)
        return (changed,) + tuple(arguments[1:])

    def m11_hook(_module, _inputs, output):
        nonlocal m11_output
        calls["m11"] += 1
        changed = output
        if arm == "complete_M11":
            changed = two.factorial.atlas.replace_positions(
                output, writer_modules["M11"], position_rows)
        m11_output = changed.detach().clone()
        return changed

    def module_hook(name):
        def replace(_module, _inputs, output):
            module_calls[name] += 1
            return two.factorial.atlas.replace_positions(
                output, writer_modules[name], position_rows)
        return replace

    handles.append(backend.model.transformer.h[10].register_forward_pre_hook(entry_hook))
    handles.append(mlp.Left.register_forward_hook(save_lr("left")))
    handles.append(mlp.Right.register_forward_hook(save_lr("right")))
    handles.append(mlp.Down.register_forward_pre_hook(patch_down))
    handles.append(mlp.register_forward_hook(m11_hook))
    targets = two.factorial.atlas.module_targets(backend.model)
    for name in other_p7:
        handles.append(targets[name].register_forward_hook(module_hook(name)))
    try:
        logits, _captures = two.factorial.atlas.mediation.parent._forward(backend, tokens)
    finally:
        for handle in handles:
            handle.remove()
    if entry_calls != 1 or calls != {"left": 1, "right": 1, "down": 1, "m11": 1}:
        raise RuntimeError("M11 factor hook coverage changed")
    if set(module_calls.values()) != {1} or len(closure) != 1 or m11_output is None:
        raise RuntimeError("P7 module/closure coverage changed")
    return logits, {"entry": entry_calls, **calls, "modules": module_calls}, m11_output, closure[0]


def full_qualified(row):
    return bool(BARS["full_recovery_min"] <= row["signed_recovery"]
        <= BARS["full_recovery_max"] and row["cosine"] >= BARS["full_cosine"]
        and row["relative_residual"] <= BARS["full_residual"]
        and row["direction_agreement"] >= BARS["full_direction"])


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior = json.loads(PRIOR.read_text())
    base_ok = observed == EXPECTED and prior.get("price") == PRICE
    waiting = {"candidate_id": prior.get("candidate_id"), "status": "awaiting_binding",
        "authority_ok": base_ok, "gpu_accessed": False, "model_loaded": False,
        "queue_touched": False, "factors": list(FACTORS), "arms": list(ARMS),
        "price": PRICE}
    if not BINDING.exists() or not NECESSITY_RESULT.exists():
        print(json.dumps(waiting, sort_keys=True))
        return
    binding = json.loads(BINDING.read_text())
    result = json.loads(NECESSITY_RESULT.read_text())
    if not base_ok or not eligibility(binding, result):
        raise RuntimeError("M11 product-factor eligibility binding failed")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({**waiting, "status": "eligible_bound"}, sort_keys=True))
        return

    import numpy as np
    from circuit_fast_screen_managed_runner import atomic_create_json
    import run_temporal_iswas_p7_residual_two_stream_ood_composition_v1 as two

    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    loc = two.factorial.atlas.mediation.parent.loc
    model_parent = two.factorial.atlas.mediation.parent
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
    selected_heads = tuple(json.loads(two.factorial.GREEDY_RESULT.read_text())[
        "selected_prefixes"][role]["heads"])

    with torch.no_grad():
        native_state, _native_m11, native_factor_calls = capture_with_m11_factors(
            two, backend, tokens)
        (native_logits, head_captures, _native_l11, _native_entry, _native_modules,
         native_entry_calls, native_module_calls) = native_state
        writer_state, writer_m11, writer_factor_calls = capture_with_m11_factors(
            two, backend, tokens, head_captures=head_captures,
            selected_labels=selected_heads, pairs=pairs, position_rows=source_rows)
        (writer_logits, _unused, _writer_l11, writer_entry, writer_modules,
         writer_entry_calls, writer_module_calls) = writer_state
        logits, calls, m11_outputs, closures = {}, {}, {}, {}
        for arm in ARMS:
            logits[arm], calls[arm], m11_outputs[arm], closures[arm] = run_factor_arm(
                two, backend, tokens, writer_entry, writer_m11,
                writer_modules, position_rows, arm)

    native = {name: loc.margins(native_logits, endpoints, name)
              for name in model_parent.LOCKED}
    writer_effect = loc.margins(writer_logits, endpoints, role) - native[role]
    effects = {arm: loc.margins(value, endpoints, role) - native[role]
               for arm, value in logits.items()}
    other_effects = {arm: loc.margins(value, endpoints, other) - native[other]
                     for arm, value in logits.items()}
    other_gold = native[other][other_pairs] - native[other]
    reports = []
    for phase in ("FIT", "HOLDOUT"):
        chosen = np.asarray([row["phase"] == phase for row, _cell, _endpoint in endpoints])
        temporal_norm = max(float(np.linalg.norm(other_gold[chosen])), 1e-30)
        for arm in ARMS:
            metric = two.factorial.atlas.mediation.accounting.effect_metrics(
                effects[arm][chosen], writer_effect[chosen], other_effects[arm][chosen])
            reports.append({"phase": phase, "arm": arm, **metric,
                "temporal_command_gold_collateral":
                    float(np.linalg.norm(other_effects[arm][chosen]) / temporal_norm)})
    by = {(row["phase"], row["arm"]): row for row in reports}
    parent_by = {(row["phase"], row["arm"]): row for row in
                 json.loads(PARENT_RESULT.read_text())["reports"]}
    necessity_by = {(row["phase"], row["arm"]): row for row in result["reports"]}
    replay_errors = []
    for phase in ("FIT", "HOLDOUT"):
        for arm, reference in (("complete_M11", parent_by[(phase, "P7_plus_identity")]),
                               ("all_three", parent_by[(phase, "P7_plus_identity")]),
                               ("none", necessity_by[(phase, "without_M11")])):
            for field in ("signed_recovery", "cosine", "relative_residual",
                          "direction_agreement", "non_target_to_target_gold_norm",
                          "temporal_command_gold_collateral"):
                replay_errors.append(abs(by[(phase, arm)][field] - reference[field]))
    replay = max(replay_errors)
    output_errors = []
    for arm in ("all_three", "complete_M11"):
        for row, positions in enumerate(position_rows):
            if positions:
                output_errors.append(float((m11_outputs[arm][row, positions].float()
                    - writer_modules["M11"][row, positions].float()).abs().max()))
    output_replay = max(output_errors, default=0.0)
    factor_effects = {}
    for phase in ("FIT", "HOLDOUT"):
        none = by[(phase, "none")]["signed_recovery"]
        full = by[(phase, "all_three")]["signed_recovery"]
        factor_effects[phase] = {
            "left_increment": by[(phase, "left")]["signed_recovery"] - none,
            "right_increment": by[(phase, "right")]["signed_recovery"] - none,
            "interaction_increment": by[(phase, "interaction")]["signed_recovery"] - none,
            "interaction_necessity_drop": full - by[(phase, "left_right")]["signed_recovery"],
            "complete_m11_contribution": full - none,
        }

    hook_ok = bool(native_factor_calls == writer_factor_calls == {"left": 1, "right": 1}
        and native_entry_calls == writer_entry_calls == 1
        and set(native_module_calls.values()) == {1}
        and set(writer_module_calls.values()) == {1}
        and all(record["entry"] == record["left"] == record["right"]
                == record["down"] == record["m11"] == 1
                and set(record["modules"].values()) == {1} for record in calls.values()))
    A = bool(base_ok and eligibility(binding, result)
        and len(endpoints) * 11 == PRICE["sequence_evaluations"] and hook_ok
        and max(closures.values()) <= BARS["closure"]
        and finite({"reports": reports, "factor_effects": factor_effects,
                    "closures": closures}))
    B = bool(replay <= BARS["replay"] and output_replay <= BARS["replay"]
        and all(full_qualified(by[(phase, arm)]) for phase in ("FIT", "HOLDOUT")
                for arm in ("complete_M11", "all_three")))
    C = all(factor_effects[phase]["interaction_necessity_drop"] >= BARS["interaction"]
            for phase in ("FIT", "HOLDOUT"))
    D = any(all(factor_effects[phase][f"{factor}_increment"] >= BARS["material"]
                for phase in ("FIT", "HOLDOUT")) for factor in ("left", "right"))
    E = all(row["temporal_command_gold_collateral"] <= BARS["collateral"]
            for row in reports)
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A or not B else "nonselective" if not E else
        "factor_attribution_null" if not C and not D else
        "interaction_only_or_synergistic" if C and not D else
        "linear_m11_response" if D and not C else "exact_three_factor_m11_interface")
    payload = {"schema": "temporal_iswas_p7_identity_m11_exact_product_factorial_ood_result_v1",
        "candidate_id": prior["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "binding_sha256": sha(BINDING), "eligibility_result_sha256": sha(NECESSITY_RESULT),
        "selected_writer_heads": list(selected_heads), "reports": reports,
        "factor_effects": factor_effects, "factor_closure_relative_l2": closures,
        "parent_replay_max_abs_error": replay,
        "m11_output_replay_max_abs_error": output_replay, "hook_calls": calls,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, payload)
    print(json.dumps({"factor_effects": factor_effects,
        "factor_closure_relative_l2": closures, "parent_replay_max_abs_error": replay,
        "m11_output_replay_max_abs_error": output_replay,
        "predictions": predictions, "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
