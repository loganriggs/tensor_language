#!/usr/bin/env python3
"""Conditionally split necessary A11 into fixed singleton/LOO head endpoints."""

# BQGATE: EXPERIMENT pred_a_conditional_authority_capture_hook_coverage_finiteness_and_exact_price pred_b_all_nine_heads_replay_complete_a11_and_parent_joint_program pred_c_prospectively_nominated_h3_is_stably_necessary_and_sufficient_at_the_endpoint pred_d_h3_explains_at_least_seventy_percent_of_complete_a11_contribution pred_e_all_head_arms_preserve_temporal_selectivity
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_iswas_p7_residual_two_stream_ood_composition_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_p7_residual_two_stream_ood_composition_v1.py"
NECESSITY_RUNNER = ROOT / "ops/run_temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1.py"
NECESSITY_RESULT = ROOT / "circuits/followups/temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1_result.json"
BINDING = ROOT / "circuits/bindings/temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1.json"
OUT = ROOT / "circuits/followups/temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1_result.json"
EXPECTED = {
    "prior": "afca834f62f088ad689619d8c58d424344b55bf59eb24f356badb57c7759ed99",
    "parent_result": "55498ec9110fbddc8ae9a86eb4bab63efeb92540322aaa8783cec2d8c7bd55a7",
    "parent_runner": "0e3f6de1b273d372006652e37c085ddf15d5b120795225ba873de82a402a19ec",
    "necessity_runner": "384a79a38294416204d5bde0e24fcee7008e391a6712de82ec59c627f7f1318a",
}
FILES = {"prior": PRIOR, "parent_result": PARENT_RESULT,
         "parent_runner": PARENT_RUNNER, "necessity_runner": NECESSITY_RUNNER}
HEADS = tuple(range(9))
ARMS = (("none", "all_nine")
        + tuple(f"H{head}_singleton" for head in HEADS)
        + tuple(f"without_H{head}" for head in HEADS))
PRICE = {"checkpoint_loads": 1, "model_forwards": 22,
         "sequence_evaluations": 2816, "scored_token_positions": 5632,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_conditional_authority_capture_hook_coverage_finiteness_and_exact_price",
    "pred_b_all_nine_heads_replay_complete_a11_and_parent_joint_program",
    "pred_c_prospectively_nominated_h3_is_stably_necessary_and_sufficient_at_the_endpoint",
    "pred_d_h3_explains_at_least_seventy_percent_of_complete_a11_contribution",
    "pred_e_all_head_arms_preserve_temporal_selectivity",
)
BARS = {"replay": 1e-5, "full_recovery_min": .9, "full_recovery_max": 1.1,
        "full_cosine": .98, "full_residual": .2, "full_direction": .9,
        "increment": .03, "loo": .03, "attribution": .03,
        "h3_share": .70, "collateral": .01}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_head_slices(native, writer, heads, position_rows, *, n_heads=9):
    """Replace selected concatenated pre-c_proj head slices at registered row positions."""
    heads = tuple(int(head) for head in heads)
    if native.shape != writer.shape or native.ndim != 3:
        raise ValueError("head tensors must share [batch, token, residual] shape")
    if len(set(heads)) != len(heads) or any(head < 0 or head >= n_heads for head in heads):
        raise ValueError("heads must be unique valid indices")
    if native.shape[-1] % n_heads:
        raise ValueError("residual width must divide into heads")
    if len(position_rows) != native.shape[0]:
        raise ValueError("position row count mismatch")
    width = native.shape[-1] // n_heads
    changed = native.clone()
    for row, positions in enumerate(position_rows):
        for position in positions:
            if position < 0 or position >= native.shape[1]:
                raise ValueError("head patch position out of range")
            for head in heads:
                sl = slice(head * width, (head + 1) * width)
                changed[row, position, sl] = writer[row, position, sl]
    return changed


def eligibility(binding, result):
    return bool(binding.get("necessity_result_sha256") == sha(NECESSITY_RESULT)
        and binding.get("necessity_runner_sha256") == EXPECTED["necessity_runner"]
        and result.get("predictions", {}).get(
            "pred_a_authority_capture_full_replay_finiteness_and_exact_price") is True
        and result.get("predictions", {}).get(
            "pred_c_attention11_is_stably_necessary_and_licenses_head_splitting") is True
        and result.get("predictions", {}).get(
            "pred_e_every_leave_one_out_arm_is_temporally_selective") is True
        and "A11" in result.get("stable_necessary_modules", ()))


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def capture_with_a11_heads(two, backend, tokens, **kwargs):
    calls = 0
    saved = None

    def capture(_module, inputs):
        nonlocal calls, saved
        calls += 1
        saved = inputs[0].detach().clone()

    handle = backend.model.transformer.h[11].attn.c_proj.register_forward_pre_hook(capture)
    try:
        state = two.factorial.capture_state_and_modules(backend, tokens, **kwargs)
    finally:
        handle.remove()
    if calls != 1 or saved is None:
        raise RuntimeError("A11 head capture coverage changed")
    return state, saved, calls


def run_head_arm(two, backend, tokens, entry, writer_heads,
                 writer_modules, position_rows, heads):
    heads = tuple(heads)
    entry_calls = 0
    head_calls = 0
    output_calls = 0
    output_capture = None
    mlps = tuple(name for name in two.P7 if name != "A11")
    module_calls = {name: 0 for name in mlps}
    handles = []

    def entry_hook(_module, arguments):
        nonlocal entry_calls
        entry_calls += 1
        changed = two.factorial.atlas.replace_positions(arguments[0], entry, position_rows)
        return (changed,) + tuple(arguments[1:])

    def head_hook(_module, inputs):
        nonlocal head_calls
        head_calls += 1
        changed = replace_head_slices(inputs[0], writer_heads, heads, position_rows)
        return (changed,) + tuple(inputs[1:])

    def output_hook(_module, _inputs, output):
        nonlocal output_calls, output_capture
        output_calls += 1
        output_capture = output.detach().clone()

    def module_hook(name):
        def replace(_module, _inputs, output):
            module_calls[name] += 1
            return two.factorial.atlas.replace_positions(
                output, writer_modules[name], position_rows)
        return replace

    block10 = backend.model.transformer.h[10]
    cproj = backend.model.transformer.h[11].attn.c_proj
    handles.append(block10.register_forward_pre_hook(entry_hook))
    handles.append(cproj.register_forward_pre_hook(head_hook))
    handles.append(cproj.register_forward_hook(output_hook))
    targets = two.factorial.atlas.module_targets(backend.model)
    for name in mlps:
        handles.append(targets[name].register_forward_hook(module_hook(name)))
    try:
        logits, _captures = two.factorial.atlas.mediation.parent._forward(backend, tokens)
    finally:
        for handle in handles:
            handle.remove()
    if entry_calls != 1 or head_calls != 1 or output_calls != 1:
        raise RuntimeError("entry/head/output hook coverage changed")
    if set(module_calls.values()) != {1}:
        raise RuntimeError("P7 MLP hook coverage changed")
    return logits, {"entry": entry_calls, "head": head_calls,
                    "output": output_calls, "modules": module_calls}, output_capture


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
        "queue_touched": False, "arms": list(ARMS), "price": PRICE}
    if not BINDING.exists() or not NECESSITY_RESULT.exists():
        print(json.dumps(waiting, sort_keys=True))
        return
    binding = json.loads(BINDING.read_text())
    result = json.loads(NECESSITY_RESULT.read_text())
    if not base_ok or not eligibility(binding, result):
        raise RuntimeError("A11 endpoint atlas eligibility binding failed")
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
        native_state, _native_a11, native_head_calls = capture_with_a11_heads(
            two, backend, tokens)
        (native_logits, head_captures, _native_l11, _native_entry, _native_modules,
         native_entry_calls, native_module_calls) = native_state
        writer_state, writer_a11, writer_head_calls = capture_with_a11_heads(
            two, backend, tokens, head_captures=head_captures,
            selected_labels=selected_heads, pairs=pairs, position_rows=source_rows)
        (writer_logits, _unused, _writer_l11, writer_entry, writer_modules,
         writer_entry_calls, writer_module_calls) = writer_state
        arm_heads = {"none": (), "all_nine": HEADS}
        arm_heads.update({f"H{head}_singleton": (head,) for head in HEADS})
        arm_heads.update({f"without_H{head}": tuple(h for h in HEADS if h != head)
                          for head in HEADS})
        logits = {}
        calls = {}
        a11_outputs = {}
        for arm in ARMS:
            logits[arm], calls[arm], a11_outputs[arm] = run_head_arm(
                two, backend, tokens, writer_entry, writer_a11,
                writer_modules, position_rows, arm_heads[arm])

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
        for arm, reference in (("all_nine", parent_by[(phase, "P7_plus_identity")]),
                               ("none", necessity_by[(phase, "without_A11")])):
            for field in ("signed_recovery", "cosine", "relative_residual",
                          "direction_agreement", "non_target_to_target_gold_norm",
                          "temporal_command_gold_collateral"):
                replay_errors.append(abs(by[(phase, arm)][field] - reference[field]))
    replay = max(replay_errors)
    cproj_errors = []
    for row, positions in enumerate(position_rows):
        if positions:
            cproj_errors.append(float((a11_outputs["all_nine"][row, positions].float()
                - writer_modules["A11"][row, positions].float()).abs().max()))
    cproj_replay = max(cproj_errors, default=0.0)
    attribution = {}
    for head in HEADS:
        name = f"H{head}"
        attribution[name] = {}
        for phase in ("FIT", "HOLDOUT"):
            none = by[(phase, "none")]["signed_recovery"]
            full = by[(phase, "all_nine")]["signed_recovery"]
            singleton = by[(phase, f"H{head}_singleton")]["signed_recovery"] - none
            loo = full - by[(phase, f"without_H{head}")]["signed_recovery"]
            endpoint = .5 * (singleton + loo)
            contribution = full - none
            share = endpoint / contribution if contribution > 1e-12 else float("nan")
            attribution[name][phase] = {"singleton_increment": singleton,
                "leave_one_out_drop": loo, "endpoint_attribution": endpoint,
                "complete_a11_contribution": contribution, "a11_share": share}

    hook_ok = bool(native_head_calls == writer_head_calls == 1
        and native_entry_calls == writer_entry_calls == 1
        and set(native_module_calls.values()) == {1}
        and set(writer_module_calls.values()) == {1}
        and all(record["entry"] == record["head"] == record["output"] == 1
                and set(record["modules"].values()) == {1} for record in calls.values()))
    A = bool(base_ok and eligibility(binding, result)
        and len(endpoints) * 22 == PRICE["sequence_evaluations"] and hook_ok
        and finite({"reports": reports, "attribution": attribution}))
    B = bool(replay <= BARS["replay"] and cproj_replay <= BARS["replay"]
        and all(full_qualified(by[(phase, "all_nine")])
                for phase in ("FIT", "HOLDOUT")))
    h3 = attribution["H3"]
    C = all(h3[phase]["singleton_increment"] >= BARS["increment"]
        and h3[phase]["leave_one_out_drop"] >= BARS["loo"]
        and h3[phase]["endpoint_attribution"] >= BARS["attribution"]
        for phase in ("FIT", "HOLDOUT"))
    D = all(h3[phase]["complete_a11_contribution"] > 0
        and h3[phase]["a11_share"] >= BARS["h3_share"]
        for phase in ("FIT", "HOLDOUT"))
    E = all(row["temporal_command_gold_collateral"] <= BARS["collateral"]
            for row in reports)
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A or not B else "nonselective" if not E else
        "distributed_a11_heads" if not C else "h3_plus_distributed_support" if not D else
        "h3_compact_a11_interface")
    payload = {"schema": "temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_result_v1",
        "candidate_id": prior["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "binding_sha256": sha(BINDING), "eligibility_result_sha256": sha(NECESSITY_RESULT),
        "selected_writer_heads": list(selected_heads), "reports": reports,
        "head_endpoint_attribution": attribution, "parent_replay_max_abs_error": replay,
        "a11_cproj_replay_max_abs_error": cproj_replay, "hook_calls": calls,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, payload)
    print(json.dumps({"head_endpoint_attribution": attribution,
        "parent_replay_max_abs_error": replay, "a11_cproj_replay_max_abs_error": cproj_replay,
        "predictions": predictions, "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
