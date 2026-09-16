#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_fresh_response_replay pred_c_fresh_bidirectional_causality pred_d_fresh_preservation pred_e_beats_coarse_program pred_f_frozen_support_specificity
"""Frozen cue-role-crossover test of the eight-edge head17.2 source program."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]

import torch

import circuit_fast_screen_managed_runner as managed
import run_setting2_regional_attention17h2_head_source_beam_v1 as base
import run_setting2_regional_attention17h2_port_preservation_search_v1 as search
import run_setting2_regional_attention17h2_port_source_fold_v1 as port
from regional_cue_row_check_v1 import validate


RUNNER = Path(__file__).resolve()
PREREG = P / "SETTING2_REGIONAL_ATTENTION17H2_HEAD_SOURCE_FRESH_V1_PREREGISTRATION.md"
ROWS = P / "REGIONAL_CITY_ROLE_CROSSOVER_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_ATTENTION17H2_HEAD_SOURCE_FRESH_V1_BINDING.json"
PARENT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_head_source_beam_v1_result.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_head_source_fresh_v1_result.json"
FROZEN = (0, 3, 6, 11, 16, 18, 19, 21)
COARSE = base.COARSE
TARGET_ONLY = base.TARGET_ONLY
_rng = random.Random(202609160548)
NULLS = set()
while len(NULLS) < 16:
    candidate = tuple(sorted(_rng.sample(range(len(base.NAMES)), len(FROZEN))))
    if candidate not in (FROZEN, COARSE, TARGET_ONLY):
        NULLS.add(candidate)
SUPPORTS = (FROZEN, COARSE, TARGET_ONLY) + tuple(sorted(NULLS))
PRICE = {
    "physical_model_executions": 12,
    "full_model_sequences": 96,
    "source_terms": 23,
    "support_candidates": 19,
    "candidate_suffix_sequences": 1824,
    "complete_corner_suffix_sequences": 96,
    "token_logits_per_sequence": 12,
    "support_nulls": 16,
    "fits": 0,
    "backwards": 0,
    "parameter_updates": 0,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    files = {
        "preregistration": PREREG,
        "rows": ROWS,
        "row_check": P / "regional_cue_row_check_v1.py",
        "parent_result": PARENT,
        "discovery_runner": Path(base.__file__).resolve(),
        "search_runner": Path(search.__file__).resolve(),
        "port_runner": Path(port.__file__).resolve(),
    }
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    parent = json.loads(PARENT.read_text())
    frozen_names = [base.NAMES[index] for index in FROZEN]
    if parent["terminal"] != "head17_2_sparse_head_source_candidate" or parent["selected"]["support"] != frozen_names or not all(parent["predictions"].values()):
        raise ValueError("discovery head program changed")
    rows = json.loads(ROWS.read_text())["rows"]
    checks = validate(rows)
    buckets = {}
    for index, row in enumerate(rows):
        buckets.setdefault(len(row["ids"]), []).append(index)
    executions = 2 * sum(math.ceil(len(indices) / 8) for indices in buckets.values())
    if len(rows) != 48 or executions != PRICE["physical_model_executions"] or len(SUPPORTS) != PRICE["support_candidates"]:
        raise ValueError("row, support, or execution price changed")
    return binding, parent, rows, checks, buckets


def plan():
    _, _, rows, checks, _ = load_bound()
    return {
        "schema": "setting2_regional_attention17h2_head_source_fresh_v1_plan",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "rows": len(rows),
        "source_terms": base.NAMES,
        "frozen_support": [base.NAMES[index] for index in FROZEN],
        "selection": "none on crossover panel",
        "price": PRICE,
        "row_checks": checks,
        "binding_sha256": sha(BINDING),
    }


@torch.no_grad()
def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast

    model = load_model_fast().cuda().eval()
    _, _, rows, checks, buckets = load_bound()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    lambdas = torch.stack([block.lambdas.detach().double().cpu() for block in model.transformer.h])
    embed_c, write_c = port.fold.helper.coefficients(lambdas)
    gammas = {layer: float(torch.prod(lambdas[layer + 1:18, 0])) for layer in range(9, 17)}
    unembed = model.state_dict()["lm_head.weight"].double().cpu()
    output_weight = model.transformer.h[17].attn.c_proj.weight[:, 2 * 128:3 * 128]
    readers = torch.stack([unembed[row["uk_id"]] - unembed[row["us_id"]] for row in rows])
    pairs = torch.tensor([[(row["uk_id"], row["us_id"])] + [pair for _, pair in port.fold.READOUTS[1:]] for row in rows], device=device, dtype=torch.long)
    cached = []
    native_pre = torch.empty(48, 1152, dtype=torch.float64)
    edited_pre = torch.empty_like(native_pre)
    native_values = torch.empty(48, len(port.fold.READOUTS), dtype=torch.float64)
    edited_values = torch.empty_like(native_values)
    full_delta = torch.empty(48, 1152, dtype=torch.float64)
    full_response = torch.empty(48, dtype=torch.float64)
    audit_num = {name: 0.0 for name in ("carry", "attention9", "source_raw", "first_state", "zero_endpoint", "full_endpoint", "head10_rounding", "head11_rounding")}
    audit_den = {name: 0.0 for name in audit_num}
    executions = 0

    for _, indices in sorted(buckets.items()):
        for offset in range(0, len(indices), 8):
            row_ids = indices[offset:offset + 8]
            tokens = torch.tensor([rows[index]["ids"] for index in row_ids], device=device)
            native = base.split_trace(model, tokens, False, embed_c, write_c)
            edited = base.split_trace(model, tokens, True, embed_c, write_c)
            executions += 2
            components = torch.stack([gammas[base.propagation_layer(name)] * (edited["writes"][name] - native["writes"][name]).double() for name in base.NAMES])
            whole = torch.stack([gammas[layer] * (edited["writes"][kind + str(layer)] - native["writes"][kind + str(layer)]).double() for layer in range(9, 17) for kind in ("attn", "mlp")])
            actual_state_delta = (edited["residual17"] - native["residual17"]).double()
            reconstructed = whole.sum(0)
            reconstructed_state = (native["residual17"].double() + reconstructed).to(model_dtype)
            full_score2, full_value = port.edited_ports(model, reconstructed_state, native["first"])
            zero_corner = port.corner_write(native["score1"], native["score2"], native["value"], output_weight)
            complete_corner = port.corner_write(native["score1"], edited["score2"], edited["value"], output_weight)
            reconstructed_corner = port.corner_write(native["score1"], full_score2, full_value, output_weight)
            batch_full_delta = complete_corner - zero_corner
            audit_num["carry"] += native["carry_num"]
            audit_den["carry"] += native["carry_den"]
            audit_num["attention9"] += native["attention9_num"]
            audit_den["attention9"] += native["attention9_den"]
            audit_num["source_raw"] += float((reconstructed - actual_state_delta).square().sum())
            audit_den["source_raw"] += float(actual_state_delta.square().sum())
            audit_num["first_state"] += float((native["first"].double() - edited["first"].double()).square().sum())
            audit_den["first_state"] += float(native["first"].double().square().sum())
            audit_num["zero_endpoint"] += float((zero_corner - native["head2"]).square().sum())
            audit_den["zero_endpoint"] += float(native["head2"].square().sum())
            audit_num["full_endpoint"] += float((reconstructed_corner - complete_corner).square().sum())
            audit_den["full_endpoint"] += float(complete_corner.square().sum())
            for layer in (10, 11):
                key = f"head{layer}_rounding"
                for trace in (native, edited):
                    audit_num[key] += trace["head_partition_correction"][str(layer)]["num"]
                    audit_den[key] += trace["head_partition_correction"][str(layer)]["den"]
            native_pre[row_ids] = native["pre"].double().cpu()
            edited_pre[row_ids] = edited["pre"].double().cpu()
            native_values[row_ids] = port.fold.score_readouts(native["logits"], [rows[index] for index in row_ids])
            edited_values[row_ids] = port.fold.score_readouts(edited["logits"], [rows[index] for index in row_ids])
            full_delta[row_ids] = batch_full_delta.cpu()
            full_response[row_ids] = (batch_full_delta.cpu() * readers[row_ids]).sum(-1)
            cached.append({"rows": row_ids, "native_state": native["residual17"], "first": native["first"], "score1": native["score1"], "native_corner": zero_corner, "components": components})

    count = len(SUPPORTS)
    candidate_deltas = torch.empty(count, 48, 1152, dtype=torch.float64)
    candidate_responses = torch.empty(count, 48, dtype=torch.float64)
    mask = torch.zeros(count, len(base.NAMES), dtype=torch.float64, device=device)
    for index, support in enumerate(SUPPORTS):
        mask[index, list(support)] = 1
    for batch in cached:
        additions = torch.einsum("sm,mbtd->sbtd", mask, batch["components"])
        _, batch_size, length, width = additions.shape
        states = (batch["native_state"].double().unsqueeze(0) + additions).to(model_dtype).reshape(count * batch_size, length, width)
        first = batch["first"].unsqueeze(0).expand(count, -1, -1, -1, -1).reshape(count * batch_size, length, 9, 128)
        score1 = batch["score1"].unsqueeze(0).expand(count, -1, -1, -1).reshape(count * batch_size, length, length)
        score2, value = port.edited_ports(model, states, first)
        writes = port.corner_write(score1, score2, value, output_weight).reshape(count, batch_size, 1152) - batch["native_corner"].unsqueeze(0)
        row_ids = batch["rows"]
        candidate_deltas[:, row_ids] = writes.cpu()
        candidate_responses[:, row_ids] = torch.einsum("sbd,bd->sb", writes.cpu(), readers[row_ids])

    row_order = torch.arange(48, device=device)
    native_pre_device = native_pre.to(device)
    edited_pre_device = edited_pre.to(device)
    full_delta_device = full_delta.to(device)
    full_install_values = search.sparse_suffix_readouts(model, (native_pre_device + full_delta_device).to(model_dtype), row_order, pairs)
    full_remove_values = search.sparse_suffix_readouts(model, (edited_pre_device - full_delta_device).to(model_dtype), row_order, pairs)
    full_install = full_install_values - native_values
    full_remove = full_remove_values - edited_values
    install_values = torch.empty(count, 48, len(port.fold.READOUTS), dtype=torch.float64)
    remove_values = torch.empty_like(install_values)
    for start in range(0, count, 8):
        stop = min(start + 8, count)
        subcount = stop - start
        deltas = candidate_deltas[start:stop].to(device)
        repeated_rows = row_order.repeat(subcount)
        install = search.sparse_suffix_readouts(model, (native_pre_device.unsqueeze(0) + deltas).to(model_dtype).reshape(subcount * 48, 1152), repeated_rows, pairs).reshape(subcount, 48, -1)
        remove = search.sparse_suffix_readouts(model, (edited_pre_device.unsqueeze(0) - deltas).to(model_dtype).reshape(subcount * 48, 1152), repeated_rows, pairs).reshape(subcount, 48, -1)
        install_values[start:stop] = install - native_values.unsqueeze(0)
        remove_values[start:stop] = remove - edited_values.unsqueeze(0)

    search.NAMES = base.NAMES
    reports = [search.candidate_metrics(index, support, candidate_responses[index], full_response, install_values, remove_values, full_install, full_remove, rows) for index, support in enumerate(SUPPORTS)]
    selected, coarse, target_only = reports[:3]
    null_reports = [{"support": report["support"], "minimax_normalized_gate_score": report["minimax_normalized_gate_score"]} for report in reports[3:]]
    null_median = float(torch.tensor([report["minimax_normalized_gate_score"] for report in null_reports], dtype=torch.float64).median())
    errors = {name + "_relative_error": math.sqrt(audit_num[name] / max(audit_den[name], 1e-30)) for name in audit_num}
    raw_source = errors.pop("source_raw_relative_error")
    errors["source_closure_relative_error"] = 0.0
    errors["bf16_recurrence_rounding_residual_relative_norm"] = raw_source
    exact_keys = [key for key in errors if key != "bf16_recurrence_rounding_residual_relative_norm"]
    instrument = executions == PRICE["physical_model_executions"] and max(errors[key] for key in exact_keys) <= 2e-6 and raw_source < 1e-5
    predictions = {
        "pred_a_exact_instrument": bool(instrument),
        "pred_b_fresh_response_replay": bool(instrument and selected["passes_response"]),
        "pred_c_fresh_bidirectional_causality": bool(instrument and selected["passes_causality"]),
        "pred_d_fresh_preservation": bool(instrument and selected["passes_controls"]),
        "pred_e_beats_coarse_program": bool(instrument and selected["minimax_normalized_gate_score"] < coarse["minimax_normalized_gate_score"]),
        "pred_f_frozen_support_specificity": bool(instrument and selected["minimax_normalized_gate_score"] + .10 <= null_median),
    }
    terminal = "fresh_head17_2_sparse_head_source_program" if all(predictions.values()) else "valid_fresh_head_source_null" if instrument else "invalid"
    result = {
        "schema": "setting2_regional_attention17h2_head_source_fresh_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "instrument_errors": errors,
        "selected": selected,
        "coarse_five_module_expansion_baseline": coarse,
        "target_only_baseline": target_only,
        "support_null_reports": null_reports,
        "support_null_median_minimax_score": null_median,
        "selection_rule": "frozen discovery support; no crossover-panel selection",
        "price": PRICE,
        "row_checks": checks,
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Frozen head-level source support on cue-role crossover rows; module deltas remain native counterfactual ports and no donor-free claim is made.",
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "errors": errors, "selected": selected, "coarse": coarse, "target_only": target_only, "null_median": null_median}, indent=2))


if __name__ == "__main__":
    main()
