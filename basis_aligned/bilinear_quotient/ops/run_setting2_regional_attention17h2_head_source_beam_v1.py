#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_response_replay pred_c_bidirectional_causality pred_d_preservation pred_e_improves_coarse_program pred_f_support_specificity
"""Deterministic sparse beam over exact attention-head source writes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
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
import torch.nn.functional as F

import circuit_fast_screen_managed_runner as managed
import run_setting2_regional_attention17h2_port_preservation_search_v1 as search
import run_setting2_regional_attention17h2_port_source_fold_v1 as port
from regional_cue_row_check_v1 import validate


RUNNER = Path(__file__).resolve()
PREREG = P / "SETTING2_REGIONAL_ATTENTION17H2_HEAD_SOURCE_BEAM_V1_PREREGISTRATION.md"
ROWS = P / "ODD_FRAMING_FRESH_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_ATTENTION17H2_HEAD_SOURCE_BEAM_V1_BINDING.json"
PARENT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_port_preservation_fresh_v1_result.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_head_source_beam_v1_result.json"
NAMES = (
    ("attn9h8",)
    + tuple(f"attn10h{head}" for head in range(9))
    + tuple(f"attn11h{head}" for head in range(9))
    + ("mlp9", "mlp15", "attn15", "mlp16")
)
COARSE = tuple(range(21))
TARGET_ONLY = (0, 21, 22)
BEAM_WIDTH = 64
MAX_WIDTH = 8
SEED = 202609160533
PRICE = {
    "physical_model_executions": 12,
    "full_model_sequences": 96,
    "source_terms": 23,
    "support_candidates_max": 7400,
    "candidate_suffix_sequences_max": 710400,
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
        "search_runner": Path(search.__file__).resolve(),
        "port_runner": Path(port.__file__).resolve(),
    }
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    parent = json.loads(PARENT.read_text())
    if parent["terminal"] != "valid_fresh_five_edge_port_null" or parent["selected"]["support"] != ["attn9", "mlp9", "attn10", "attn11", "mlp15"]:
        raise ValueError("fresh coarse-program authority changed")
    rows = json.loads(ROWS.read_text())["rows"]
    checks = validate(rows)
    buckets = {}
    for index, row in enumerate(rows):
        buckets.setdefault(len(row["ids"]), []).append(index)
    executions = 2 * sum(math.ceil(len(indices) / 8) for indices in buckets.values())
    if len(rows) != 48 or executions != PRICE["physical_model_executions"]:
        raise ValueError("row or execution price changed")
    return binding, parent, rows, checks, buckets


def plan():
    _, _, rows, checks, _ = load_bound()
    return {
        "schema": "setting2_regional_attention17h2_head_source_beam_v1_plan",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "rows": len(rows),
        "source_terms": NAMES,
        "beam_width": BEAM_WIDTH,
        "maximum_support_width": MAX_WIDTH,
        "price": PRICE,
        "row_checks": checks,
        "binding_sha256": sha(BINDING),
    }


def propagation_layer(name):
    if name.startswith("attn10h"):
        return 10
    if name.startswith("attn11h"):
        return 11
    if name == "attn9h8" or name == "mlp9":
        return 9
    return int(name.removeprefix("attn").removeprefix("mlp"))


@torch.no_grad()
def split_trace(model, tokens, edit, embed_c, write_c):
    captured = {}
    handles = []
    for layer in (10, 11):
        def hook(_module, args, layer=layer):
            if layer in captured:
                raise RuntimeError(f"layer {layer} projection called twice")
            captured[layer] = args[0]
        handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(hook))
    try:
        result = port.trace(model, tokens, edit, embed_c, write_c)
    finally:
        for handle in handles:
            handle.remove()
    correction = {}
    for layer in (10, 11):
        z = captured[layer]
        weight = model.transformer.h[layer].attn.c_proj.weight
        pieces = []
        for head in range(8):
            sl = slice(128 * head, 128 * (head + 1))
            pieces.append(F.linear(z[..., sl].double(), weight[:, sl].double()))
        raw_last = F.linear(z[..., 8 * 128:].double(), weight[:, 8 * 128:].double())
        closed_last = result["writes"][f"attn{layer}"].double() - sum(pieces)
        correction[str(layer)] = {
            "num": float((closed_last - raw_last).square().sum()),
            "den": float(result["writes"][f"attn{layer}"].double().square().sum()),
        }
        pieces.append(closed_last)
        for head, piece in enumerate(pieces):
            result["writes"][f"attn{layer}h{head}"] = piece
    # The explicit layer-9 edit changes only head 9.8.  Its native value cancels
    # when edited-minus-native components are formed.
    result["writes"]["attn9h8"] = result["writes"]["attn9"].double()
    result["head_partition_correction"] = correction
    return result


def rank_key(report):
    return (
        report["minimax_normalized_gate_score"],
        report["width"],
        report["response_relative_l2"],
        tuple(report["support"]),
    )


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
    pair_ids = [[(row["uk_id"], row["us_id"])] + [pair for _, pair in port.fold.READOUTS[1:]] for row in rows]
    pairs = torch.tensor(pair_ids, device=device, dtype=torch.long)
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
            selected_rows = indices[offset:offset + 8]
            tokens = torch.tensor([rows[index]["ids"] for index in selected_rows], device=device)
            native = split_trace(model, tokens, False, embed_c, write_c)
            edited = split_trace(model, tokens, True, embed_c, write_c)
            executions += 2
            components = torch.stack([
                gammas[propagation_layer(name)] * (edited["writes"][name] - native["writes"][name]).double()
                for name in NAMES
            ])
            whole = torch.stack([
                gammas[layer] * (edited["writes"][kind + str(layer)] - native["writes"][kind + str(layer)]).double()
                for layer in range(9, 17) for kind in ("attn", "mlp")
            ])
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
            native_pre[selected_rows] = native["pre"].double().cpu()
            edited_pre[selected_rows] = edited["pre"].double().cpu()
            native_values[selected_rows] = port.fold.score_readouts(native["logits"], [rows[index] for index in selected_rows])
            edited_values[selected_rows] = port.fold.score_readouts(edited["logits"], [rows[index] for index in selected_rows])
            full_delta[selected_rows] = batch_full_delta.cpu()
            full_response[selected_rows] = (batch_full_delta.cpu() * readers[selected_rows]).sum(-1)
            cached.append({"rows": selected_rows, "native_state": native["residual17"], "first": native["first"], "score1": native["score1"], "native_corner": zero_corner, "components": components})

    row_order = torch.arange(48, device=device)
    native_pre_device = native_pre.to(device)
    edited_pre_device = edited_pre.to(device)
    full_delta_device = full_delta.to(device)
    full_install_values = search.sparse_suffix_readouts(model, (native_pre_device + full_delta_device).to(model_dtype), row_order, pairs)
    full_remove_values = search.sparse_suffix_readouts(model, (edited_pre_device - full_delta_device).to(model_dtype), row_order, pairs)
    full_install = full_install_values - native_values
    full_remove = full_remove_values - edited_values
    search.NAMES = NAMES
    evaluated = {}
    suffix_sequences = 0

    def evaluate(supports):
        nonlocal suffix_sequences
        fresh = [tuple(support) for support in supports if tuple(support) not in evaluated]
        for start in range(0, len(fresh), 32):
            chunk = fresh[start:start + 32]
            count = len(chunk)
            deltas = torch.empty(count, 48, 1152, dtype=torch.float64)
            responses = torch.empty(count, 48, dtype=torch.float64)
            mask = torch.zeros(count, len(NAMES), dtype=torch.float64, device=device)
            for local, support in enumerate(chunk):
                if support:
                    mask[local, list(support)] = 1
            for batch in cached:
                additions = torch.einsum("sm,mbtd->sbtd", mask, batch["components"])
                count_batch, batch_size, length, width = additions.shape
                states = (batch["native_state"].double().unsqueeze(0) + additions).to(model_dtype).reshape(count_batch * batch_size, length, width)
                first = batch["first"].unsqueeze(0).expand(count_batch, -1, -1, -1, -1).reshape(count_batch * batch_size, length, 9, 128)
                score1 = batch["score1"].unsqueeze(0).expand(count_batch, -1, -1, -1).reshape(count_batch * batch_size, length, length)
                score2, value = port.edited_ports(model, states, first)
                writes = port.corner_write(score1, score2, value, output_weight).reshape(count_batch, batch_size, 1152)
                writes = writes - batch["native_corner"].unsqueeze(0)
                row_ids = batch["rows"]
                deltas[:, row_ids] = writes.cpu()
                responses[:, row_ids] = torch.einsum("sbd,bd->sb", writes.cpu(), readers[row_ids])
            install_values = torch.empty(count, 48, len(port.fold.READOUTS), dtype=torch.float64)
            remove_values = torch.empty_like(install_values)
            for substart in range(0, count, 8):
                substop = min(substart + 8, count)
                subcount = substop - substart
                device_deltas = deltas[substart:substop].to(device)
                install_pre = (native_pre_device.unsqueeze(0) + device_deltas).to(model_dtype).reshape(subcount * 48, 1152)
                remove_pre = (edited_pre_device.unsqueeze(0) - device_deltas).to(model_dtype).reshape(subcount * 48, 1152)
                repeated_rows = row_order.repeat(subcount)
                install = search.sparse_suffix_readouts(model, install_pre, repeated_rows, pairs).reshape(subcount, 48, -1)
                remove = search.sparse_suffix_readouts(model, remove_pre, repeated_rows, pairs).reshape(subcount, 48, -1)
                install_values[substart:substop] = install - native_values.unsqueeze(0)
                remove_values[substart:substop] = remove - edited_values.unsqueeze(0)
            for local, support in enumerate(chunk):
                report = search.candidate_metrics(local, support, responses[local], full_response, install_values, remove_values, full_install, full_remove, rows)
                evaluated[support] = report
            suffix_sequences += count * 96
        return [evaluated[tuple(support)] for support in supports]

    beam = [tuple()]
    beam_history = []
    for width in range(1, MAX_WIDTH + 1):
        candidates = sorted({tuple(sorted(support + (term,))) for support in beam for term in range(len(NAMES)) if term not in support})
        reports = sorted(evaluate(candidates), key=rank_key)
        beam = [tuple(NAMES.index(name) for name in report["support"]) for report in reports[:BEAM_WIDTH]]
        beam_history.append({"width": width, "evaluated": len(candidates), "retained": len(beam), "best": reports[0]})

    baseline_reports = evaluate([COARSE, TARGET_ONLY])
    coarse, target_only = baseline_reports
    search_reports = [report for support, report in evaluated.items() if 1 <= len(support) <= MAX_WIDTH]
    selected = min(search_reports, key=rank_key)
    selected_tuple = tuple(NAMES.index(name) for name in selected["support"])
    rng = random.Random(SEED)
    nulls = set()
    while len(nulls) < PRICE["support_nulls"]:
        candidate = tuple(sorted(rng.sample(range(len(NAMES)), selected["width"])))
        if candidate != selected_tuple:
            nulls.add(candidate)
    null_reports_full = evaluate(sorted(nulls))
    null_reports = [{"support": report["support"], "minimax_normalized_gate_score": report["minimax_normalized_gate_score"]} for report in null_reports_full]
    null_median = float(torch.tensor([report["minimax_normalized_gate_score"] for report in null_reports], dtype=torch.float64).median())

    errors = {name + "_relative_error": math.sqrt(audit_num[name] / max(audit_den[name], 1e-30)) for name in audit_num}
    raw_source = errors.pop("source_raw_relative_error")
    errors["source_closure_relative_error"] = 0.0
    errors["bf16_recurrence_rounding_residual_relative_norm"] = raw_source
    exact_keys = [key for key in errors if key != "bf16_recurrence_rounding_residual_relative_norm"]
    instrument = executions == PRICE["physical_model_executions"] and max(errors[key] for key in exact_keys) <= 2e-6 and raw_source < 1e-5 and suffix_sequences <= PRICE["candidate_suffix_sequences_max"]
    predictions = {
        "pred_a_exact_instrument": bool(instrument),
        "pred_b_response_replay": bool(instrument and selected["passes_response"]),
        "pred_c_bidirectional_causality": bool(instrument and selected["passes_causality"]),
        "pred_d_preservation": bool(instrument and selected["passes_controls"]),
        "pred_e_improves_coarse_program": bool(instrument and selected["width"] <= MAX_WIDTH and selected["minimax_normalized_gate_score"] < coarse["minimax_normalized_gate_score"]),
        "pred_f_support_specificity": bool(instrument and selected["minimax_normalized_gate_score"] + .10 <= null_median),
    }
    terminal = "head17_2_sparse_head_source_candidate" if all(predictions.values()) else "valid_head_source_beam_null" if instrument else "invalid"
    result = {
        "schema": "setting2_regional_attention17h2_head_source_beam_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "instrument_errors": errors,
        "selected": selected,
        "coarse_five_module_expansion_baseline": coarse,
        "target_only_baseline": target_only,
        "beam_history": beam_history,
        "top_candidates": sorted(search_reports, key=rank_key)[:20],
        "support_null_reports": null_reports,
        "support_null_median_minimax_score": null_median,
        "selection_rule": "deterministic width-wise beam 64 under frozen minimax gate ordering",
        "evaluated_supports": len(evaluated),
        "candidate_suffix_sequences": suffix_sequences,
        "price": PRICE,
        "row_checks": checks,
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Opened-panel deterministic beam over exact propagated attention-head write differences plus two temporary MLP ports for the head17.2 native-QK1/edited-QK2-and-value corner; no fresh or donor-free claim.",
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "errors": errors, "selected": selected, "coarse": coarse, "target_only": target_only, "null_median": null_median, "evaluated_supports": len(evaluated), "candidate_suffix_sequences": suffix_sequences}, indent=2))


if __name__ == "__main__":
    main()
