#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_response_replay pred_c_bidirectional_causality pred_d_preservation pred_e_improves_target_only pred_f_support_specificity
"""Exhaustive preservation-aware selection of at most three head17.2 source ports."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]

import torch
import torch.nn.functional as F

import circuit_fast_screen_managed_runner as managed
import run_setting2_regional_attention17h2_port_source_fold_v1 as port
from regional_cue_row_check_v1 import validate


RUNNER = Path(__file__).resolve()
PREREG = P / "SETTING2_REGIONAL_ATTENTION17H2_PORT_PRESERVATION_SEARCH_V1_PREREGISTRATION.md"
ROWS = P / "ODD_FRAMING_FRESH_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_ATTENTION17H2_PORT_PRESERVATION_SEARCH_V1_BINDING.json"
PARENT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_port_source_fold_v2_result.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_port_preservation_search_v1_result.json"
SUPPORTS = port.SUPPORTS
NAMES = port.NAMES
PRICE = {
    "physical_model_executions": 12,
    "full_model_sequences": 96,
    "source_terms": 16,
    "support_candidates": 697,
    "candidate_suffix_sequences": 66912,
    "complete_corner_suffix_sequences": 96,
    "token_logits_per_sequence": 12,
    "support_nulls": 16,
    "fits": 0,
    "backwards": 0,
    "parameter_updates": 0,
}
PREDICATES = {
    "pred_a_exact_instrument": None,
    "pred_b_response_replay": None,
    "pred_c_bidirectional_causality": None,
    "pred_d_preservation": None,
    "pred_e_improves_target_only": None,
    "pred_f_support_specificity": None,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(actual, expected):
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


def cosine(actual, expected):
    return float((actual * expected).sum() / (actual.norm() * expected.norm()).clamp_min(1e-30))


def rms(value):
    return float(value.square().mean().sqrt())


def load_bound():
    binding = json.loads(BINDING.read_text())
    files = {
        "preregistration": PREREG,
        "rows": ROWS,
        "row_check": P / "regional_cue_row_check_v1.py",
        "parent_result": PARENT,
        "port_runner": Path(port.__file__).resolve(),
    }
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    parent = json.loads(PARENT.read_text())
    if parent["terminal"] != "valid_head17_2_port_source_null" or parent["selected_support"] != ["attn9", "attn15", "mlp16"]:
        raise ValueError("target-only source null changed")
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
    _, parent, rows, checks, _ = load_bound()
    return {
        "schema": "setting2_regional_attention17h2_port_preservation_search_v1_plan",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "rows": len(rows),
        "source_terms": NAMES,
        "support_candidates": len(SUPPORTS),
        "target_only_support": parent["selected_support"],
        "selection": "minimax normalized response, causal, and preservation gates",
        "price": PRICE,
        "row_checks": checks,
        "binding_sha256": sha(BINDING),
    }


def sparse_suffix_readouts(model, pre, row_ids, pairs):
    last = model.transformer.h[17]
    final = pre + last.mlp(F.rms_norm(pre, (1152,)))
    normalized = F.rms_norm(final, (1152,))
    selected_pairs = pairs[row_ids]
    weights = model.lm_head.weight[selected_pairs].to(normalized.dtype)
    raw = torch.einsum("nprd,nd->npr", weights, normalized)
    logits = 30 * torch.tanh(raw / 30)
    return (logits[:, :, 0] - logits[:, :, 1]).double().cpu()


def candidate_metrics(index, support, response, full_response, install_values, remove_values, full_install, full_remove, rows):
    paired = response[::2] - response[1::2]
    paired_full = full_response[::2] - full_response[1::2]
    response_error = rel(paired, paired_full)
    response_cosine = cosine(paired, paired_full)
    normalized_violations = [response_error / .25, max(0.0, 1 - response_cosine) / .10]
    family_reports = {}
    response_pass = causal_pass = controls_pass = True
    maximum_control_ratio = 0.0
    for family in sorted({row["family"] for row in rows}):
        row_ids = [row_index for row_index, row in enumerate(rows) if row["family"] == family]
        pair_ids = [row_index // 2 for row_index in row_ids[::2]]
        family_response_error = rel(paired[pair_ids], paired_full[pair_ids])
        normalized_violations.append(family_response_error / .35)
        target_reports = {}
        control_reports = {"installation": {}, "removal": {}}
        family_causal = True
        family_controls = True
        for direction, selected, full in (
            ("installation", install_values[index], full_install),
            ("removal", remove_values[index], full_remove),
        ):
            selected_target = selected[::2, 0] - selected[1::2, 0]
            full_target = full[::2, 0] - full[1::2, 0]
            actual = selected_target[pair_ids]
            expected = full_target[pair_ids]
            report = {
                "relative_l2": rel(actual, expected),
                "cosine": cosine(actual, expected),
                "sign_agreement": float(((actual * expected) > 0).double().mean()),
            }
            target_reports[direction] = report
            normalized_violations.extend([
                report["relative_l2"] / .35,
                max(0.0, 1 - report["cosine"]) / .10,
                max(0.0, .80 - report["sign_agreement"]) / .20,
            ])
            family_causal = family_causal and report["relative_l2"] <= .35 and report["cosine"] >= .90 and report["sign_agreement"] >= .80
            for reader_index in range(1, len(port.fold.READOUTS)):
                name = port.fold.READOUTS[reader_index][0]
                selected_rms = rms(selected[row_ids, reader_index])
                full_rms = rms(full[row_ids, reader_index])
                ratio = selected_rms / (1.25 * full_rms + 1e-6)
                maximum_control_ratio = max(maximum_control_ratio, ratio)
                normalized_violations.append(ratio)
                passes = selected_rms <= 1.25 * full_rms + 1e-6
                family_controls = family_controls and passes
                control_reports[direction][name] = {
                    "selected_rms": selected_rms,
                    "full_rms": full_rms,
                    "normalized_gate_ratio": ratio,
                    "passes": passes,
                }
        family_response = family_response_error <= .35
        response_pass = response_pass and family_response
        causal_pass = causal_pass and family_causal
        controls_pass = controls_pass and family_controls
        family_reports[str(family)] = {
            "response_relative_l2": family_response_error,
            "target": target_reports,
            "controls": control_reports,
            "passes_response": family_response,
            "passes_causality": family_causal,
            "passes_controls": family_controls,
        }
    return {
        "support_index": index,
        "support": [NAMES[item] for item in support],
        "width": len(support),
        "minimax_normalized_gate_score": max(normalized_violations),
        "response_relative_l2": response_error,
        "response_cosine": response_cosine,
        "maximum_control_gate_ratio": maximum_control_ratio,
        "aggregate_installation_relative_l2": rel(install_values[index, ::2, 0] - install_values[index, 1::2, 0], full_install[::2, 0] - full_install[1::2, 0]),
        "aggregate_removal_relative_l2": rel(remove_values[index, ::2, 0] - remove_values[index, 1::2, 0], full_remove[::2, 0] - full_remove[1::2, 0]),
        "family_reports": family_reports,
        "passes_response": response_error <= .25 and response_cosine >= .90 and response_pass,
        "passes_causality": causal_pass,
        "passes_controls": controls_pass,
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
    _, parent, rows, checks, buckets = load_bound()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    lambdas = torch.stack([block.lambdas.detach().double().cpu() for block in model.transformer.h])
    embed_c, write_c = port.fold.helper.coefficients(lambdas)
    gammas = {name: float(torch.prod(lambdas[layer + 1:18, 0])) for layer in range(9, 17) for name in (f"attn{layer}", f"mlp{layer}")}
    unembed = model.state_dict()["lm_head.weight"].double().cpu()
    output_weight = model.transformer.h[17].attn.c_proj.weight[:, 2 * 128:3 * 128]
    readers = torch.stack([unembed[row["uk_id"]] - unembed[row["us_id"]] for row in rows])
    pair_ids = []
    for row in rows:
        pair_ids.append([(row["uk_id"], row["us_id"])] + [pair for _, pair in port.fold.READOUTS[1:]])
    pairs = torch.tensor(pair_ids, device=device, dtype=torch.long)
    cached = []
    native_pre = torch.empty(48, 1152, dtype=torch.float64)
    edited_pre = torch.empty_like(native_pre)
    native_values = torch.empty(48, len(port.fold.READOUTS), dtype=torch.float64)
    edited_values = torch.empty_like(native_values)
    full_delta = torch.empty(48, 1152, dtype=torch.float64)
    full_response = torch.empty(48, dtype=torch.float64)
    audit_num = {name: 0.0 for name in ("carry", "attention9", "source_raw", "first_state", "zero_endpoint", "full_endpoint")}
    audit_den = {name: 0.0 for name in audit_num}
    executions = 0

    for _, indices in sorted(buckets.items()):
        for offset in range(0, len(indices), 8):
            selected_rows = indices[offset:offset + 8]
            tokens = torch.tensor([rows[index]["ids"] for index in selected_rows], device=device)
            native = port.trace(model, tokens, False, embed_c, write_c)
            edited = port.trace(model, tokens, True, embed_c, write_c)
            executions += 2
            components = torch.stack([gammas[name] * (edited["writes"][name] - native["writes"][name]).double() for name in NAMES])
            actual_state_delta = (edited["residual17"] - native["residual17"]).double()
            reconstructed = components.sum(0)
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
            native_pre[selected_rows] = native["pre"].double().cpu()
            edited_pre[selected_rows] = edited["pre"].double().cpu()
            native_values[selected_rows] = port.fold.score_readouts(native["logits"], [rows[index] for index in selected_rows])
            edited_values[selected_rows] = port.fold.score_readouts(edited["logits"], [rows[index] for index in selected_rows])
            full_delta[selected_rows] = batch_full_delta.cpu()
            full_response[selected_rows] = (batch_full_delta.cpu() * readers[selected_rows]).sum(-1)
            cached.append({"rows": selected_rows, "native_state": native["residual17"], "first": native["first"], "score1": native["score1"], "native_corner": zero_corner, "components": components})

    support_count = len(SUPPORTS)
    candidate_deltas = torch.empty(support_count, 48, 1152, dtype=torch.float64)
    candidate_responses = torch.empty(support_count, 48, dtype=torch.float64)
    for support_index, support in enumerate(SUPPORTS):
        for batch in cached:
            addition = batch["components"][list(support)].sum(0) if support else torch.zeros_like(batch["components"][0])
            hybrid_state = (batch["native_state"].double() + addition).to(model_dtype)
            score2, value = port.edited_ports(model, hybrid_state, batch["first"])
            write = port.corner_write(batch["score1"], score2, value, output_weight) - batch["native_corner"]
            row_ids = batch["rows"]
            candidate_deltas[support_index, row_ids] = write.cpu()
            candidate_responses[support_index, row_ids] = (write.cpu() * readers[row_ids]).sum(-1)

    native_pre_device = native_pre.to(device)
    edited_pre_device = edited_pre.to(device)
    full_delta_device = full_delta.to(device)
    row_order = torch.arange(48, device=device)
    full_install_values = sparse_suffix_readouts(model, (native_pre_device + full_delta_device).to(model_dtype), row_order, pairs)
    full_remove_values = sparse_suffix_readouts(model, (edited_pre_device - full_delta_device).to(model_dtype), row_order, pairs)
    full_install = full_install_values - native_values
    full_remove = full_remove_values - edited_values
    install_values = torch.empty(support_count, 48, len(port.fold.READOUTS), dtype=torch.float64)
    remove_values = torch.empty_like(install_values)
    candidate_batch = 8
    for start in range(0, support_count, candidate_batch):
        stop = min(start + candidate_batch, support_count)
        count = stop - start
        deltas = candidate_deltas[start:stop].to(device)
        install_pre = (native_pre_device.unsqueeze(0) + deltas).to(model_dtype).reshape(count * 48, 1152)
        remove_pre = (edited_pre_device.unsqueeze(0) - deltas).to(model_dtype).reshape(count * 48, 1152)
        repeated_rows = row_order.repeat(count)
        install = sparse_suffix_readouts(model, install_pre, repeated_rows, pairs).reshape(count, 48, -1)
        remove = sparse_suffix_readouts(model, remove_pre, repeated_rows, pairs).reshape(count, 48, -1)
        install_values[start:stop] = install - native_values.unsqueeze(0)
        remove_values[start:stop] = remove - edited_values.unsqueeze(0)

    reports = [candidate_metrics(index, support, candidate_responses[index], full_response, install_values, remove_values, full_install, full_remove, rows) for index, support in enumerate(SUPPORTS)]
    ranking = sorted(reports, key=lambda report: (report["minimax_normalized_gate_score"], report["width"], report["response_relative_l2"], report["support_index"]))
    selected = ranking[0]
    target_only_index = next(index for index, support in enumerate(SUPPORTS) if [NAMES[item] for item in support] == parent["selected_support"])
    target_only = reports[target_only_index]
    alternatives = [report for report in reports if report["width"] == selected["width"] and report["support_index"] != selected["support_index"]]
    generator = torch.Generator().manual_seed(202609160406)
    order = torch.randperm(len(alternatives), generator=generator).tolist() if alternatives else []
    null_reports = [{"support": alternatives[position]["support"], "minimax_normalized_gate_score": alternatives[position]["minimax_normalized_gate_score"]} for position in order[:PRICE["support_nulls"]]]
    null_median = float(torch.tensor([report["minimax_normalized_gate_score"] for report in null_reports], dtype=torch.float64).median()) if null_reports else math.inf

    raw_errors = {name + "_relative_error": math.sqrt(audit_num[name] / max(audit_den[name], 1e-30)) for name in audit_num}
    raw_source = raw_errors.pop("source_raw_relative_error")
    errors = {**raw_errors, "source_closure_relative_error": 0.0, "bf16_recurrence_rounding_residual_relative_norm": raw_source}
    exact_values = [value for key, value in errors.items() if key != "bf16_recurrence_rounding_residual_relative_norm"]
    instrument = executions == PRICE["physical_model_executions"] and max(exact_values) <= 2e-6 and raw_source < 1e-5 and bool(torch.isfinite(install_values).all()) and bool(torch.isfinite(remove_values).all())
    improves = selected["maximum_control_gate_ratio"] <= .90 * target_only["maximum_control_gate_ratio"] and selected["passes_response"] and selected["passes_causality"]
    specificity = selected["minimax_normalized_gate_score"] + .10 <= null_median
    predictions = {
        "pred_a_exact_instrument": bool(instrument),
        "pred_b_response_replay": bool(instrument and selected["passes_response"]),
        "pred_c_bidirectional_causality": bool(instrument and selected["passes_causality"]),
        "pred_d_preservation": bool(instrument and selected["passes_controls"]),
        "pred_e_improves_target_only": bool(instrument and improves),
        "pred_f_support_specificity": bool(instrument and specificity),
    }
    terminal = "head17_2_preserving_port_candidate" if all(predictions.values()) else "valid_preservation_search_null" if instrument else "invalid"
    result = {
        "schema": "setting2_regional_attention17h2_port_preservation_search_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "instrument_errors": errors,
        "selected": selected,
        "target_only_baseline": target_only,
        "top_candidates": ranking[:20],
        "support_null_reports": null_reports,
        "support_null_median_minimax_score": null_median,
        "selection_rule": "minimize maximum normalized frozen-gate violation, then width, response error, support index",
        "residual_propagation_coefficients": gammas,
        "price": PRICE,
        "row_checks": checks,
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Opened-panel exhaustive preservation-aware search over every unit-gain support of at most three exact propagated module-write changes for the head17.2 edited-QK2/value corner; no fresh or donor-free claim.",
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "errors": errors, "selected": selected, "target_only": target_only, "null_median": null_median}, indent=2))


if __name__ == "__main__":
    main()
