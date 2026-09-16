#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_sparse_port_response pred_c_causal_installation pred_d_causal_removal pred_e_control_nonworsening pred_f_support_specificity
"""Sparse source fold for the edited QK2/value ports of head17.2."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
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
import run_setting2_regional_attention17h2_factor_interaction_fold_v1 as fold
from regional_cue_row_check_v1 import validate


RUNNER = Path(__file__).resolve()
PREREG = P / "SETTING2_REGIONAL_ATTENTION17H2_PORT_SOURCE_FOLD_V1_PREREGISTRATION.md"
ROWS = P / "ODD_FRAMING_FRESH_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_ATTENTION17H2_PORT_SOURCE_FOLD_V1_BINDING.json"
PARENT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_factor_corner_fresh_v2_result.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_port_source_fold_v1_result.json"
NAMES = tuple(name for layer in range(9, 17) for name in (f"attn{layer}", f"mlp{layer}"))
SUPPORTS = tuple(support for width in range(4) for support in itertools.combinations(range(len(NAMES)), width))
PREDICATES = {
    "pred_a_exact_instrument": None,
    "pred_b_sparse_port_response": None,
    "pred_c_causal_installation": None,
    "pred_d_causal_removal": None,
    "pred_e_control_nonworsening": None,
    "pred_f_support_specificity": None,
}
PRICE = {
    "physical_model_executions": 12,
    "full_model_sequences": 96,
    "source_terms": 16,
    "support_candidates": 697,
    "support_nulls": 16,
    "additional_suffix_evaluations": 4,
    "additional_suffix_sequences": 192,
    "fits": 0,
    "backwards": 0,
    "parameter_updates": 0,
}
EPS = torch.finfo(torch.float32).eps


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
        "factor_runner": Path(fold.__file__).resolve(),
    }
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    parent = json.loads(PARENT.read_text())
    if parent["terminal"] != "fresh_head17_2_factor_corner" or not all(parent["predictions"].values()):
        raise ValueError("fresh factor-corner authority changed")
    rows = json.loads(ROWS.read_text())["rows"]
    checks = validate(rows)
    buckets = {}
    for index, row in enumerate(rows):
        buckets.setdefault(len(row["ids"]), []).append(index)
    executions = 2 * sum(math.ceil(len(indices) / 8) for indices in buckets.values())
    if len(rows) != 48 or executions != PRICE["physical_model_executions"] or len(SUPPORTS) != PRICE["support_candidates"]:
        raise ValueError("row, support, or execution price changed")
    return binding, rows, checks, buckets


def plan():
    _, rows, checks, _ = load_bound()
    return {
        "schema": "setting2_regional_attention17h2_port_source_fold_v1_plan",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "rows": len(rows),
        "source_terms": NAMES,
        "support_candidates": len(SUPPORTS),
        "price": PRICE,
        "row_checks": checks,
        "binding_sha256": sha(BINDING),
    }


def head_factors(model, residual, first, independent=False):
    at = model.transformer.h[17].attn
    norm = F.rms_norm(residual.to(dtype=next(model.parameters()).dtype), (1152,))
    batch, length, _ = norm.shape
    qraw = at.c_q(norm).view(batch, length, 9, 128)
    kraw = at.c_k(norm).view_as(qraw)
    q2raw = at.c_q2(norm).view_as(qraw)
    k2raw = at.c_k2(norm).view_as(qraw)
    value = at.c_v(norm).view_as(qraw)
    value = (1 - at.lamb) * value + at.lamb * first.view_as(value)
    cos, sin = at.rotary(qraw)
    q = fold.apply_rotary(F.rms_norm(qraw, (128,)), cos, sin)
    k = fold.apply_rotary(F.rms_norm(kraw, (128,)), cos, sin)
    q2 = fold.apply_rotary(F.rms_norm(q2raw, (128,)), cos, sin)
    k2 = fold.apply_rotary(F.rms_norm(k2raw, (128,)), cos, sin)
    score1 = torch.einsum("bqd,bkd->bqk", q[:, :, 2], k[:, :, 2]) / 128
    score2 = torch.einsum("bqd,bkd->bqk", q2[:, :, 2], k2[:, :, 2]) / 128
    independent_head2 = fold.projected_head_writes(q, k, q2, k2, value, at.c_proj.weight)[:, 2, -1] if independent else None
    return score1, score2, value[:, :, 2], independent_head2


def edited_ports(model, residual, first):
    """Compute only head17.2 QK2 and value for exhaustive support search."""
    at = model.transformer.h[17].attn
    norm = F.rms_norm(residual.to(dtype=next(model.parameters()).dtype), (1152,))
    batch, length, _ = norm.shape
    rows = slice(2 * 128, 3 * 128)
    q2raw = F.linear(norm, at.c_q2.weight[rows].to(norm.dtype))
    k2raw = F.linear(norm, at.c_k2.weight[rows].to(norm.dtype))
    value = F.linear(norm, at.c_v.weight[rows].to(norm.dtype))
    value = (1 - at.lamb) * value + at.lamb * first.view(batch, length, 9, 128)[:, :, 2]
    cos, sin = at.rotary(q2raw)
    q2 = fold.apply_rotary(F.rms_norm(q2raw, (128,)), cos, sin)
    k2 = fold.apply_rotary(F.rms_norm(k2raw, (128,)), cos, sin)
    score2 = torch.einsum("bqd,bkd->bqk", q2, k2) / 128
    return score2, value


def corner_write(score1, score2, value, output_weight):
    # The final query can attend to every key, so no causal mask is needed here.
    routed = torch.einsum("bk,bkd->bd", score1[:, -1].double() * score2[:, -1].double(), value.double())
    return routed @ output_weight.double().T


@torch.no_grad()
def trace(model, tokens, edit, embed_c, write_c):
    batch, length = tokens.shape
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    first = None
    early = {}
    for layer, block in enumerate(model.transformer.h[:8]):
        residual = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, first = block.attn(F.rms_norm(residual, (1152,)), first)
        mlp = block.mlp(F.rms_norm(residual + attention, (1152,)))
        early[f"attn{layer}"] = attention
        early[f"mlp{layer}"] = mlp
        x = residual + attention + mlp

    block8, block9 = model.transformer.h[8], model.transformer.h[9]
    residual8 = block8.lambdas[0] * x + block8.lambdas[1] * x0
    attention8, first = block8.attn(F.rms_norm(residual8, (1152,)), first)
    mlp8 = block8.mlp(F.rms_norm(residual8 + attention8, (1152,)))
    x = residual8 + attention8 + mlp8

    residual9 = block9.lambdas[0] * x + block9.lambdas[1] * x0
    norm9 = F.rms_norm(residual9, (1152,))
    attention9, first_out = block9.attn(norm9, first)
    attention9_native = attention9
    at9 = block9.attn
    parts = [embed_c * x0] + [write_c[name] * early[name] for name in fold.helper.SOURCE_NAMES[1:]]
    carry = sum(parts)
    carry_native = block9.lambdas[0] * residual8 + block9.lambdas[1] * x0
    denominator = (residual9.square().mean(-1, keepdim=True) + EPS).sqrt()
    normalized = [part / denominator for part in parts]
    late_ids = {fold.helper.SOURCE_NAMES.index(name) for name in fold.helper.LATE_NAMES}
    late = sum(normalized[index] for index in range(17) if index in late_ids)
    remainder = sum(normalized[index] for index in range(17) if index not in late_ids)
    qraw = at9.c_q(norm9).view(batch, length, 9, 128)
    kraw = at9.c_k(norm9).view_as(qraw)
    q2raw = at9.c_q2(norm9).view_as(qraw)
    k2raw = at9.c_k2(norm9).view_as(qraw)
    value9 = at9.c_v(norm9).view_as(qraw)
    value9 = (1 - at9.lamb) * value9 + at9.lamb * first.view_as(value9)
    cos9, sin9 = at9.rotary(qraw)
    q9 = fold.helper.rotate(F.rms_norm(qraw, (128,)), cos9, sin9)
    k9 = fold.helper.rotate(F.rms_norm(kraw, (128,)), cos9, sin9)
    q29 = fold.helper.rotate(F.rms_norm(q2raw, (128,)), cos9, sin9)
    k29 = fold.helper.rotate(F.rms_norm(k2raw, (128,)), cos9, sin9)
    groups = (late, remainder)
    qgroups = [fold.helper.rotate(z, cos9, sin9) for z in fold.helper.group_project(groups, qraw[:, :, 8], at9.c_q.weight[8 * 128:9 * 128])]
    kgroups = [fold.helper.rotate(z, cos9, sin9) for z in fold.helper.group_project(groups, kraw[:, :, 8], at9.c_k.weight[8 * 128:9 * 128])]
    score1 = torch.einsum("bqhd,bkhd->bhqk", q9, k9) / 128
    score2 = torch.einsum("bqhd,bkhd->bhqk", q29, k29) / 128
    mask = torch.tril(torch.ones(length, length, device=tokens.device, dtype=torch.bool))
    pattern = (score1 * score2).masked_fill(~mask, 0)
    zall = torch.einsum("bhqk,bkhd->bhqd", pattern, value9)
    manual9 = at9.c_proj(zall.transpose(1, 2).contiguous().view_as(norm9))
    qgroups_terms = [torch.einsum("bqd,bkd->bqk", qgroups[u], kgroups[v]) / 128 for u, v in ((0, 0), (0, 1), (1, 0))]
    selected = sum(qgroups_terms).masked_fill(~mask, 0)
    zedit = torch.einsum("bqk,bkd->bqd", selected * score2[:, 8], value9[:, :, 8])
    if edit:
        attention9 = attention9 - zedit @ at9.c_proj.weight[:, 8 * 128:9 * 128].T

    writes = {"attn9": attention9}
    x = residual9 + attention9
    mlp9 = block9.mlp(F.rms_norm(x, (1152,)))
    writes["mlp9"] = mlp9
    x = x + mlp9
    first = first_out
    for layer in range(10, 17):
        block = model.transformer.h[layer]
        residual = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, first = block.attn(F.rms_norm(residual, (1152,)), first)
        mlp = block.mlp(F.rms_norm(residual + attention, (1152,)))
        writes[f"attn{layer}"] = attention
        writes[f"mlp{layer}"] = mlp
        x = residual + attention + mlp

    last = model.transformer.h[17]
    residual17 = last.lambdas[0] * x + last.lambdas[1] * x0
    norm17 = F.rms_norm(residual17, (1152,))
    attention17, _ = last.attn(norm17, first)
    pre = (residual17 + attention17)[:, -1]
    final = pre + last.mlp(F.rms_norm(pre, (1152,)))
    logits = 30 * torch.tanh(model.lm_head(F.rms_norm(final, (1152,))) / 30)
    native_score1, native_score2, native_value, head2 = head_factors(model, residual17, first, independent=True)
    return {
        "residual17": residual17,
        "first": first,
        "writes": writes,
        "pre": pre,
        "logits": logits,
        "score1": native_score1,
        "score2": native_score2,
        "value": native_value,
        "head2": head2,
        "carry_num": float((carry - carry_native).square().sum()),
        "carry_den": float(carry_native.square().sum()),
        "attention9_num": float((manual9 - attention9_native).square().sum()),
        "attention9_den": float(attention9_native.square().sum()),
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
    _, rows, checks, buckets = load_bound()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    lambdas = torch.stack([block.lambdas.detach().double().cpu() for block in model.transformer.h])
    embed_c, write_c = fold.helper.coefficients(lambdas)
    gammas = {name: float(torch.prod(lambdas[layer + 1:18, 0])) for layer in range(9, 17) for name in (f"attn{layer}", f"mlp{layer}")}
    unembed = model.state_dict()["lm_head.weight"].double().cpu()
    output_weight = model.transformer.h[17].attn.c_proj.weight[:, 2 * 128:3 * 128]
    cached = []
    native_pre = torch.empty(48, 1152, dtype=torch.float64)
    edited_pre = torch.empty_like(native_pre)
    native_logits = torch.empty(48, 50304, dtype=torch.float64)
    edited_logits = torch.empty_like(native_logits)
    full_delta = torch.empty_like(native_pre)
    readers = torch.stack([unembed[row["uk_id"]] - unembed[row["us_id"]] for row in rows])
    audit_num = {name: 0.0 for name in ("carry", "attention9", "source_closure", "first_state", "zero_endpoint", "full_endpoint")}
    audit_den = {name: 0.0 for name in audit_num}
    executions = 0

    for _, indices in sorted(buckets.items()):
        for offset in range(0, len(indices), 8):
            selected_rows = indices[offset:offset + 8]
            tokens = torch.tensor([rows[index]["ids"] for index in selected_rows], device=device)
            native = trace(model, tokens, False, embed_c, write_c)
            edited = trace(model, tokens, True, embed_c, write_c)
            executions += 2
            components = torch.stack([
                gammas[name] * (edited["writes"][name] - native["writes"][name]).double()
                for name in NAMES
            ])
            actual_state_delta = (edited["residual17"] - native["residual17"]).double()
            reconstructed = components.sum(0)
            full_state = (native["residual17"].double() + reconstructed).to(model_dtype)
            full_score2, full_value = edited_ports(model, full_state, native["first"])
            zero_corner = corner_write(native["score1"], native["score2"], native["value"], output_weight)
            complete_corner = corner_write(native["score1"], edited["score2"], edited["value"], output_weight)
            reconstructed_corner = corner_write(native["score1"], full_score2, full_value, output_weight)
            batch_full_delta = complete_corner - zero_corner
            audit_num["carry"] += native["carry_num"]
            audit_den["carry"] += native["carry_den"]
            audit_num["attention9"] += native["attention9_num"]
            audit_den["attention9"] += native["attention9_den"]
            audit_num["source_closure"] += float((reconstructed - actual_state_delta).square().sum())
            audit_den["source_closure"] += float(actual_state_delta.square().sum())
            audit_num["first_state"] += float((native["first"].double() - edited["first"].double()).square().sum())
            audit_den["first_state"] += float(native["first"].double().square().sum())
            audit_num["zero_endpoint"] += float((zero_corner - native["head2"]).square().sum())
            audit_den["zero_endpoint"] += float(native["head2"].square().sum())
            audit_num["full_endpoint"] += float((reconstructed_corner - complete_corner).square().sum())
            audit_den["full_endpoint"] += float(complete_corner.square().sum())
            native_pre[selected_rows] = native["pre"].double().cpu()
            edited_pre[selected_rows] = edited["pre"].double().cpu()
            native_logits[selected_rows] = native["logits"].double().cpu()
            edited_logits[selected_rows] = edited["logits"].double().cpu()
            full_delta[selected_rows] = batch_full_delta.cpu()
            cached.append({
                "rows": selected_rows,
                "native_state": native["residual17"],
                "first": native["first"],
                "score1": native["score1"],
                "native_corner": zero_corner,
                "components": components,
            })

    full_response = (full_delta * readers).sum(-1)
    paired_full = full_response[::2] - full_response[1::2]
    candidate_reports = []
    for support in SUPPORTS:
        response = torch.empty(48, dtype=torch.float64)
        for batch in cached:
            addition = batch["components"][list(support)].sum(0) if support else torch.zeros_like(batch["components"][0])
            hybrid_state = (batch["native_state"].double() + addition).to(model_dtype)
            score2, value = edited_ports(model, hybrid_state, batch["first"])
            write = corner_write(batch["score1"], score2, value, output_weight) - batch["native_corner"]
            row_ids = batch["rows"]
            response[row_ids] = (write.cpu() * readers[row_ids]).sum(-1)
        paired = response[::2] - response[1::2]
        candidate_reports.append((rel(paired, paired_full), len(support), support, cosine(paired, paired_full)))
    candidate_reports.sort(key=lambda item: (item[0], item[1], item[2]))
    response_error, _, support, response_cosine = candidate_reports[0]
    support_names = [NAMES[index] for index in support]

    selected_delta = torch.empty_like(full_delta)
    for batch in cached:
        addition = batch["components"][list(support)].sum(0) if support else torch.zeros_like(batch["components"][0])
        hybrid_state = (batch["native_state"].double() + addition).to(model_dtype)
        score2, value = edited_ports(model, hybrid_state, batch["first"])
        write = corner_write(batch["score1"], score2, value, output_weight) - batch["native_corner"]
        selected_delta[batch["rows"]] = write.cpu()

    paired_selected = (selected_delta * readers).sum(-1)[::2] - (selected_delta * readers).sum(-1)[1::2]
    alternatives = [item for item in candidate_reports if item[1] == len(support) and item[2] != support]
    generator = torch.Generator().manual_seed(202609160349)
    order = torch.randperm(len(alternatives), generator=generator).tolist() if alternatives else []
    null_reports = [
        {"support": [NAMES[index] for index in alternatives[position][2]], "response_relative_l2": alternatives[position][0]}
        for position in order[:PRICE["support_nulls"]]
    ]
    null_median = float(torch.tensor([item["response_relative_l2"] for item in null_reports], dtype=torch.float64).median()) if null_reports else math.inf

    suffix_inputs = (native_pre + full_delta, native_pre + selected_delta, edited_pre - full_delta, edited_pre - selected_delta)
    suffix_logits = [fold.suffix_logits(model, value.to(device=device, dtype=model_dtype)).double().cpu() for value in suffix_inputs]
    values = torch.stack([
        fold.score_readouts(native_logits, rows),
        fold.score_readouts(suffix_logits[0], rows),
        fold.score_readouts(suffix_logits[1], rows),
        fold.score_readouts(edited_logits, rows),
        fold.score_readouts(suffix_logits[2], rows),
        fold.score_readouts(suffix_logits[3], rows),
    ])
    install_effect = values[1:3] - values[0]
    removal_effect = values[4:6] - values[3]
    full_install_all = install_effect[0, ::2, 0] - install_effect[0, 1::2, 0]
    selected_install_all = install_effect[1, ::2, 0] - install_effect[1, 1::2, 0]
    full_remove_all = removal_effect[0, ::2, 0] - removal_effect[0, 1::2, 0]
    selected_remove_all = removal_effect[1, ::2, 0] - removal_effect[1, 1::2, 0]

    response_pass = installation_pass = removal_pass = control_pass = True
    family_reports = {}
    for family in sorted({row["family"] for row in rows}):
        row_ids = [index for index, row in enumerate(rows) if row["family"] == family]
        pair_ids = [index // 2 for index in row_ids[::2]]
        response_family_error = rel(paired_selected[pair_ids], paired_full[pair_ids])
        fi, si = full_install_all[pair_ids], selected_install_all[pair_ids]
        fr, sr = full_remove_all[pair_ids], selected_remove_all[pair_ids]
        installation = {"relative_l2": rel(si, fi), "cosine": cosine(si, fi), "sign_agreement": float(((si * fi) > 0).double().mean())}
        removal = {"relative_l2": rel(sr, fr), "cosine": cosine(sr, fr), "sign_agreement": float(((sr * fr) > 0).double().mean())}
        install_controls = {fold.READOUTS[index][0]: {"full_rms": rms(install_effect[0, row_ids, index]), "selected_rms": rms(install_effect[1, row_ids, index])} for index in range(1, len(fold.READOUTS))}
        removal_controls = {fold.READOUTS[index][0]: {"full_rms": rms(removal_effect[0, row_ids, index]), "selected_rms": rms(removal_effect[1, row_ids, index])} for index in range(1, len(fold.READOUTS))}
        family_response = response_family_error <= .35
        family_install = installation["relative_l2"] <= .35 and installation["cosine"] >= .90 and installation["sign_agreement"] >= .80
        family_remove = removal["relative_l2"] <= .35 and removal["cosine"] >= .90 and removal["sign_agreement"] >= .80
        family_controls = all(item["selected_rms"] <= 1.25 * item["full_rms"] + 1e-6 for item in (*install_controls.values(), *removal_controls.values()))
        response_pass = response_pass and family_response
        installation_pass = installation_pass and family_install
        removal_pass = removal_pass and family_remove
        control_pass = control_pass and family_controls
        family_reports[str(family)] = {
            "response_relative_l2": response_family_error,
            "installation": installation,
            "removal": removal,
            "installation_controls": install_controls,
            "removal_controls": removal_controls,
            "passes_response": family_response,
            "passes_installation": family_install,
            "passes_removal": family_remove,
            "passes_controls": family_controls,
        }

    errors = {name + "_relative_error": math.sqrt(audit_num[name] / max(audit_den[name], 1e-30)) for name in audit_num}
    finite = all(torch.isfinite(value).all() for value in (full_delta, selected_delta, values))
    instrument = executions == PRICE["physical_model_executions"] and max(errors.values()) <= 2e-6 and bool(finite)
    specificity = response_error + .05 <= null_median
    predictions = {
        "pred_a_exact_instrument": bool(instrument),
        "pred_b_sparse_port_response": bool(instrument and len(support) <= 3 and response_error <= .25 and response_cosine >= .90 and response_pass),
        "pred_c_causal_installation": bool(instrument and installation_pass),
        "pred_d_causal_removal": bool(instrument and removal_pass),
        "pred_e_control_nonworsening": bool(instrument and control_pass),
        "pred_f_support_specificity": bool(instrument and specificity),
    }
    terminal = "head17_2_sparse_port_source_candidate" if all(predictions.values()) else "valid_head17_2_port_source_null" if instrument else "invalid"
    result = {
        "schema": "setting2_regional_attention17h2_port_source_fold_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "instrument_errors": errors,
        "selected_support": support_names,
        "selected_support_indices": list(support),
        "response_relative_l2": response_error,
        "response_cosine": response_cosine,
        "aggregate_installation_relative_l2": rel(selected_install_all, full_install_all),
        "aggregate_removal_relative_l2": rel(selected_remove_all, full_remove_all),
        "family_reports": family_reports,
        "support_null_reports": null_reports,
        "support_null_median_response_relative_l2": null_median,
        "top_candidates": [
            {"support": [NAMES[index] for index in item[2]], "response_relative_l2": item[0], "response_cosine": item[3]}
            for item in candidate_reports[:20]
        ],
        "residual_propagation_coefficients": gammas,
        "price": PRICE,
        "row_checks": checks,
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Opened-panel exhaustive unit-gain support fold of exact layer9-16 propagated write changes into the fresh-confirmed head17.2 native-QK1/edited-QK2-and-value corner; selected write deltas remain live ports.",
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "errors": errors, "selected_support": support_names, "response_error": response_error, "response_cosine": response_cosine, "installation_error": result["aggregate_installation_relative_l2"], "removal_error": result["aggregate_removal_relative_l2"], "null_median": null_median, "families": family_reports}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
