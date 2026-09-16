#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_sparse_response_replay pred_c_causal_installation pred_d_causal_removal pred_e_control_nonworsening pred_f_equal_norm_random_null
"""Exact QK1/QK2/value Möbius fold of the induced head17.2 response."""
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
import run_setting2_regional_head9_8_qk1_late_group_fresh_routing_v3 as helper
from regional_cue_row_check_v1 import validate
from regional_grouped_interaction_tools import apply_rotary
from squared_attention_head_tools import projected_head_writes


RUNNER = Path(__file__).resolve()
PREREG = P / "SETTING2_REGIONAL_ATTENTION17H2_FACTOR_INTERACTION_FOLD_V1_PREREGISTRATION.md"
ROWS = P / "ODD_FRAMING_FRESH_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_ATTENTION17H2_FACTOR_INTERACTION_FOLD_V1_BINDING.json"
UPSTREAM = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_head9_8_qk1_late_group_fresh_routing_v3_result.json"
HEAD_FOLD = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17_head_response_fold_v1_result.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_factor_interaction_fold_v1_result.json"

FACTORS = ("qk1", "qk2", "value")
TERMS = ("qk1", "qk2", "qk1_x_qk2", "value", "qk1_x_value", "qk2_x_value", "qk1_x_qk2_x_value")
READOUTS = (
    ("target", None),
    ("work_jobs", (670, 3946)),
    ("cat_dog", (3797, 3290)),
    ("red_blue", (2266, 4171)),
    ("monday_tuesday", (3321, 3431)),
    ("apple_orange", (17180, 10912)),
)
PRICE = {
    "physical_model_executions": 12,
    "full_model_sequences": 96,
    "factor_corners": 8,
    "support_candidates": 64,
    "equal_norm_random_nulls": 8,
    "additional_suffix_evaluations": 20,
    "additional_suffix_sequences": 960,
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


def rms(x):
    return float(x.square().mean().sqrt())


def load_bound():
    binding = json.loads(BINDING.read_text())
    files = {
        "preregistration": PREREG,
        "rows": ROWS,
        "row_check": P / "regional_cue_row_check_v1.py",
        "upstream_result": UPSTREAM,
        "head_fold_result": HEAD_FOLD,
        "intervention_helper": Path(helper.__file__).resolve(),
        "rotary_helper": RUNNER.parent / "regional_grouped_interaction_tools.py",
        "head_helper": RUNNER.parent / "squared_attention_head_tools.py",
    }
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    upstream = json.loads(UPSTREAM.read_text())
    head_fold = json.loads(HEAD_FOLD.read_text())
    if upstream["terminal"] != "valid_null" or not upstream["predictions"]["pred_d_selected_material_and_exceeds_rr"]:
        raise ValueError("upstream edit authority changed")
    if head_fold["terminal"] != "attention17_head_localized" or head_fold["ranking"][0] != 2 or not all(head_fold["predictions"].values()):
        raise ValueError("head17.2 localization authority changed")
    rows = json.loads(ROWS.read_text())["rows"]
    checks = validate(rows)
    buckets = {}
    for index, row in enumerate(rows):
        buckets.setdefault(len(row["ids"]), []).append(index)
    executions = 2 * sum(math.ceil(len(indices) / 8) for indices in buckets.values())
    if len(rows) != 48 or executions != PRICE["physical_model_executions"]:
        raise ValueError("row or execution authority changed")
    return binding, rows, checks, buckets


def plan():
    _, rows, checks, _ = load_bound()
    return {
        "schema": "setting2_regional_attention17h2_factor_interaction_fold_v1_plan",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "rows": len(rows),
        "factors": FACTORS,
        "terms": TERMS,
        "price": PRICE,
        "row_checks": checks,
        "binding_sha256": sha(BINDING),
    }


def head_write(score1, score2, value, output_weight):
    length = score1.shape[-1]
    mask = torch.tril(torch.ones(length, length, device=score1.device, dtype=torch.bool))
    pattern = (score1 * score2).masked_fill(~mask, 0)
    z = torch.einsum("bqk,bkd->bqd", pattern, value)
    return z @ output_weight.T


@torch.no_grad()
def run_to_head17(model, tokens, edit, embed_c, write_c):
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
    parts = [embed_c * x0] + [write_c[name] * early[name] for name in helper.SOURCE_NAMES[1:]]
    carry = sum(parts)
    carry_native = block9.lambdas[0] * residual8 + block9.lambdas[1] * x0
    denominator = (residual9.square().mean(-1, keepdim=True) + EPS).sqrt()
    normalized = [part / denominator for part in parts]
    late_ids = {helper.SOURCE_NAMES.index(name) for name in helper.LATE_NAMES}
    late = sum(normalized[index] for index in range(17) if index in late_ids)
    remainder = sum(normalized[index] for index in range(17) if index not in late_ids)
    qraw = at9.c_q(norm9).view(batch, length, 9, 128)
    kraw = at9.c_k(norm9).view_as(qraw)
    q2raw = at9.c_q2(norm9).view_as(qraw)
    k2raw = at9.c_k2(norm9).view_as(qraw)
    value9 = at9.c_v(norm9).view_as(qraw)
    value9 = (1 - at9.lamb) * value9 + at9.lamb * first.view_as(value9)
    cos9, sin9 = at9.rotary(qraw)
    q9 = helper.rotate(F.rms_norm(qraw, (128,)), cos9, sin9)
    k9 = helper.rotate(F.rms_norm(kraw, (128,)), cos9, sin9)
    q29 = helper.rotate(F.rms_norm(q2raw, (128,)), cos9, sin9)
    k29 = helper.rotate(F.rms_norm(k2raw, (128,)), cos9, sin9)
    groups = (late, remainder)
    qgroups = [helper.rotate(z, cos9, sin9) for z in helper.group_project(groups, qraw[:, :, 8], at9.c_q.weight[8 * 128:9 * 128])]
    kgroups = [helper.rotate(z, cos9, sin9) for z in helper.group_project(groups, kraw[:, :, 8], at9.c_k.weight[8 * 128:9 * 128])]
    score1 = torch.einsum("bqhd,bkhd->bhqk", q9, k9) / 128
    score2 = torch.einsum("bqhd,bkhd->bhqk", q29, k29) / 128
    selected = sum(torch.einsum("bqd,bkd->bqk", qgroups[u], kgroups[v]) / 128 for u, v in ((0, 0), (0, 1), (1, 0)))
    mask = torch.tril(torch.ones(length, length, device=tokens.device, dtype=torch.bool))
    pattern = (score1 * score2).masked_fill(~mask, 0)
    zall = torch.einsum("bhqk,bkhd->bhqd", pattern, value9)
    manual9 = at9.c_proj(zall.transpose(1, 2).contiguous().view_as(norm9))
    if edit:
        selected = selected.masked_fill(~mask, 0)
        zedit = torch.einsum("bqk,bkd->bqd", selected * score2[:, 8], value9[:, :, 8])
        attention9 = attention9 - zedit @ at9.c_proj.weight[:, 8 * 128:9 * 128].T
    x = residual9 + attention9
    x = x + block9.mlp(F.rms_norm(x, (1152,)))
    first = first_out
    for block in model.transformer.h[10:17]:
        residual = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, first = block.attn(F.rms_norm(residual, (1152,)), first)
        x = residual + attention + block.mlp(F.rms_norm(residual + attention, (1152,)))

    last = model.transformer.h[17]
    residual17 = last.lambdas[0] * x + last.lambdas[1] * x0
    norm17 = F.rms_norm(residual17, (1152,))
    first_in = first
    attention17, _ = last.attn(norm17, first)
    at17 = last.attn
    qraw17 = at17.c_q(norm17).view(batch, length, 9, 128)
    kraw17 = at17.c_k(norm17).view_as(qraw17)
    q2raw17 = at17.c_q2(norm17).view_as(qraw17)
    k2raw17 = at17.c_k2(norm17).view_as(qraw17)
    value17 = at17.c_v(norm17).view_as(qraw17)
    value17 = (1 - at17.lamb) * value17 + at17.lamb * first_in.view_as(value17)
    cos17, sin17 = at17.rotary(qraw17)
    q17 = apply_rotary(F.rms_norm(qraw17, (128,)), cos17, sin17)
    k17 = apply_rotary(F.rms_norm(kraw17, (128,)), cos17, sin17)
    q217 = apply_rotary(F.rms_norm(q2raw17, (128,)), cos17, sin17)
    k217 = apply_rotary(F.rms_norm(k2raw17, (128,)), cos17, sin17)
    heads17 = projected_head_writes(q17, k17, q217, k217, value17, at17.c_proj.weight)
    score117 = torch.einsum("bqd,bkd->bqk", q17[:, :, 2], k17[:, :, 2]) / 128
    score217 = torch.einsum("bqd,bkd->bqk", q217[:, :, 2], k217[:, :, 2]) / 128
    output_weight = at17.c_proj.weight[:, 2 * 128:3 * 128]
    projected2 = head_write(score117, score217, value17[:, :, 2], output_weight)
    pre = (residual17 + attention17)[:, -1]
    final = pre + last.mlp(F.rms_norm(pre, (1152,)))
    logits = 30 * torch.tanh(model.lm_head(F.rms_norm(final, (1152,))) / 30)
    return {
        "pre": pre,
        "logits": logits,
        "score1": score117,
        "score2": score217,
        "value": value17[:, :, 2],
        "head2": projected2[:, -1],
        "carry_num": float((carry - carry_native).square().sum()),
        "carry_den": float(carry_native.square().sum()),
        "attention9_num": float((manual9 - attention9_native).square().sum()),
        "attention9_den": float(attention9_native.square().sum()),
        "attention17_num": float((heads17.sum(1) - attention17).square().sum()),
        "attention17_den": float(attention17.square().sum()),
        "head2_num": float((heads17[:, 2] - projected2).square().sum()),
        "head2_den": float(projected2.square().sum()),
    }


def suffix_logits(model, pre):
    last = model.transformer.h[17]
    final = pre + last.mlp(F.rms_norm(pre, (1152,)))
    return 30 * torch.tanh(model.lm_head(F.rms_norm(final, (1152,))) / 30)


def score_readouts(logits, rows):
    result = torch.empty(len(rows), len(READOUTS), dtype=torch.float64)
    logits = logits.double().cpu()
    for row_index, row in enumerate(rows):
        pairs = [(row["uk_id"], row["us_id"])] + [pair for _, pair in READOUTS[1:]]
        for reader, (left, right) in enumerate(pairs):
            result[row_index, reader] = logits[row_index, left] - logits[row_index, right]
    return result


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
    lambdas = torch.stack([block.lambdas.detach().double().cpu() for block in model.transformer.h])
    embed_c, write_c = helper.coefficients(lambdas)
    unembed = model.state_dict()["lm_head.weight"].double().cpu()
    components = torch.empty(7, 48, 1152, dtype=torch.float64)
    full_delta = torch.empty(48, 1152, dtype=torch.float64)
    native_pre = torch.empty(48, 1152, dtype=torch.float64)
    edited_pre = torch.empty_like(native_pre)
    native_logits = torch.empty(48, 50304, dtype=torch.float64)
    edited_logits = torch.empty_like(native_logits)
    carry_num = carry_den = attention9_num = attention9_den = 0.0
    attention17_num = attention17_den = head2_num = head2_den = corner_num = corner_den = closure_num = closure_den = 0.0
    executions = 0
    for _, indices in sorted(buckets.items()):
        for offset in range(0, len(indices), 8):
            selected = indices[offset:offset + 8]
            tokens = torch.tensor([rows[index]["ids"] for index in selected], device=device)
            native = run_to_head17(model, tokens, False, embed_c, write_c)
            edited = run_to_head17(model, tokens, True, embed_c, write_c)
            executions += 2
            for run in (native, edited):
                carry_num += run["carry_num"]
                carry_den += run["carry_den"]
                attention9_num += run["attention9_num"]
                attention9_den += run["attention9_den"]
                attention17_num += run["attention17_num"]
                attention17_den += run["attention17_den"]
                head2_num += run["head2_num"]
                head2_den += run["head2_den"]
            corners = {}
            output_weight = model.transformer.h[17].attn.c_proj.weight[:, 2 * 128:3 * 128]
            for mask in range(8):
                score1 = edited["score1"] if mask & 1 else native["score1"]
                score2 = edited["score2"] if mask & 2 else native["score2"]
                value = edited["value"] if mask & 4 else native["value"]
                corners[mask] = head_write(score1, score2, value, output_weight)[:, -1].double()
            corner_num += float((corners[0] - native["head2"]).square().sum() + (corners[7] - edited["head2"]).square().sum())
            corner_den += float(native["head2"].square().sum() + edited["head2"].square().sum())
            terms = []
            for mask in range(1, 8):
                term = torch.zeros_like(corners[0])
                subset = mask
                while True:
                    sign = -1 if ((mask.bit_count() - subset.bit_count()) % 2) else 1
                    term = term + sign * corners[subset]
                    if subset == 0:
                        break
                    subset = (subset - 1) & mask
                terms.append(term)
            delta = corners[7] - corners[0]
            closure_num += float((sum(terms) - delta).square().sum())
            closure_den += float(delta.square().sum())
            components[:, selected] = torch.stack(terms).cpu()
            full_delta[selected] = delta.cpu()
            native_pre[selected] = native["pre"].double().cpu()
            edited_pre[selected] = edited["pre"].double().cpu()
            native_logits[selected] = native["logits"].double().cpu()
            edited_logits[selected] = edited["logits"].double().cpu()

    readers = torch.stack([unembed[row["uk_id"]] - unembed[row["us_id"]] for row in rows])
    response = torch.einsum("trd,rd->tr", components, readers)
    paired_response = response[:, ::2] - response[:, 1::2]
    full_response = paired_response.sum(0)
    candidates = []
    for width in range(4):
        for support in itertools.combinations(range(7), width):
            predicted = paired_response[list(support)].sum(0) if support else torch.zeros_like(full_response)
            candidates.append((rel(predicted, full_response), width, support, cosine(predicted, full_response) if predicted.norm() else 0.0))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    selected_error, _, support, selected_cosine = candidates[0]
    selected_write = components[list(support)].sum(0) if support else torch.zeros_like(full_delta)
    family_response_errors = {}
    for family in sorted({row["family"] for row in rows}):
        pair_ids = [index // 2 for index in range(0, 48, 2) if rows[index]["family"] == family]
        family_response_errors[str(family)] = rel(paired_response[list(support)][:, pair_ids].sum(0) if support else torch.zeros_like(full_response[pair_ids]), full_response[pair_ids])

    tensors = [native_pre, native_pre + full_delta, native_pre + selected_write, edited_pre, edited_pre - full_delta, edited_pre - selected_write]
    model_dtype = next(model.parameters()).dtype
    logits = [native_logits]
    for tensor in tensors[1:3]:
        logits.append(suffix_logits(model, tensor.to(device=device, dtype=model_dtype)).double().cpu())
    logits.append(edited_logits)
    for tensor in tensors[4:]:
        logits.append(suffix_logits(model, tensor.to(device=device, dtype=model_dtype)).double().cpu())
    values = torch.stack([score_readouts(value, rows) for value in logits])
    install_effect = values[1:3] - values[0]
    removal_effect = values[4:6] - values[3]
    generator = torch.Generator(device="cpu").manual_seed(202609160328)
    output_weight = model.transformer.h[17].attn.c_proj.weight[:, 2 * 128:3 * 128].double().cpu()
    random_values = []
    random_errors = []
    full_install_all = install_effect[0, ::2, 0] - install_effect[0, 1::2, 0]
    full_remove_all = removal_effect[0, ::2, 0] - removal_effect[0, 1::2, 0]
    selected_install_all = install_effect[1, ::2, 0] - install_effect[1, 1::2, 0]
    selected_remove_all = removal_effect[1, ::2, 0] - removal_effect[1, 1::2, 0]
    for null_index in range(PRICE["equal_norm_random_nulls"]):
        coefficients = torch.randn(48, 128, generator=generator, dtype=torch.float64)
        random_write = coefficients @ output_weight.T
        random_write = random_write * (selected_write.norm(dim=-1) / random_write.norm(dim=-1).clamp_min(1e-30))[:, None]
        random_install_logits = suffix_logits(model, (native_pre + random_write).to(device=device, dtype=model_dtype)).double().cpu()
        random_remove_logits = suffix_logits(model, (edited_pre - random_write).to(device=device, dtype=model_dtype)).double().cpu()
        random_install = score_readouts(random_install_logits, rows) - values[0]
        random_remove = score_readouts(random_remove_logits, rows) - values[3]
        random_install_target = random_install[::2, 0] - random_install[1::2, 0]
        random_remove_target = random_remove[::2, 0] - random_remove[1::2, 0]
        entry = {"installation_relative_l2": rel(random_install_target, full_install_all), "removal_relative_l2": rel(random_remove_target, full_remove_all)}
        random_errors.append(entry)
        random_values.append((random_install, random_remove))
    selected_install_error = rel(selected_install_all, full_install_all)
    selected_remove_error = rel(selected_remove_all, full_remove_all)
    random_null_pass = selected_install_error + .10 <= min(item["installation_relative_l2"] for item in random_errors) and selected_remove_error + .10 <= min(item["removal_relative_l2"] for item in random_errors)

    family_reports = {}
    install_pass = removal_pass = control_pass = True
    for family in sorted({row["family"] for row in rows}):
        row_ids = [index for index, row in enumerate(rows) if row["family"] == family]
        pair_ids = [index // 2 for index in row_ids[::2]]
        full_install = install_effect[0, ::2, 0] - install_effect[0, 1::2, 0]
        selected_install = install_effect[1, ::2, 0] - install_effect[1, 1::2, 0]
        full_remove = removal_effect[0, ::2, 0] - removal_effect[0, 1::2, 0]
        selected_remove = removal_effect[1, ::2, 0] - removal_effect[1, 1::2, 0]
        fi, si = full_install[pair_ids], selected_install[pair_ids]
        fr, sr = full_remove[pair_ids], selected_remove[pair_ids]
        install = {"relative_l2": rel(si, fi), "cosine": cosine(si, fi), "sign_agreement": float(((si * fi) > 0).double().mean()), "full_target_rms": rms(fi), "selected_target_rms": rms(si)}
        removal = {"relative_l2": rel(sr, fr), "cosine": cosine(sr, fr), "sign_agreement": float(((sr * fr) > 0).double().mean()), "full_target_rms": rms(fr), "selected_target_rms": rms(sr)}
        install_controls = {READOUTS[index][0]: {"full_rms": rms(install_effect[0, row_ids, index]), "selected_rms": rms(install_effect[1, row_ids, index])} for index in range(1, len(READOUTS))}
        removal_controls = {READOUTS[index][0]: {"full_rms": rms(removal_effect[0, row_ids, index]), "selected_rms": rms(removal_effect[1, row_ids, index])} for index in range(1, len(READOUTS))}
        family_install_pass = install["relative_l2"] <= .35 and install["cosine"] >= .90 and install["sign_agreement"] >= .80
        family_removal_pass = removal["relative_l2"] <= .35 and removal["cosine"] >= .90 and removal["sign_agreement"] >= .80
        family_control_pass = all(value["selected_rms"] <= 1.25 * value["full_rms"] + 1e-6 for value in (*install_controls.values(), *removal_controls.values()))
        install_pass = install_pass and family_install_pass
        removal_pass = removal_pass and family_removal_pass
        control_pass = control_pass and family_control_pass
        family_reports[str(family)] = {"response_relative_l2": family_response_errors[str(family)], "installation": install, "removal": removal, "installation_controls": install_controls, "removal_controls": removal_controls, "passes_installation": family_install_pass, "passes_removal": family_removal_pass, "passes_controls": family_control_pass}

    errors = {
        "carry_source_reconstruction_relative_error": math.sqrt(carry_num / max(carry_den, 1e-30)),
        "manual_attention9_reconstruction_relative_error": math.sqrt(attention9_num / max(attention9_den, 1e-30)),
        "attention17_head_sum_relative_error": math.sqrt(attention17_num / max(attention17_den, 1e-30)),
        "head17_2_projection_relative_error": math.sqrt(head2_num / max(head2_den, 1e-30)),
        "factor_corner_endpoint_relative_error": math.sqrt(corner_num / max(corner_den, 1e-30)),
        "mobius_closure_relative_error": math.sqrt(closure_num / max(closure_den, 1e-30)),
    }
    instrument = executions == PRICE["physical_model_executions"] and max(errors.values()) <= 2e-6 and bool(torch.isfinite(values).all())
    sparse_response = selected_error <= .25 and selected_cosine >= .90 and max(family_response_errors.values()) <= .35
    predictions = {
        "pred_a_exact_instrument": bool(instrument),
        "pred_b_sparse_response_replay": bool(instrument and sparse_response),
        "pred_c_causal_installation": bool(instrument and sparse_response and install_pass),
        "pred_d_causal_removal": bool(instrument and sparse_response and removal_pass),
        "pred_e_control_nonworsening": bool(instrument and sparse_response and install_pass and removal_pass and control_pass),
        "pred_f_equal_norm_random_null": bool(instrument and sparse_response and install_pass and removal_pass and random_null_pass),
    }
    terminal = "head17_2_sparse_factor_candidate" if all(predictions.values()) else "valid_head17_2_factor_null" if instrument else "invalid"
    result = {
        "schema": "setting2_regional_attention17h2_factor_interaction_fold_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "instrument_errors": errors,
        "selected_support_indices": support,
        "selected_support_terms": [TERMS[index] for index in support],
        "selected_response_relative_l2": selected_error,
        "selected_response_cosine": selected_cosine,
        "family_response_relative_l2": family_response_errors,
        "family_reports": family_reports,
        "component_response_norm_ratios": {TERMS[index]: float(paired_response[index].norm() / full_response.norm().clamp_min(1e-30)) for index in range(7)},
        "equal_norm_random_null_errors": random_errors,
        "selected_aggregate_installation_relative_l2": selected_install_error,
        "selected_aggregate_removal_relative_l2": selected_remove_error,
        "price": PRICE,
        "row_checks": checks,
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Opened-panel exact seven-term QK1/QK2/value Möbius decomposition of the induced head17.2 response, with native-background installation and upstream-edited-background removal; no fresh transfer or factor generation.",
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "errors": errors, "support": result["selected_support_terms"], "response_error": selected_error, "response_cosine": selected_cosine, "families": family_reports}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
