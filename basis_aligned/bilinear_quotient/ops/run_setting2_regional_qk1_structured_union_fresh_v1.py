#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_native_capability pred_c_complete_route_live pred_d_structured_union_selective pred_e_unrelated_reader_control
"""Prospective recursive screen of structured head9.8 QK1 unions."""
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
import run_setting2_regional_head9_8_qk1_late_group_fresh_routing_v3 as helper
from regional_cue_row_check_v1 import validate


RUNNER = Path(__file__).resolve()
PREREG = P / "SETTING2_REGIONAL_QK1_STRUCTURED_UNION_FRESH_V1_PREREGISTRATION.md"
ROWS = P / "SETTING2_REGIONAL_QK1_STRUCTURED_UNION_FRESH_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_QK1_STRUCTURED_UNION_FRESH_V1_BINDING.json"
PARENT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_head9_8_qk1_late_group_fresh_routing_v3_result.json"
BLOCK_NULL = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_qk1_block_collateral_v1_result.json"
PRICE_PROOF = P / "QK1_STRUCTURED_UNION_BILINEAR_COMPLEXITY_20260915_1450_RESULT.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_qk1_structured_union_fresh_v1_result.json"

ARMS = ("native", "complete", "query_late", "key_late", "cross_only")
UNION_INDICES = {
    "complete": (0, 1, 2),
    "query_late": (0, 1),
    "key_late": (0, 2),
    "cross_only": (1, 2),
}
READOUTS = (
    ("target", None),
    ("work_jobs", (670, 3946)),
    ("cat_dog", (3797, 3290)),
    ("red_blue", (2266, 4171)),
    ("monday_tuesday", (3321, 3431)),
    ("apple_orange", (17180, 10912)),
)
PRICE = {
    "physical_model_executions": 30,
    "sequences_per_arm": 48,
    "arms": 5,
    "fits": 0,
    "backwards": 0,
    "parameter_updates": 0,
    "qk1_scalar_multiplies": {"complete": 256, "query_late": 128, "key_late": 128, "cross_only": 256},
}
EPS = torch.finfo(torch.float32).eps


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(x):
    return float(x.square().mean().sqrt())


def load_bound():
    binding = json.loads(BINDING.read_text())
    files = {
        "preregistration": PREREG,
        "rows": ROWS,
        "row_check": P / "regional_cue_row_check_v1.py",
        "intervention_helper": Path(helper.__file__).resolve(),
        "parent_result": PARENT,
        "block_null": BLOCK_NULL,
        "price_proof": PRICE_PROOF,
    }
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    parent = json.loads(PARENT.read_text())
    block = json.loads(BLOCK_NULL.read_text())
    price = json.loads(PRICE_PROOF.read_text())
    if parent["terminal"] != "valid_null" or not parent["predictions"]["pred_d_selected_material_and_exceeds_rr"]:
        raise ValueError("complete routing parent is not live")
    if block["terminal"] != "valid_qk1_block_split" or not block["predictions"]["pred_a_exact_instrument"] or block["predictions"]["pred_d_selectivity_improving_block"]:
        raise ValueError("singleton-block null authority changed")
    expected_price = {"complete_without_RR": 256, "query_late_DD_DR": 128, "key_late_DD_RD": 128, "cross_only_DR_RD": 256}
    if price["max_absolute_replay_error"] > 1e-12 or any(price["arms"][name]["minimum_separable_scalar_products"] != value for name, value in expected_price.items()):
        raise ValueError("price proof changed")
    rows_doc = json.loads(ROWS.read_text())
    rows = rows_doc["rows"]
    checks = validate(rows)
    buckets = {}
    for index, row in enumerate(rows):
        buckets.setdefault(len(row["ids"]), []).append(index)
    executions = len(ARMS) * sum(math.ceil(len(indices) / 8) for indices in buckets.values())
    if len(rows) != 48 or rows_doc["prior_context_overlap"] != 0 or executions != PRICE["physical_model_executions"]:
        raise ValueError("row authority or execution price changed")
    return binding, rows, checks, buckets


def plan():
    _, rows, checks, _ = load_bound()
    return {
        "schema": "setting2_regional_qk1_structured_union_fresh_v1_plan",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "rows": len(rows),
        "arms": ARMS,
        "late_names": helper.LATE_NAMES,
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
    _, rows, checks, buckets = load_bound()
    device = next(model.parameters()).device
    lambdas = torch.stack([block.lambdas.detach().double().cpu() for block in model.transformer.h])
    embed_c, write_c = helper.coefficients(lambdas)
    values = torch.empty(len(ARMS), len(rows), len(READOUTS), dtype=torch.float64)
    carry_num = carry_den = attention_num = attention_den = 0.0
    replay_num = {arm: 0.0 for arm in ARMS[1:]}
    replay_den = {arm: 0.0 for arm in ARMS[1:]}
    executions = 0
    late_indices = {helper.SOURCE_NAMES.index(name) for name in helper.LATE_NAMES}

    for _, indices in sorted(buckets.items()):
        for offset in range(0, len(indices), 8):
            selected = indices[offset:offset + 8]
            tokens = torch.tensor([rows[index]["ids"] for index in selected], device=device)
            batch, length = tokens.shape
            for arm_index, arm in enumerate(ARMS):
                x = F.rms_norm(model.transformer.wte(tokens), (1152,))
                x0 = x
                first = None
                writes = {}
                for layer, block in enumerate(model.transformer.h[:8]):
                    residual = block.lambdas[0] * x + block.lambdas[1] * x0
                    attention, first = block.attn(F.rms_norm(residual, (1152,)), first)
                    mlp = block.mlp(F.rms_norm(residual + attention, (1152,)))
                    writes[f"attn{layer}"] = attention
                    writes[f"mlp{layer}"] = mlp
                    x = residual + attention + mlp
                block8, block9 = model.transformer.h[8], model.transformer.h[9]
                residual8 = block8.lambdas[0] * x + block8.lambdas[1] * x0
                attention8, first = block8.attn(F.rms_norm(residual8, (1152,)), first)
                mlp8 = block8.mlp(F.rms_norm(residual8 + attention8, (1152,)))
                x = residual8 + attention8 + mlp8
                residual9 = block9.lambdas[0] * x + block9.lambdas[1] * x0
                norm9 = F.rms_norm(residual9, (1152,))
                attention9, first_out = block9.attn(norm9, first)
                at = block9.attn

                parts = [embed_c * x0] + [write_c[name] * writes[name] for name in helper.SOURCE_NAMES[1:]]
                carry = sum(parts)
                carry_native = block9.lambdas[0] * residual8 + block9.lambdas[1] * x0
                denominator = (residual9.square().mean(-1, keepdim=True) + EPS).sqrt()
                normalized = [part / denominator for part in parts]
                late = sum(normalized[index] for index in range(17) if index in late_indices)
                remainder = sum(normalized[index] for index in range(17) if index not in late_indices)

                qraw = at.c_q(norm9).view(batch, length, 9, 128)
                kraw = at.c_k(norm9).view_as(qraw)
                q2raw = at.c_q2(norm9).view_as(qraw)
                k2raw = at.c_k2(norm9).view_as(qraw)
                value = at.c_v(norm9).view_as(qraw)
                value = (1 - at.lamb) * value + at.lamb * first.view_as(value)
                cos, sin = at.rotary(qraw)
                q = helper.rotate(F.rms_norm(qraw, (128,)), cos, sin)
                k = helper.rotate(F.rms_norm(kraw, (128,)), cos, sin)
                q2 = helper.rotate(F.rms_norm(q2raw, (128,)), cos, sin)
                k2 = helper.rotate(F.rms_norm(k2raw, (128,)), cos, sin)
                groups = (late, remainder)
                qgroups = [helper.rotate(z, cos, sin) for z in helper.group_project(groups, qraw[:, :, 8], at.c_q.weight[8 * 128:9 * 128])]
                kgroups = [helper.rotate(z, cos, sin) for z in helper.group_project(groups, kraw[:, :, 8], at.c_k.weight[8 * 128:9 * 128])]
                s1 = torch.einsum("bqhd,bkhd->bhqk", q, k) / 128
                s2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / 128
                terms = [torch.einsum("bqd,bkd->bqk", qgroups[u], kgroups[v]) / 128 for u in range(2) for v in range(2)]
                mask = torch.tril(torch.ones(length, length, device=device, dtype=torch.bool))
                pattern = (s1 * s2).masked_fill(~mask, 0)
                zall = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
                manual = at.c_proj(zall.transpose(1, 2).contiguous().view_as(norm9))

                unions = {name: sum(terms[index] for index in term_indices) for name, term_indices in UNION_INDICES.items()}
                direct = {
                    "complete": terms[0] + terms[1] + terms[2],
                    "query_late": torch.einsum("bqd,bkd->bqk", qgroups[0], kgroups[0] + kgroups[1]) / 128,
                    "key_late": torch.einsum("bqd,bkd->bqk", qgroups[0] + qgroups[1], kgroups[0]) / 128,
                    "cross_only": torch.einsum("bqd,bkd->bqk", qgroups[0], kgroups[1]) / 128 + torch.einsum("bqd,bkd->bqk", qgroups[1], kgroups[0]) / 128,
                }
                if arm_index == 0:
                    carry_num += float((carry - carry_native).square().sum())
                    carry_den += float(carry_native.square().sum())
                    attention_num += float((manual - attention9).square().sum())
                    attention_den += float(attention9.square().sum())
                    for name in ARMS[1:]:
                        replay_num[name] += float((unions[name] - direct[name]).square().sum())
                        replay_den[name] += float(direct[name].square().sum())
                if arm != "native":
                    selected_pattern = (unions[arm] * s2[:, 8]).masked_fill(~mask, 0)
                    edit = -torch.einsum("bqk,bkd->bqd", selected_pattern, value[:, :, 8])
                    attention9 = attention9 + edit @ at.c_proj.weight[:, 8 * 128:9 * 128].T

                x = residual9 + attention9
                x = x + block9.mlp(F.rms_norm(x, (1152,)))
                first = first_out
                for block in model.transformer.h[10:]:
                    residual = block.lambdas[0] * x + block.lambdas[1] * x0
                    attention, first = block.attn(F.rms_norm(residual, (1152,)), first)
                    x = residual + attention + block.mlp(F.rms_norm(residual + attention, (1152,)))
                logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x[:, -1], (1152,))) / 30)).double().cpu()
                for local, row_index in enumerate(selected):
                    pairs = [(rows[row_index]["uk_id"], rows[row_index]["us_id"])] + [pair for _, pair in READOUTS[1:]]
                    for reader, (left, right) in enumerate(pairs):
                        values[arm_index, row_index, reader] = logits[local, left] - logits[local, right]
                executions += 1

    effects = values - values[0]
    pair_target = effects[:, ::2, 0] - effects[:, 1::2, 0]
    native_pair = values[0, ::2, 0] - values[0, 1::2, 0]
    family_reports = {}
    candidate_pass = {name: True for name in ARMS[2:]}
    capability = True
    route_live = True
    unrelated_control = {name: True for name in ARMS[2:]}
    for family in sorted({row["family"] for row in rows}):
        row_ids = [index for index, row in enumerate(rows) if row["family"] == family]
        pair_ids = [index // 2 for index in row_ids[::2]]
        native = native_pair[pair_ids]
        complete_target = pair_target[1, pair_ids]
        complete_work = effects[1, row_ids, 1]
        complete_ratio = rms(complete_work) / max(rms(complete_target), 1e-30)
        capability = capability and int((native > 0).sum()) >= 9 and rms(native) >= .01
        route_live = route_live and rms(complete_target) >= .002
        candidates = {}
        for arm_index, name in enumerate(ARMS[2:], 2):
            target = pair_target[arm_index, pair_ids]
            work = effects[arm_index, row_ids, 1]
            unrelated = {READOUTS[index][0]: rms(effects[arm_index, row_ids, index]) for index in range(2, len(READOUTS))}
            target_rms = rms(target)
            retained = target_rms / max(rms(complete_target), 1e-30)
            work_ratio = rms(work) / max(target_rms, 1e-30)
            passes_selectivity = retained >= .20 and work_ratio <= .70 * complete_ratio
            passes_unrelated = max(unrelated.values()) <= target_rms
            candidate_pass[name] = candidate_pass[name] and passes_selectivity
            unrelated_control[name] = unrelated_control[name] and passes_unrelated
            candidates[name] = {
                "target_rms": target_rms,
                "target_retained_fraction": retained,
                "work_jobs_rms": rms(work),
                "work_jobs_to_target_ratio": work_ratio,
                "work_jobs_ratio_improvement_fraction": 1 - work_ratio / max(complete_ratio, 1e-30),
                "unrelated_rms": unrelated,
                "passes_selectivity": passes_selectivity,
                "passes_unrelated": passes_unrelated,
            }
        family_reports[str(family)] = {
            "native_positive_pairs": int((native > 0).sum()),
            "native_paired_target_rms": rms(native),
            "complete_target_rms": rms(complete_target),
            "complete_work_jobs_rms": rms(complete_work),
            "complete_work_jobs_to_target_ratio": complete_ratio,
            "candidates": candidates,
        }

    selected_candidates = [name for name in ARMS[2:] if candidate_pass[name] and unrelated_control[name]]
    carry_error = math.sqrt(carry_num / max(carry_den, 1e-30))
    attention_error = math.sqrt(attention_num / max(attention_den, 1e-30))
    replay_errors = {name: math.sqrt(replay_num[name] / max(replay_den[name], 1e-30)) for name in ARMS[1:]}
    instrument = executions == PRICE["physical_model_executions"] and max(carry_error, attention_error, *replay_errors.values()) <= 2e-6 and bool(torch.isfinite(values).all())
    predictions = {
        "pred_a_exact_instrument": bool(instrument),
        "pred_b_native_capability": bool(instrument and capability),
        "pred_c_complete_route_live": bool(instrument and capability and route_live),
        "pred_d_structured_union_selective": bool(instrument and capability and route_live and selected_candidates),
        "pred_e_unrelated_reader_control": bool(instrument and capability and route_live and selected_candidates),
    }
    if not instrument or not capability:
        terminal = "invalid"
    elif selected_candidates:
        terminal = "fresh_structured_qk1_union"
    else:
        terminal = "valid_structured_union_null"
    metrics = {
        "carry_source_reconstruction_relative_error": carry_error,
        "manual_attention9_reconstruction_relative_error": attention_error,
        "structured_union_replay_relative_errors": replay_errors,
        "physical_model_executions": executions,
        "rows": len(rows),
        "pairs": len(rows) // 2,
    }
    result = {
        "schema": "setting2_regional_qk1_structured_union_fresh_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "selected_candidates": selected_candidates,
        "candidate_pass_both_families": candidate_pass,
        "unrelated_control_both_families": unrelated_control,
        "metrics": metrics,
        "family_reports": family_reports,
        "arms": ARMS,
        "readouts": [name for name, _ in READOUTS],
        "price": PRICE,
        "row_checks": checks,
        "outcome_access": {"target_logits": True, "control_logits": True, "fits": False, "row_selection": False},
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Prospective fresh-context recursive removal of complete, query-late, key-late, and cross-only head9.8 QK1 unions; native QK2/value/suffix; no fit or parameter update.",
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected_candidates": selected_candidates, "metrics": metrics, "families": family_reports}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
