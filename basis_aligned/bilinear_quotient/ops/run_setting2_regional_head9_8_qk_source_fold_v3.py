#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_top3_concentration pred_c_attention8_routing pred_d_stable_leader
"""Exact carry8/attention8/MLP8 source-pair folds for both head9.8 QK factors."""
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
from folded_normalized_router_v1 import EPS, rotary
from regional_cue_row_check_v1 import validate


RUNNER = Path(__file__).resolve()
PREREG = P / "SETTING2_REGIONAL_HEAD9_8_QK_SOURCE_FOLD_V1_PREREGISTRATION.md"
ROWS = P / "FIRST_TOKEN_PATH_FRESH_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_HEAD9_8_QK_SOURCE_FOLD_V3_BINDING.json"
PARENT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attn9_head_mlp16_fold_v2_result.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_head9_8_qk_source_fold_v3_result.json"
PRICE = {"physical_prefix_forwards": 14, "sequences": 96, "qk_factors": 2,
         "ordered_source_terms_per_factor": 9, "fits": 0, "backwards": 0,
         "parameter_updates": 0}
NAMES = ("carry8", "attn8", "mlp8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


def load_bound() -> tuple[dict, list[dict]]:
    binding = json.loads(BINDING.read_text())
    files = {"preregistration": PREREG, "rows": ROWS,
             "row_check": P / "regional_cue_row_check_v1.py", "parent_result": PARENT,
             "correction_v2": P / "SETTING2_REGIONAL_HEAD9_8_QK_SOURCE_FOLD_V2_CORRECTION.md",
             "correction_v3": P / "SETTING2_REGIONAL_HEAD9_8_QK_SOURCE_FOLD_V3_CORRECTION.md",
             "failed_v1_runner": ROOT / "basis_aligned/bilinear_quotient/ops/run_setting2_regional_head9_8_qk_source_fold_v1.py",
             "failed_v2_runner": ROOT / "basis_aligned/bilinear_quotient/ops/run_setting2_regional_head9_8_qk_source_fold_v2.py"}
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    parent = json.loads(PARENT.read_text())
    if parent["terminal"] != "valid_attention9_head_fold" or parent["frozen_head_ranking"][0] != "head8":
        raise ValueError("parent head9.8 path is not leading")
    rows = json.loads(ROWS.read_text())["rows"]
    checks = validate(rows)
    buckets = {}
    for i, row in enumerate(rows):
        buckets.setdefault(len(row["ids"]), []).append(i)
    if len(rows) != 96 or sum(math.ceil(len(v) / 8) for v in buckets.values()) != 14:
        raise ValueError("row/forward price changed")
    return {**binding, "row_checks": checks, "buckets": buckets}, rows


def plan() -> dict:
    bound, rows = load_bound()
    return {"schema": "setting2_regional_head9_8_qk_source_fold_v3_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "source_groups": NAMES, "price": PRICE,
            "row_checks": bound["row_checks"], "binding_sha256": sha(BINDING)}


def normalized_parts(parts: list[torch.Tensor], total: torch.Tensor) -> list[torch.Tensor]:
    denominator = (total.square().mean(-1, keepdim=True) + EPS).sqrt()
    return [part / denominator for part in parts]


def projected_parts(parts: list[torch.Tensor], total: torch.Tensor,
                    weight: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    projected = [part @ weight.T for part in parts]
    projected_total = total @ weight.T
    denominator = (projected_total.square().mean(-1, keepdim=True) + EPS).sqrt()
    return [part / denominator for part in projected], projected_total / denominator


@torch.no_grad()
def main() -> None:
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
    bound, rows = load_bound()
    device = next(model.parameters()).device
    state = model.state_dict()
    lambdas = torch.stack([block.lambdas.detach().double().cpu() for block in model.transformer.h])
    coefficient9 = float(torch.prod(lambdas[10:18, 0]))
    coefficient16 = float(lambdas[17, 0])
    projection9 = state["transformer.h.9.attn.c_proj.weight"].double().cpu()
    q_weights = [[state[f"transformer.h.9.attn.c_{name}.weight"][8 * 128:9 * 128].double().cpu()
                  for name in pair] for pair in (("q", "k"), ("q2", "k2"))]
    left, right, down = [state[f"transformer.h.17.mlp.{name}.weight"].double().cpu()
                         for name in ("Left", "Right", "Down")]
    unembed = state["lm_head.weight"].double().cpu()

    folded = torch.empty(2, 9, 96, dtype=torch.float64)
    directs = torch.empty(2, 96, dtype=torch.float64)
    native_head = torch.empty(96, 128, dtype=torch.float64)
    recomputed_head = torch.empty_like(native_head)
    raw_exact_num = raw_exact_den = 0.0
    raw_native_num = raw_native_den = 0.0
    score_error_num = [0.0, 0.0]
    score_error_den = [0.0, 0.0]
    context = {"selected": None}

    def head_hook(module, args):
        native_head[context["selected"]] = args[0][:, -1, 8 * 128:9 * 128].double().cpu()

    handle = model.transformer.h[9].attn.c_proj.register_forward_pre_hook(head_hook)
    forward_count = 0
    try:
        for _, indices in sorted(bound["buckets"].items()):
            for offset in range(0, len(indices), 8):
                selected = indices[offset:offset + 8]
                context["selected"] = selected
                tokens = torch.tensor([rows[i]["ids"] for i in selected], device=device)
                x = F.rms_norm(model.transformer.wte(tokens), (1152,))
                x0, first_value = x, None
                for block in model.transformer.h[:8]:
                    x, first_value = block(x, first_value, x0)
                block8, block9 = model.transformer.h[8], model.transformer.h[9]
                r8 = block8.lambdas[0] * x + block8.lambdas[1] * x0
                a8, first_value = block8.attn(F.rms_norm(r8, (1152,)), first_value)
                m8 = block8.mlp(F.rms_norm(r8 + a8, (1152,)))
                x = r8 + a8 + m8
                r9_native = block9.lambdas[0] * x + block9.lambdas[1] * x0
                source_parts = [(block9.lambdas[0].double() * r8.double() +
                                 block9.lambdas[1].double() * x0.double()).cpu(),
                                (block9.lambdas[0].double() * a8.double()).cpu(),
                                (block9.lambdas[0].double() * m8.double()).cpu()]
                r9 = sum(source_parts)
                r9_expected = (block9.lambdas[0].double().cpu() *
                               (r8.double().cpu() + a8.double().cpu() + m8.double().cpu()) +
                               block9.lambdas[1].double().cpu() * x0.double().cpu())
                raw_exact_num += float((r9 - r9_expected).square().sum())
                raw_exact_den += float(r9_expected.square().sum())
                raw_native_num += float((r9 - r9_native.double().cpu()).square().sum())
                raw_native_den += float(r9_native.double().cpu().square().sum())
                input_parts = normalized_parts(source_parts, r9)
                input_total = sum(input_parts)
                factor_scores, factor_components = [], []
                length = tokens.shape[1]
                rotations = torch.stack([rotary(pos, 128) for pos in range(length)])
                for factor in range(2):
                    q_parts, q_total = projected_parts(input_parts, input_total, q_weights[factor][0])
                    k_parts, k_total = projected_parts(input_parts, input_total, q_weights[factor][1])
                    q_parts = [torch.einsum("btd,tde->bte", part, rotations.transpose(1, 2))
                               for part in q_parts]
                    k_parts = [torch.einsum("btd,tde->bte", part, rotations.transpose(1, 2))
                               for part in k_parts]
                    q_total = torch.einsum("btd,tde->bte", q_total, rotations.transpose(1, 2))
                    k_total = torch.einsum("btd,tde->bte", k_total, rotations.transpose(1, 2))
                    components = torch.stack([torch.einsum("bd,btd->bt", q_parts[u][:, -1], k_parts[v]) / 128
                                              for u in range(3) for v in range(3)])
                    total_score = torch.einsum("bd,btd->bt", q_total[:, -1], k_total) / 128
                    factor_components.append(components)
                    factor_scores.append(total_score)
                    score_error_num[factor] += float((components.sum(0) - total_score).square().sum())
                    score_error_den[factor] += float(total_score.square().sum())

                norm9_native = F.rms_norm(r9_native, (1152,))
                value = block9.attn.c_v(norm9_native).view(len(selected), length, 9, 128)
                value = ((1 - block9.attn.lamb) * value +
                         block9.attn.lamb * first_value.view_as(value))[:, :, 8].double().cpu()
                z_terms = []
                for factor in range(2):
                    other = factor_scores[1 - factor]
                    z_terms.append(torch.stack([((component * other)[:, :, None] * value).sum(1)
                                                 for component in factor_components[factor]]))
                z_direct = ((factor_scores[0] * factor_scores[1])[:, :, None] * value).sum(1)
                recomputed_head[selected] = z_direct

                x, first_value = block9(x, first_value, x0)
                mlp16 = None
                for layer in range(10, 17):
                    block = model.transformer.h[layer]
                    r = block.lambdas[0] * x + block.lambdas[1] * x0
                    attention, first_value = block.attn(F.rms_norm(r, (1152,)), first_value)
                    mlp = block.mlp(F.rms_norm(r + attention, (1152,)))
                    x = r + attention + mlp
                    if layer == 16:
                        mlp16 = mlp[:, -1].double().cpu()
                last = model.transformer.h[17]
                raw17 = last.lambdas[0] * x + last.lambdas[1] * x0
                attention17, first_value = last.attn(F.rms_norm(raw17, (1152,)), first_value)
                pre17 = (raw17 + attention17)[:, -1].double().cpu()
                p = coefficient16 * mlp16
                readers = torch.stack([unembed[rows[i]["uk_id"]] - unembed[rows[i]["us_id"]]
                                       for i in selected])
                folded_down = readers @ down
                lp, rp = p @ left.T, p @ right.T
                denominator17 = pre17.square().mean(-1) + EPS

                def cross(source: torch.Tensor) -> torch.Tensor:
                    return (((source @ left.T) * rp + lp * (source @ right.T)) *
                            folded_down).sum(-1) / denominator17

                direct_write = coefficient9 * z_direct @ projection9[:, 8 * 128:9 * 128].T
                for factor in range(2):
                    for term in range(9):
                        write = coefficient9 * z_terms[factor][term] @ projection9[:, 8 * 128:9 * 128].T
                        folded[factor, term, selected] = cross(write)
                    directs[factor, selected] = cross(direct_write)
                forward_count += 1
    finally:
        handle.remove()

    raw_exact_error = math.sqrt(raw_exact_num / max(raw_exact_den, 1e-30))
    raw_native_bridge = math.sqrt(raw_native_num / max(raw_native_den, 1e-30))
    score_errors = [math.sqrt(score_error_num[f] / max(score_error_den[f], 1e-30)) for f in range(2)]
    head_bridge = rel(recomputed_head, native_head)
    folded_errors = [rel(folded[f].sum(0), directs[f]) for f in range(2)]
    direct_factor_bridge = rel(directs[0], directs[1])
    reports, rankings, top3_errors, attention8_ratios = [], [], [], []
    families = sorted(set(row["family_name"] for row in rows))
    for factor in range(2):
        delta = directs[factor, 1::2] - directs[factor, 0::2]
        delta_terms = folded[factor, :, 1::2] - folded[factor, :, 0::2]
        denom = delta.square().sum().clamp_min(1e-30)
        factor_report = {}
        for term, (u, v) in enumerate((u, v) for u in range(3) for v in range(3)):
            name = f"{NAMES[u]}_x_{NAMES[v]}"
            family_ratios = {}
            for family in families:
                ids = [i // 2 for i in range(0, 96, 2) if rows[i]["family_name"] == family]
                family_ratios[family] = float(delta_terms[term, ids].norm() /
                                              delta[ids].norm().clamp_min(1e-30))
            factor_report[name] = {"change_norm_ratio": float(delta_terms[term].norm() / delta.norm().clamp_min(1e-30)),
                                   "aligned_fraction": float((delta_terms[term] * delta).sum() / denom),
                                   "family_change_norm_ratios": family_ratios}
        ranking = sorted(factor_report, key=lambda n: factor_report[n]["change_norm_ratio"], reverse=True)
        term_index = {f"{NAMES[u]}_x_{NAMES[v]}": 3 * u + v for u in range(3) for v in range(3)}
        top3_error = rel(sum(delta_terms[term_index[n]] for n in ranking[:3]), delta)
        attention_terms = [i for i, (u, v) in enumerate((u, v) for u in range(3) for v in range(3))
                           if u == 1 or v == 1]
        attention_ratio = float(delta_terms[attention_terms].sum(0).norm() /
                                delta.norm().clamp_min(1e-30))
        reports.append(factor_report); rankings.append(ranking)
        top3_errors.append(top3_error); attention8_ratios.append(attention_ratio)
    instrument = (forward_count == 14 and raw_exact_error <= 1e-8 and
                  max(score_errors + folded_errors) <= 1e-8 and head_bridge <= 1e-6)
    best_factor = min(range(2), key=lambda f: top3_errors[f])
    stable = any(reports[f][rankings[f][0]]["change_norm_ratio"] >= .20 and
                 min(reports[f][rankings[f][0]]["family_change_norm_ratios"].values()) >= .20
                 for f in range(2))
    predictions = {"pred_a_exact_instrument": bool(instrument),
                   "pred_b_top3_concentration": bool(instrument and min(top3_errors) <= .50),
                   "pred_c_attention8_routing": bool(instrument and max(attention8_ratios) >= .10),
                   "pred_d_stable_leader": bool(instrument and stable)}
    result = {"schema": "setting2_regional_head9_8_qk_source_fold_v3_result",
              "terminal": "valid_qk_source_fold" if instrument else "invalid",
              "predictions": predictions,
              "metrics": {"raw_source_exact_relative_error": raw_exact_error,
                          "raw_native_rounding_bridge_relative_error": raw_native_bridge,
                          "qk_score_sum_relative_errors": score_errors,
                          "head9_8_native_bridge_relative_error": head_bridge,
                          "folded_sum_relative_errors": folded_errors,
                          "direct_factor_bridge_relative_error": direct_factor_bridge,
                          "top3_replay_relative_errors": top3_errors,
                          "attention8_aggregate_change_norm_ratios": attention8_ratios,
                          "physical_prefix_forwards": forward_count, "sequences": 96,
                          "pair_count": 48},
              "rankings": {f"qk{f + 1}": rankings[f] for f in range(2)},
              "top3_terms": {f"qk{f + 1}": rankings[f][:3] for f in range(2)},
              "term_reports": {f"qk{f + 1}": reports[f] for f in range(2)},
              "best_concentration_factor": f"qk{best_factor + 1}",
              "price": PRICE, "row_checks": planned["row_checks"],
              "outcome_access": {"behavioral_logits": False, "new_rows": False, "fits": False},
              "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Exact head9.8 QK source-pair attribution in the regional MLP16/MLP17 folded path. Native denominators, values, suffix, and reader are held fixed; no behavioral, causal-sufficiency, value-source, OOD, or compression claim."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": result["terminal"], "predictions": predictions,
                      "metrics": result["metrics"], "rankings": result["rankings"],
                      "top_ratios": {f"qk{f + 1}": {n: reports[f][n]["change_norm_ratio"]
                                                     for n in rankings[f]} for f in range(2)}}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
