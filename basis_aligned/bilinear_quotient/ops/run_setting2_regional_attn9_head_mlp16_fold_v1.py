#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_head_fold_instrument pred_b_top2_head_concentration pred_c_stable_leading_head pred_d_head9_8_cross_setting_overlap
"""Split the task-matched attention9 x MLP16 folded path into native heads."""
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
from regional_cue_row_check_v1 import validate


RUNNER = Path(__file__).resolve()
PREREG = P / "SETTING2_REGIONAL_ATTN9_HEAD_MLP16_FOLD_V1_PREREGISTRATION.md"
ROWS = P / "FIRST_TOKEN_PATH_FRESH_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_ATTN9_HEAD_MLP16_FOLD_V1_BINDING.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attn9_head_mlp16_fold_v1_result.json"
PRICE = {"physical_prefix_forwards": 14, "sequences": 96, "head_terms": 9,
         "fits": 0, "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_exact_head_fold_instrument": None,
                       "pred_b_top2_head_concentration": None,
                       "pred_c_stable_leading_head": None,
                       "pred_d_head9_8_cross_setting_overlap": None}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def projected_head_writes(values: torch.Tensor, projection: torch.Tensor,
                          coefficient: float) -> list[torch.Tensor]:
    return [coefficient * values[:, head * 128:(head + 1) * 128] @
            projection[:, head * 128:(head + 1) * 128].T for head in range(9)]


def load_bound() -> tuple[dict, list[dict]]:
    binding = json.loads(BINDING.read_text())
    files = {"preregistration": PREREG, "rows": ROWS,
             "row_check": P / "regional_cue_row_check_v1.py",
             "parent_result": ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_mlp16_upstream_source_fold_v1_result.json"}
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    parent = json.loads(files["parent_result"].read_text())
    if parent["terminal"] != "valid_upstream_source_fold" or parent["frozen_source_ranking"][0] != "attn9":
        raise ValueError("parent attention9 path is not leading")
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
    return {"schema": "setting2_regional_attn9_head_mlp16_fold_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "heads": list(range(9)), "price": PRICE,
            "row_checks": bound["row_checks"], "binding_sha256": sha(BINDING)}


def rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


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
    head_values = torch.empty(96, 1152, dtype=torch.float64)
    attention9 = torch.empty_like(head_values)
    mlp16 = torch.empty_like(head_values)
    raw17 = torch.empty_like(head_values)
    pre17 = torch.empty_like(head_values)
    context = {"selected": None}

    def head_hook(module, args):
        head_values[context["selected"]] = args[0][:, -1].double().cpu()

    def attention_hook(module, args, output):
        attention9[context["selected"]] = output[0][:, -1].double().cpu()

    def mlp_hook(module, args, output):
        mlp16[context["selected"]] = output[:, -1].double().cpu()

    handles = [model.transformer.h[9].attn.c_proj.register_forward_pre_hook(head_hook),
               model.transformer.h[9].attn.register_forward_hook(attention_hook),
               model.transformer.h[16].mlp.register_forward_hook(mlp_hook)]
    forward_count = 0
    try:
        for _, indices in sorted(bound["buckets"].items()):
            for offset in range(0, len(indices), 8):
                selected = indices[offset:offset + 8]
                context["selected"] = selected
                tokens = torch.tensor([rows[i]["ids"] for i in selected], device=device)
                x = F.rms_norm(model.transformer.wte(tokens), (1152,))
                x0, first_value = x, None
                for block in model.transformer.h[:17]:
                    x, first_value = block(x, first_value, x0)
                last = model.transformer.h[17]
                raw = last.lambdas[0] * x + last.lambdas[1] * x0
                attention, first_value = last.attn(F.rms_norm(raw, (1152,)), first_value)
                raw17[selected] = raw[:, -1].double().cpu()
                pre17[selected] = (raw + attention)[:, -1].double().cpu()
                forward_count += 1
    finally:
        for handle in handles:
            handle.remove()

    state = model.state_dict()
    lambdas = torch.stack([block.lambdas.detach().double().cpu() for block in model.transformer.h])
    coefficient9 = float(torch.prod(lambdas[10:18, 0]))
    coefficient16 = float(lambdas[17, 0])
    projection9 = state["transformer.h.9.attn.c_proj.weight"].double().cpu()
    heads = projected_head_writes(head_values, projection9, coefficient9)
    head_reconstruction_error = rel(sum(heads), coefficient9 * attention9)
    p = coefficient16 * mlp16
    left, right, down = [state[f"transformer.h.17.mlp.{name}.weight"].double().cpu()
                         for name in ("Left", "Right", "Down")]
    unembed = state["lm_head.weight"]
    readers = torch.stack([unembed[row["uk_id"]] - unembed[row["us_id"]] for row in rows]).double().cpu()
    folded_down = readers @ down
    lp, rp = p @ left.T, p @ right.T
    denominator = pre17.square().mean(-1) + torch.finfo(torch.float32).eps

    def cross(source: torch.Tensor) -> torch.Tensor:
        return (((source @ left.T) * rp + lp * (source @ right.T)) * folded_down).sum(-1) / denominator

    terms = {f"head{head}": cross(value) for head, value in enumerate(heads)}
    direct = cross(coefficient9 * attention9)
    delta = direct[1::2] - direct[0::2]
    delta_terms = {name: value[1::2] - value[0::2] for name, value in terms.items()}
    reports = {}
    denom_delta = delta.square().sum().clamp_min(1e-30)
    families = sorted(set(row["family_name"] for row in rows))
    for name, value in delta_terms.items():
        family_ratios = {}
        for family in families:
            indices = [i // 2 for i in range(0, 96, 2) if rows[i]["family_name"] == family]
            family_ratios[family] = float(value[indices].norm() / delta[indices].norm().clamp_min(1e-30))
        reports[name] = {"change_norm_ratio": float(value.norm() / delta.norm().clamp_min(1e-30)),
                         "aligned_fraction": float((value * delta).sum() / denom_delta),
                         "family_change_norm_ratios": family_ratios}
    ranking = sorted(reports, key=lambda name: reports[name]["change_norm_ratio"], reverse=True)
    top2 = ranking[:2]
    term_error = rel(sum(terms.values()), direct)
    top2_error = rel(sum(delta_terms[name] for name in top2), delta)
    leading = reports[ranking[0]]
    instrument = forward_count == 14 and head_reconstruction_error <= 1e-6 and term_error <= 1e-8
    predictions = {"pred_a_exact_head_fold_instrument": bool(instrument),
                   "pred_b_top2_head_concentration": bool(instrument and top2_error <= 0.40),
                   "pred_c_stable_leading_head": bool(instrument and leading["change_norm_ratio"] >= 0.25
                                                       and min(leading["family_change_norm_ratios"].values()) >= 0.05),
                   "pred_d_head9_8_cross_setting_overlap": bool(instrument and (ranking[0] == "head8" or reports["head8"]["change_norm_ratio"] >= 0.20))}
    result = {"schema": "setting2_regional_attn9_head_mlp16_fold_v1_result",
              "terminal": "valid_attention9_head_fold" if instrument else "invalid",
              "predictions": predictions,
              "metrics": {"head_write_reconstruction_relative_error": head_reconstruction_error,
                          "head_cross_sum_relative_error": term_error,
                          "top2_replay_relative_error": top2_error,
                          "change_cancellation_ratio": float(delta.norm() / sum(v.norm() for v in delta_terms.values()).clamp_min(1e-30)),
                          "physical_prefix_forwards": forward_count, "sequences": len(rows), "pair_count": len(delta)},
              "propagation_coefficient_attention9": coefficient9,
              "propagation_coefficient_mlp16": coefficient16,
              "head_reports": reports, "frozen_head_ranking": ranking, "top2_heads": top2,
              "price": PRICE, "row_checks": planned["row_checks"],
              "outcome_access": {"behavioral_logits": False, "new_rows": False, "fits": False},
              "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Exact task-matched attention9-head x MLP16 folded MLP17 numerator screen. No QK/value source decomposition, behavioral outcome, causal reuse, OOD, compression, or adoption claim."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": result["terminal"], "predictions": predictions,
                      "metrics": result["metrics"], "ranking": ranking,
                      "ratios": {name: reports[name]["change_norm_ratio"] for name in ranking}}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
