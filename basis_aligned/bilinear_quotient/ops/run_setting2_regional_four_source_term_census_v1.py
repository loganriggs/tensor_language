#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_task_matched_instrument pred_b_head17_2_terms_live pred_c_earlier_residual_self_dominant pred_d_earlier_residual_mlp16_cross_live
"""Capture regional MLP17 sources and exactly census ten normalized interaction terms."""
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
PREREG = P / "SETTING2_REGIONAL_FOUR_SOURCE_TERM_CENSUS_V1_PREREGISTRATION.md"
ROWS = P / "FIRST_TOKEN_PATH_FRESH_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_FOUR_SOURCE_TERM_CENSUS_V1_BINDING.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_four_source_term_census_v1_result.json"
PRICE = {"physical_prefix_forwards": 14, "sequences": 96, "fits": 0,
         "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {
    "pred_a_exact_task_matched_instrument": None,
    "pred_b_head17_2_terms_live": None,
    "pred_c_earlier_residual_self_dominant": None,
    "pred_d_earlier_residual_mlp16_cross_live": None,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_terms(sources: dict[str, torch.Tensor], left: torch.Tensor,
               right: torch.Tensor, folded_down: torch.Tensor) -> dict[str, torch.Tensor]:
    lhs = {name: value @ left.T for name, value in sources.items()}
    rhs = {name: value @ right.T for name, value in sources.items()}
    order = tuple(sources)
    terms = {}
    for i, first in enumerate(order):
        for second in order[i:]:
            value = lhs[first] * rhs[second]
            if first != second:
                value = value + lhs[second] * rhs[first]
            terms[first + second] = (value * folded_down).sum(-1)
    return terms


def load_bound() -> tuple[dict, list[dict]]:
    binding = json.loads(BINDING.read_text())
    files = {"preregistration": PREREG, "rows": ROWS,
             "row_check": P / "regional_cue_row_check_v1.py"}
    if binding["files"] != {name: sha(path) for name, path in files.items()}:
        raise ValueError("bound input changed")
    if binding["price"] != PRICE:
        raise ValueError("bound price changed")
    rows = json.loads(ROWS.read_text())["rows"]
    checks = validate(rows)
    buckets = {}
    for i, row in enumerate(rows):
        buckets.setdefault(len(row["ids"]), []).append(i)
    forwards = sum(math.ceil(len(indices) / 8) for indices in buckets.values())
    if len(rows) != PRICE["sequences"] or forwards != PRICE["physical_prefix_forwards"]:
        raise ValueError("row/forward price changed")
    return {**binding, "row_checks": checks, "buckets": buckets}, rows


def plan() -> dict:
    binding, rows = load_bound()
    return {"schema": "setting2_regional_four_source_term_census_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "families": sorted(set(row["family_name"] for row in rows)),
            "source_order": ["e", "p", "o", "a"], "term_count": 10,
            "price": PRICE, "binding_sha256": sha(BINDING),
            "row_checks": binding["row_checks"]}


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
    last = model.transformer.h[17]
    saved = {name: torch.empty(len(rows), 1152, dtype=torch.float64)
             for name in ("raw", "mlp16", "pre")}
    saved_head = torch.empty(len(rows), 128, dtype=torch.float64)
    captured = {}
    forward_count = 0

    def mlp16_hook(module, args, output):
        captured["mlp16"] = output[:, -1].detach()

    def head_input_hook(module, args):
        captured["head"] = args[0][:, -1, 2 * 128 : 3 * 128].detach()

    handles = [model.transformer.h[16].mlp.register_forward_hook(mlp16_hook),
               last.attn.c_proj.register_forward_pre_hook(head_input_hook)]
    try:
        for length, indices in sorted(bound["buckets"].items()):
            for offset in range(0, len(indices), 8):
                selected = indices[offset:offset + 8]
                tokens = torch.tensor([rows[i]["ids"] for i in selected], device=device)
                x = F.rms_norm(model.transformer.wte(tokens), (1152,))
                x0, first_value = x, None
                for block in model.transformer.h[:17]:
                    x, first_value = block(x, first_value, x0)
                raw = last.lambdas[0] * x + last.lambdas[1] * x0
                attention, first_value = last.attn(F.rms_norm(raw, (1152,)), first_value)
                pre = raw + attention
                forward_count += 1
                saved["raw"][selected] = raw[:, -1].double().cpu()
                saved["mlp16"][selected] = captured["mlp16"].double().cpu()
                saved["pre"][selected] = pre[:, -1].double().cpu()
                saved_head[selected] = captured["head"].double().cpu()
    finally:
        for handle in handles:
            handle.remove()

    state = model.state_dict()
    left, right, down = [state[f"transformer.h.17.mlp.{name}.weight"].double().cpu()
                         for name in ("Left", "Right", "Down")]
    unembed = state["lm_head.weight"]
    out_slice = state["transformer.h.17.attn.c_proj.weight"].double().cpu()[:, 2 * 128 : 3 * 128]
    reader = torch.stack([unembed[row["uk_id"]] - unembed[row["us_id"]] for row in rows]).double().cpu()
    folded_down = reader @ down
    p = float(last.lambdas[0].detach().cpu()) * saved["mlp16"]
    a = saved_head @ out_slice.T
    sources = {"e": saved["raw"] - p, "p": p,
               "o": saved["pre"] - saved["raw"] - a, "a": a}
    terms = pair_terms(sources, left, right, folded_down)
    direct = (((saved["pre"] @ left.T) * (saved["pre"] @ right.T)) * folded_down).sum(-1)
    raw_sum = sum(terms.values())
    denom = saved["pre"].square().mean(-1) + torch.finfo(torch.float32).eps
    normalized = {name: value / denom for name, value in terms.items()}
    normalized_direct = direct / denom
    normalized_sum = sum(normalized.values())
    delta = normalized_direct[1::2] - normalized_direct[0::2]
    raw_delta = direct[1::2] - direct[0::2]
    delta_terms = {name: value[1::2] - value[0::2] for name, value in normalized.items()}
    raw_delta_terms = {name: value[1::2] - value[0::2] for name, value in terms.items()}
    denom_delta = delta.square().sum().clamp_min(1e-30)
    reports = {}
    for name, value in delta_terms.items():
        ratio = float(value.norm() / delta.norm().clamp_min(1e-30))
        family_ratios = {}
        for family in sorted(set(row["family_name"] for row in rows)):
            pair_indices = [i // 2 for i in range(0, len(rows), 2) if rows[i]["family_name"] == family]
            family_ratios[family] = float(value[pair_indices].norm() / delta[pair_indices].norm().clamp_min(1e-30))
        reports[name] = {"normalized_change_norm_ratio": ratio,
                         "raw_change_norm_ratio": float(raw_delta_terms[name].norm() / raw_delta.norm().clamp_min(1e-30)),
                         "aligned_fraction": float((value * delta).sum() / denom_delta),
                         "family_normalized_change_norm_ratios": family_ratios,
                         "live": ratio >= 0.05}
    ranking = sorted(reports, key=lambda name: reports[name]["normalized_change_norm_ratio"], reverse=True)
    head_terms = sum(value for name, value in delta_terms.items() if "a" in name)
    head_ratio = float(head_terms.norm() / delta.norm().clamp_min(1e-30))
    exact = (max(rel(raw_sum, direct), rel(normalized_sum, normalized_direct)) <= 1e-8
             and forward_count == PRICE["physical_prefix_forwards"])
    predictions = {
        "pred_a_exact_task_matched_instrument": bool(exact),
        "pred_b_head17_2_terms_live": bool(exact and head_ratio >= 0.05),
        "pred_c_earlier_residual_self_dominant": bool(exact and ranking[0] == "ee" and reports["ee"]["normalized_change_norm_ratio"] >= 0.50),
        "pred_d_earlier_residual_mlp16_cross_live": bool(exact and reports["ep"]["normalized_change_norm_ratio"] >= 0.05),
    }
    result = {"schema": "setting2_regional_four_source_term_census_v1_result",
              "terminal": "valid_task_matched_term_census" if exact else "invalid",
              "predictions": predictions,
              "metrics": {"raw_sum_relative_error": rel(raw_sum, direct),
                          "normalized_sum_relative_error": rel(normalized_sum, normalized_direct),
                          "normalized_change_cancellation_ratio": float(delta.norm() / sum(v.norm() for v in delta_terms.values()).clamp_min(1e-30)),
                          "head17_2_aggregate_change_norm_ratio": head_ratio,
                          "pair_count": len(delta), "physical_prefix_forwards": forward_count,
                          "sequences": len(rows)},
              "term_reports": reports, "frozen_normalized_change_ranking": ranking,
              "source_shapes": {name: list(value.shape) for name, value in sources.items()},
              "price": PRICE, "row_checks": planned["row_checks"],
              "outcome_access": {"behavioral_logits": False, "new_rows": False, "fits": False},
              "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Task-matched regional MLP17 numerator source-term census with native shared input denominator. No behavior/logit outcome, causal identification, source compression, OOD, or adoption claim."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": result["terminal"], "predictions": predictions,
                      "metrics": result["metrics"], "ranking": ranking,
                      "term_ratios": {k: v["normalized_change_norm_ratio"] for k, v in reports.items()}}, indent=2))
    assert exact


if __name__ == "__main__":
    main()
