#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_propagated_source_instrument pred_b_top5_source_concentration pred_c_earlier_attention_source_live pred_d_stable_leading_source
"""Fold every propagated pre-L17 source against MLP16 at regional readers."""
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
PREREG = P / "SETTING2_REGIONAL_MLP16_UPSTREAM_SOURCE_FOLD_V1_PREREGISTRATION.md"
ROWS = P / "FIRST_TOKEN_PATH_FRESH_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_MLP16_UPSTREAM_SOURCE_FOLD_V1_BINDING.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_mlp16_upstream_source_fold_v1_result.json"
PRICE = {"physical_prefix_forwards": 14, "sequences": 96, "captured_module_writes": 34,
         "fits": 0, "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_exact_propagated_source_instrument": None,
                       "pred_b_top5_source_concentration": None,
                       "pred_c_earlier_attention_source_live": None,
                       "pred_d_stable_leading_source": None}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def propagated_coefficients(lambdas: torch.Tensor) -> tuple[float, dict[str, float]]:
    """Exact coefficients of x0 and block writes in raw block-17 input."""
    assert lambdas.shape == (18, 2)
    embed = 1.0
    for layer in range(17):
        embed = float(lambdas[layer, 0]) * embed + float(lambdas[layer, 1])
    embed = float(lambdas[17, 0]) * embed + float(lambdas[17, 1])
    coefficients = {}
    for layer in range(17):
        coefficient = float(torch.prod(lambdas[layer + 1:18, 0]))
        coefficients[f"attn{layer}"] = coefficient
        coefficients[f"mlp{layer}"] = coefficient
    return embed, coefficients


def load_bound() -> tuple[dict, list[dict]]:
    binding = json.loads(BINDING.read_text())
    files = {"preregistration": PREREG, "rows": ROWS,
             "row_check": P / "regional_cue_row_check_v1.py",
             "parent_result": ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_four_source_term_census_v1_result.json"}
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    parent = json.loads(files["parent_result"].read_text())
    if parent["terminal"] != "valid_task_matched_term_census" or not parent["predictions"]["pred_d_earlier_residual_mlp16_cross_live"]:
        raise ValueError("parent ep path is not live")
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
    return {"schema": "setting2_regional_mlp16_upstream_source_fold_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "source_count": 34, "price": PRICE,
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
    writes = {f"attn{layer}": torch.empty(96, 1152, dtype=torch.float64) for layer in range(17)}
    writes.update({f"mlp{layer}": torch.empty(96, 1152, dtype=torch.float64) for layer in range(17)})
    x0_saved = torch.empty(96, 1152, dtype=torch.float64)
    raw17_saved = torch.empty_like(x0_saved)
    pre17_saved = torch.empty_like(x0_saved)
    context = {"selected": None}

    def attention_hook(layer):
        def save(module, args, output):
            writes[f"attn{layer}"][context["selected"]] = output[0][:, -1].double().cpu()
        return save

    def mlp_hook(layer):
        def save(module, args, output):
            writes[f"mlp{layer}"][context["selected"]] = output[:, -1].double().cpu()
        return save

    handles = []
    for layer in range(17):
        handles.append(model.transformer.h[layer].attn.register_forward_hook(attention_hook(layer)))
        handles.append(model.transformer.h[layer].mlp.register_forward_hook(mlp_hook(layer)))
    forward_count = 0
    try:
        for _, indices in sorted(bound["buckets"].items()):
            for offset in range(0, len(indices), 8):
                selected = indices[offset:offset + 8]
                context["selected"] = selected
                tokens = torch.tensor([rows[i]["ids"] for i in selected], device=device)
                x = F.rms_norm(model.transformer.wte(tokens), (1152,))
                x0, first_value = x, None
                x0_saved[selected] = x0[:, -1].double().cpu()
                for block in model.transformer.h[:17]:
                    x, first_value = block(x, first_value, x0)
                last = model.transformer.h[17]
                raw17 = last.lambdas[0] * x + last.lambdas[1] * x0
                attention17, first_value = last.attn(F.rms_norm(raw17, (1152,)), first_value)
                raw17_saved[selected] = raw17[:, -1].double().cpu()
                pre17_saved[selected] = (raw17 + attention17)[:, -1].double().cpu()
                forward_count += 1
    finally:
        for handle in handles:
            handle.remove()

    state = model.state_dict()
    lambdas = torch.stack([block.lambdas.detach().double().cpu() for block in model.transformer.h])
    embed_coefficient, coefficients = propagated_coefficients(lambdas)
    propagated = {"embedding": embed_coefficient * x0_saved}
    propagated.update({name: coefficients[name] * value for name, value in writes.items()})
    reconstruction = sum(propagated.values())
    source_reconstruction_error = rel(reconstruction, raw17_saved)
    p = propagated.pop("mlp16")
    e = sum(propagated.values())
    e_partition_error = rel(e + p, raw17_saved)

    left, right, down = [state[f"transformer.h.17.mlp.{name}.weight"].double().cpu()
                         for name in ("Left", "Right", "Down")]
    unembed = state["lm_head.weight"]
    readers = torch.stack([unembed[row["uk_id"]] - unembed[row["us_id"]] for row in rows]).double().cpu()
    folded_down = readers @ down
    lp, rp = p @ left.T, p @ right.T
    denominator = pre17_saved.square().mean(-1) + torch.finfo(torch.float32).eps

    def cross(source: torch.Tensor) -> torch.Tensor:
        return (((source @ left.T) * rp + lp * (source @ right.T)) * folded_down).sum(-1) / denominator

    terms = {name: cross(value) for name, value in propagated.items()}
    direct = cross(e)
    term_sum = sum(terms.values())
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
    top5 = ranking[:5]
    top5_replay_error = rel(sum(delta_terms[name] for name in top5), delta)
    instrument = (forward_count == 14 and source_reconstruction_error <= 1e-5
                  and e_partition_error <= 1e-5 and rel(term_sum, direct) <= 1e-8)
    attention_max = max(reports[f"attn{layer}"]["change_norm_ratio"] for layer in range(17))
    leading = reports[ranking[0]]
    predictions = {"pred_a_exact_propagated_source_instrument": bool(instrument),
                   "pred_b_top5_source_concentration": bool(instrument and top5_replay_error <= 0.50),
                   "pred_c_earlier_attention_source_live": bool(instrument and attention_max >= 0.10),
                   "pred_d_stable_leading_source": bool(instrument and leading["change_norm_ratio"] >= 0.15
                                                         and min(leading["family_change_norm_ratios"].values()) >= 0.05)}
    result = {"schema": "setting2_regional_mlp16_upstream_source_fold_v1_result",
              "terminal": "valid_upstream_source_fold" if instrument else "invalid",
              "predictions": predictions,
              "metrics": {"source_reconstruction_relative_error": source_reconstruction_error,
                          "e_plus_p_partition_relative_error": e_partition_error,
                          "source_cross_sum_relative_error": rel(term_sum, direct),
                          "top5_replay_relative_error": top5_replay_error,
                          "attention_source_max_change_norm_ratio": attention_max,
                          "change_cancellation_ratio": float(delta.norm() / sum(v.norm() for v in delta_terms.values()).clamp_min(1e-30)),
                          "physical_prefix_forwards": forward_count, "sequences": len(rows), "pair_count": len(delta)},
              "embed_coefficient": embed_coefficient, "propagation_coefficients": coefficients,
              "source_reports": reports, "frozen_source_ranking": ranking, "top5_sources": top5,
              "price": PRICE, "row_checks": planned["row_checks"],
              "outcome_access": {"behavioral_logits": False, "new_rows": False, "fits": False},
              "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Exact propagated embedding/module-write decomposition of the task-matched regional earlier-residual x MLP16 MLP17 numerator path. Descriptive source ranking only; no behavior, causal identification, OOD, compression, or adoption claim."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": result["terminal"], "predictions": predictions,
                      "metrics": result["metrics"], "top10": ranking[:10],
                      "top10_ratios": {name: reports[name]["change_norm_ratio"] for name in ranking[:10]}}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
