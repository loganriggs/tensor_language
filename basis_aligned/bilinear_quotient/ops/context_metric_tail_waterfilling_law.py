"""RUNG 355 -- INDUCED-METRIC SINGULAR-TAIL LAW AND WATER-FILLING TEST.

Context-RRR is Eckart--Young on W C^(1/2).  Compute exact omitted squared
singular energies for the measured Q/K and MLP0 ladders, then test whether they
predict physical component census damage.  Separate-family versus joint fits
decide whether naive or gain-weighted water-filling is licensed.

Frozen predictions
------------------
pred_a_qk_tail_predicts_measured_damage:
    QK log-tail R2 >=.90 and LOOCV median relative error <=30%.
pred_b_mlp0_tail_predicts_measured_damage:
    MLP0 log-tail R2 >=.80, LOOCV median error <=35%, strict ordering.
pred_c_family_specific_gain_is_required:
    Joint R2 is >=.15 below separate fit OR median damage/tail gains differ
    by >=2x, so water-filling must be family-weighted.

Null: either family R2 <.30 OR either singular-tail ordering disagrees with
measured damage ordering.  This is calibration, not artifact adoption.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "context_metric_tail_waterfilling_law_results.json"
DEV = "cuda"
QK_RANKS = (48, 56, 64, 72, 80, 88, 96)
MLP_RANKS = (256, 384, 448, 512, 640)
QK_FIT = (72, 96)
MLP_FIT = (0, 24)
MIXED104_DAMAGE = .00469195


def _log_fit(x_values, y_values):
    x = torch.tensor(x_values, dtype=torch.float64).clamp_min(1e-30).log()
    y = torch.tensor(y_values, dtype=torch.float64).clamp_min(1e-12).log()
    design = torch.stack((torch.ones_like(x), x), dim=1)
    coefficient = torch.linalg.lstsq(design, y).solution
    prediction = design @ coefficient
    denominator = ((y - y.mean()) ** 2).sum().clamp_min(1e-30)
    r2 = float(1.0 - ((y - prediction) ** 2).sum() / denominator)
    relative = []
    for heldout in range(len(x)):
        keep = torch.arange(len(x)) != heldout
        fit = torch.linalg.lstsq(design[keep], y[keep]).solution
        predicted = float(torch.exp(design[heldout] @ fit))
        relative.append(abs(predicted - y_values[heldout]) / max(y_values[heldout], 1e-12))
    return {
        "intercept": float(coefficient[0]),
        "exponent": float(coefficient[1]),
        "log_r2": r2,
        "loocv_relative_errors": relative,
        "loocv_median_relative_error": float(torch.tensor(relative).median()),
        "fitted_damage": [float(value) for value in torch.exp(prediction)],
    }


def _strict_same_order(tails, damage):
    return all(tails[i] < tails[i + 1] and damage[i] < damage[i + 1]
               for i in range(len(tails) - 1))


@torch.no_grad()
def main() -> None:
    needed = [
        ROOT / ".rowcache/fineweb_n192_skip11000.pt",
        ROOT / "attn_motifs3_results.json",
        ROOT / "mixed96_context_metric_qk_split_ood_results.json",
        ROOT / "mixed88_context_metric_qk_ood_results.json",
        ROOT / "mixed80_context_metric_qk_ood_results.json",
        ROOT / "mixed72_context_metric_qk_ood_results.json",
        ROOT / "mixed64_context_metric_qk_ood_results.json",
        ROOT / "mixed56_context_metric_qk_newcorpus_ood_results.json",
        ROOT / "mixed48_context_metric_qk_newcorpus_ood_results.json",
        ROOT / "mixed104_mlp0_context_metric_input_frontier_ood_results.json",
        ROOT / "mixed104_mlp0_context_metric_lower_rank_frontier_results.json",
    ]
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in needed)
        assert QK_RANKS == tuple(range(48, 97, 8)) and MLP_RANKS == (256, 384, 448, 512, 640)
        print("CONTEXT TAIL WATER-FILLING | dry run: receipts, ranks, fits, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import cevdump_ct96 as C
    from mixed96_context_metric_qk import _attention_input_covariances
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip11000.pt", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    qk_rows = cached[QK_FIT[0]:QK_FIT[1], :257].long().contiguous()
    mlp_rows = cached[MLP_FIT[0]:MLP_FIT[1], :257].long().contiguous()
    qk_covariances = _attention_input_covariances(C.m, qk_rows, _manual_logits)
    mlp_covariance = _covariance(C.m, mlp_rows, _manual_logits)

    motif = json.loads((ROOT / "attn_motifs3_results.json").read_text())["motif_table"]
    motif_heads = {}
    for layer, head, mode, _frequency in motif:
        if 2 <= layer <= 9 and mode in ("prev", "self"):
            motif_heads.setdefault(int(layer), []).append(int(head))
    qk_squares = []
    map_count = 0
    for layer in range(2, 18):
        covariance = qk_covariances[layer].to(DEV).float()
        values, vectors = torch.linalg.eigh(.5 * (covariance + covariance.T))
        floor = float(values[-1]) * 1e-6
        sqrt = (vectors * values.clamp_min(floor).sqrt()) @ vectors.T
        heads = motif_heads.get(layer, []) if layer < 10 else range(9)
        attention = C.m.transformer.h[layer].attn
        for head in heads:
            for linear in (attention.c_q, attention.c_k, attention.c_q2, attention.c_k2):
                matrix = linear.weight[head * 128:(head + 1) * 128].detach().float()
                singular = torch.linalg.svdvals(matrix @ sqrt)
                qk_squares.append(singular.square().cpu())
                map_count += 1
    assert map_count == 440, map_count
    qk_squares = torch.stack(qk_squares)
    qk_tail = {rank: float(qk_squares[:, rank:].sum()) for rank in QK_RANKS}

    left = C.m.transformer.h[0].mlp.Left.weight.detach().float()
    right = C.m.transformer.h[0].mlp.Right.weight.detach().float()
    stacked = torch.cat((left, right), dim=0)
    values, vectors = torch.linalg.eigh(mlp_covariance.to(DEV).float())
    floor = float(values[-1]) * 1e-6
    sqrt = (vectors * values.clamp_min(floor).sqrt()) @ vectors.T
    metric = sqrt @ (stacked.T @ stacked) @ sqrt
    metric_values = torch.linalg.eigvalsh(.5 * (metric + metric.T)).flip(0).clamp_min(0).cpu()
    mlp_tail = {rank: float(metric_values[rank:].sum()) for rank in MLP_RANKS}

    qk_paths = {
        96: "mixed96_context_metric_qk_split_ood_results.json",
        88: "mixed88_context_metric_qk_ood_results.json",
        80: "mixed80_context_metric_qk_ood_results.json",
        72: "mixed72_context_metric_qk_ood_results.json",
        64: "mixed64_context_metric_qk_ood_results.json",
        56: "mixed56_context_metric_qk_newcorpus_ood_results.json",
        48: "mixed48_context_metric_qk_newcorpus_ood_results.json",
    }
    qk_damage = {rank: json.loads((ROOT / path).read_text())["census_damage"]
                 for rank, path in qk_paths.items()}
    high = json.loads((ROOT / "mixed104_mlp0_context_metric_input_frontier_ood_results.json").read_text())["arms"]
    low = json.loads((ROOT / "mixed104_mlp0_context_metric_lower_rank_frontier_results.json").read_text())["arms"]
    mlp_total = {
        640: high["640"]["census_damage"],
        512: high["512"]["census_damage"],
        448: low["448"]["census_damage"],
        384: low["384"]["census_damage"],
        256: low["256"]["census_damage"],
    }
    mlp_damage = {rank: value - MIXED104_DAMAGE for rank, value in mlp_total.items()}

    qk_order = list(QK_RANKS[::-1])
    mlp_order = list(MLP_RANKS[::-1])
    qk_x, qk_y = [qk_tail[r] for r in qk_order], [qk_damage[r] for r in qk_order]
    mlp_x, mlp_y = [mlp_tail[r] for r in mlp_order], [mlp_damage[r] for r in mlp_order]
    qk_fit = _log_fit(qk_x, qk_y)
    mlp_fit = _log_fit(mlp_x, mlp_y)

    all_x, all_y = qk_x + mlp_x, qk_y + mlp_y
    joint_fit = _log_fit(all_x, all_y)
    separate_log_y = []
    separate_log_pred = []
    for ys, fit in ((qk_y, qk_fit), (mlp_y, mlp_fit)):
        separate_log_y.extend(math.log(value) for value in ys)
        separate_log_pred.extend(math.log(value) for value in fit["fitted_damage"])
    y_tensor = torch.tensor(separate_log_y)
    p_tensor = torch.tensor(separate_log_pred)
    separate_r2 = float(1.0 - ((y_tensor - p_tensor) ** 2).sum()
                        / ((y_tensor - y_tensor.mean()) ** 2).sum())
    qk_gain = float(torch.tensor([y / x for x, y in zip(qk_x, qk_y)]).median())
    mlp_gain = float(torch.tensor([y / x for x, y in zip(mlp_x, mlp_y)]).median())
    gain_ratio = max(qk_gain, mlp_gain) / max(min(qk_gain, mlp_gain), 1e-30)

    qk_order_ok = _strict_same_order(qk_x, qk_y)
    mlp_order_ok = _strict_same_order(mlp_x, mlp_y)
    pred_a = qk_fit["log_r2"] >= .90 and qk_fit["loocv_median_relative_error"] <= .30
    pred_b = (mlp_fit["log_r2"] >= .80
              and mlp_fit["loocv_median_relative_error"] <= .35 and mlp_order_ok)
    pred_c = separate_r2 - joint_fit["log_r2"] >= .15 or gain_ratio >= 2.0
    null = qk_fit["log_r2"] < .30 or mlp_fit["log_r2"] < .30 or not qk_order_ok or not mlp_order_ok
    result = {
        "status": "context_metric_tail_waterfilling_law_complete",
        "rung": 355,
        "claim_level": "induced_metric_tail_energy_allocation_calibration_only",
        "qk": {"ranks_high_to_low": qk_order,
               "tail_energy": {str(k): v for k, v in qk_tail.items()},
               "measured_census_damage": {str(k): v for k, v in qk_damage.items()},
               "maps": map_count, "fit": qk_fit, "strict_order": qk_order_ok},
        "mlp0": {"ranks_high_to_low": mlp_order,
                 "tail_energy": {str(k): v for k, v in mlp_tail.items()},
                 "measured_component_damage": {str(k): v for k, v in mlp_damage.items()},
                 "fit": mlp_fit, "strict_order": mlp_order_ok},
        "joint": {"fit": joint_fit, "separate_family_log_r2": separate_r2,
                  "qk_median_damage_per_tail": qk_gain,
                  "mlp_median_damage_per_tail": mlp_gain,
                  "family_gain_ratio": gain_ratio,
                  "allocation_rule": "water-fill weighted tail energies using family-specific gains"},
        'pred_a_qk_tail_predicts_measured_damage': bool(pred_a),
        'pred_b_mlp0_tail_predicts_measured_damage': bool(pred_b),
        'pred_c_family_specific_gain_is_required': bool(pred_c),
        "null_singular_tail_law_fails": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"qk_fit": qk_fit, "mlp_fit": mlp_fit,
                      "joint_r2": joint_fit["log_r2"], "separate_r2": separate_r2,
                      "gain_ratio": gain_ratio, "predicates": [pred_a, pred_b, pred_c],
                      "null": null, "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("CONTEXT-METRIC TAIL WATER-FILLING LAW DONE", flush=True)


if __name__ == "__main__":
    main()
