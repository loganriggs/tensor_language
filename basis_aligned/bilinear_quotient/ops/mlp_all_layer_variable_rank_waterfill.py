"""RUNG 375 -- ALL-LAYER VARIABLE-RANK CONTEXT-TAIL WATER-FILL.

Use the independently frozen MLP tail exponent and fit-A layer gains to
predict fit-B damage, then compare distributed p1024/p896/p768 allocations
with the existing two-p768 construction at exactly matched saving.

Frozen predictions
------------------
pred_a: fit-B median factor<=4, Spearman>=.60, rank order agrees >=14/18.
pred_b: exact-four optimum uses >=3 layers and <=80% comparator damage.
pred_c: exact-five optimum predicts <= comparator damage, with exact prices.

Null: fit-B median factor>8 or exact-four >=95% comparator damage.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp_all_layer_variable_rank_waterfill_results.json"
SCREEN = ROOT / "mlp_all_layer_context_metric_shared_input_screen_results.json"
CACHE = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
EXPONENT = 1.0823445857830927
RANKS = (512, 640, 768, 896)
FIT_A = (0, 24)
FIT_B = (24, 48)
D = 1152
H = 4608
UNIT = 1_327_104


def _damage(arm: dict) -> float:
    return max(1e-4, max(float(arm["fineweb_damage"]), float(arm["wikitext_damage"])))


def _spearman(a: list[float], b: list[float]) -> float:
    x = torch.tensor(a, dtype=torch.float64)
    y = torch.tensor(b, dtype=torch.float64)
    rx = torch.argsort(torch.argsort(x)).double()
    ry = torch.argsort(torch.argsort(y)).double()
    return float(torch.corrcoef(torch.stack((rx, ry)))[0, 1])


def _allocate(predicted: dict[int, dict[int, float]], units: int) -> dict:
    options = ((1152, 0), (896, 1), (768, 2), (640, 3), (512, 4))
    dp = {0: (0.0, [])}
    for layer in range(18):
        nxt = {}
        for used, (damage, choices) in dp.items():
            for rank, cost in options:
                total = used + cost
                if total > units:
                    continue
                candidate = (damage + (0.0 if rank == 1152 else predicted[layer][rank]),
                             choices + [[layer, rank]])
                if total not in nxt or candidate[0] < nxt[total][0]:
                    nxt[total] = candidate
        dp = nxt
    damage, choices = dp[units]
    installed = [[layer, rank] for layer, rank in choices if rank < 1152]
    return {"saving_units": units, "saving_scalars": units * UNIT,
            "predicted_component_damage": damage, "installed": installed,
            "installed_layers": len(installed)}


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert SCREEN.exists() and CACHE.exists()
        assert 2 * H * D - (D + 2 * H) * 896 == UNIT
        assert 517_067_062 - 5 * UNIT == 510_431_542
        print("MLP VARIABLE-RANK WATERFILL | dry run: receipts, ranks, prices, bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import mlp_all_layer_context_metric_shared_input_screen as M
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    from tier2_model import load_elriggs

    cached = torch.load(CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    rows_a = cached[FIT_A[0]:FIT_A[1], :257].long().contiguous()
    rows_b = cached[FIT_B[0]:FIT_B[1], :257].long().contiguous()
    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and len(model.transformer.h) == 18
    covariances = {"a": M._covariances(model, rows_a, _manual_logits),
                   "b": M._covariances(model, rows_b, _manual_logits)}
    tails = {"a": {}, "b": {}}
    for split in ("a", "b"):
        for layer in range(18):
            covariance = covariances[split][layer]
            values_c, vectors_c = torch.linalg.eigh(covariance)
            floor = float(values_c[-1]) * 1e-6
            sqrt = (vectors_c * values_c.clamp_min(floor).sqrt()) @ vectors_c.T
            mlp = model.transformer.h[layer].mlp
            stacked = torch.cat((mlp.Left.weight.detach().float(),
                                 mlp.Right.weight.detach().float()), dim=0)
            metric = sqrt @ (stacked.T @ stacked) @ sqrt
            values = torch.linalg.eigvalsh(.5 * (metric + metric.T)).flip(0).clamp_min(0)
            tails[split][layer] = {rank: float(values[rank:].sum()) for rank in RANKS}
            print(f"{split} L{layer:02d} tails " + " ".join(
                f"p{rank}={tails[split][layer][rank]:.4g}" for rank in RANKS), flush=True)
    screen = json.loads(SCREEN.read_text())
    predicted, actual_b, predicted_b = {}, [], []
    order_agree = 0
    layer_rows = {}
    for layer in range(18):
        arms = screen["arms"][str(layer)]
        y_a = {rank: _damage(arms[f"context_rrr_fit_a_p{rank}"]) for rank in (512, 768)}
        gains = [y_a[rank] / max(tails["a"][layer][rank], 1e-30) ** EXPONENT
                 for rank in (512, 768)]
        gain = math.sqrt(gains[0] * gains[1])
        predicted[layer] = {rank: gain * max(tails["b"][layer][rank], 1e-30) ** EXPONENT
                            for rank in RANKS}
        y_b = {rank: _damage(arms[f"context_rrr_fit_b_p{rank}"]) for rank in (512, 768)}
        for rank in (512, 768):
            actual_b.append(y_b[rank])
            predicted_b.append(predicted[layer][rank])
        order_agree += int(y_b[768] <= y_b[512])
        layer_rows[str(layer)] = {"fit_a_damage": y_a, "fit_b_damage": y_b,
                                  "fit_a_geomean_gain": gain,
                                  "predicted_fit_b": predicted[layer]}
    log_errors = [abs(math.log(p / y)) for p, y in zip(predicted_b, actual_b)]
    median_factor = math.exp(float(torch.tensor(log_errors).median()))
    rho = _spearman(predicted_b, actual_b)
    exact4 = _allocate(predicted, 4)
    exact5 = _allocate(predicted, 5)
    comparator = predicted[0][768] + predicted[4][768]
    ratio4 = exact4["predicted_component_damage"] / comparator
    ratio5 = exact5["predicted_component_damage"] / comparator
    pred_a = median_factor <= 4.0 and rho >= .60 and order_agree >= 14
    pred_b = exact4["installed_layers"] >= 3 and ratio4 <= .80
    pred_c = (ratio5 <= 1.0 and exact5["saving_scalars"] == 6_635_520
              and 517_067_062 - exact5["saving_scalars"] == 510_431_542)
    null = median_factor > 8.0 or ratio4 >= .95
    result = {
        "status": "mlp_all_layer_variable_rank_waterfill_complete",
        "rung": 375,
        "claim_level": "all_layer_tail_calibrated_variable_rank_allocator_screen",
        "fixed_exponent_source": "rung355 MLP0 induced-metric law",
        "fixed_exponent": EXPONENT,
        "damage_target": "max(1e-4, max(FineWeb, WikiText CE added))",
        "fit_rows": {"cache": CACHE.name, "fit_a": list(FIT_A), "fit_b": list(FIT_B)},
        "ranks": list(RANKS), "saving_unit_scalars": UNIT,
        "fit_b_median_multiplicative_error": median_factor,
        "fit_b_spearman": rho, "fit_b_rank_order_agree_layers": order_agree,
        "comparator_mlp04_p768_predicted_damage": comparator,
        "exact4": exact4, "exact4_damage_ratio_vs_mlp04": ratio4,
        "exact5": exact5, "exact5_damage_ratio_vs_mlp04": ratio5,
        "layers": layer_rows,
        'pred_a_tail_model_transfers_to_fit_b': bool(pred_a),
        'pred_b_exact_four_spreading_beats_two_deep_cuts': bool(pred_b),
        'pred_c_exact_five_candidate_beats_comparator_at_lower_price': bool(pred_c),
        "null_variable_rank_waterfill_not_useful": bool(null),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("layers",)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
