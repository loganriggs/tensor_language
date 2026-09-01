"""RUNG 379 -- VALUE-FAMILY PRICE-FIRST TAIL CLOSURE.

Compute exact context-metric c_v tails and use the measured value96 surcharge
to give every legal rank an optimistic damage estimate under the full observed
MLP/QK exponent envelope.  Compare with the adopted MLP04 exchange rate.

Frozen predictions
------------------
pred_a: tails strictly decrease; r96 map/price/surcharge identities exact.
pred_b: every optimistic value exchange is >=2x adopted MLP04 exchange.
pred_c: the best noncalibration rank is also >=2x, with exact tripwires.

Null: any rank is <=1x MLP04 exchange or tail ordering fails.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "value_context_tail_price_closure_results.json"
FIT_CACHE = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
RANKS = tuple(range(64, 113, 8))
EXPONENT_LOW = 1.0823445857830927
EXPONENT_HIGH = 1.6921
VALUE96 = ROOT / "mixed80_context_qk_value96_context_ood_results.json"
QK80 = ROOT / "mixed80_context_metric_qk_ood_results.json"
MLP04 = ROOT / "mixed64_context_qk_mlp04_context_p768_ood_results.json"
QK64 = ROOT / "mixed64_context_metric_qk_ood_results.json"
MAPS = 144


@torch.no_grad()
def main() -> None:
    needed = (FIT_CACHE, VALUE96, QK80, MLP04, QK64)
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in needed)
        assert RANKS == (64, 72, 80, 88, 96, 104, 112)
        assert MAPS * (128 * 1152 - 96 * (128 + 1152)) == 3_538_944
        print("VALUE TAIL PRICE | dry run: receipts, ranks, price, bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import cevdump_ct96 as C
    from mixed96_context_metric_qk import _attention_input_covariances
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    cached = torch.load(FIT_CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    rows = cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()
    covariances = _attention_input_covariances(C.m, rows, _manual_logits)
    squares = []
    for layer in LAYERS:
        covariance = covariances[layer].float()
        values, vectors = torch.linalg.eigh(covariance)
        floor = float(values[-1]) * 1e-6
        sqrt = (vectors * values.clamp_min(floor).sqrt()) @ vectors.T
        value = C.m.transformer.h[layer].attn.c_v.weight.detach().float()
        for head in range(9):
            matrix = value[head * 128:(head + 1) * 128]
            squares.append(torch.linalg.svdvals(matrix @ sqrt).square().cpu())
    assert len(squares) == MAPS
    squares = torch.stack(squares)
    tails = {rank: float(squares[:, rank:].sum()) for rank in RANKS}
    ordered = all(tails[RANKS[i]] > tails[RANKS[i + 1]] for i in range(len(RANKS) - 1))

    value96 = json.loads(VALUE96.read_text())
    qk80 = json.loads(QK80.read_text())
    mlp04 = json.loads(MLP04.read_text())
    qk64 = json.loads(QK64.read_text())
    measured = float(value96["census_damage"] - qk80["census_damage"])
    mlp_surcharge = float(mlp04["census_damage"] - qk64["census_damage"])
    mlp_saving = 2 * 2_654_208
    mlp_exchange = mlp_surcharge / mlp_saving
    rows_out = {}
    for rank in RANKS:
        ratio = tails[rank] / tails[96]
        exponent = EXPONENT_HIGH if ratio <= 1.0 else EXPONENT_LOW
        optimistic_damage = measured * ratio ** exponent
        saving = MAPS * (128 * 1152 - rank * (128 + 1152))
        exchange = optimistic_damage / saving
        rows_out[str(rank)] = {
            "tail_energy": tails[rank], "tail_ratio_vs_r96": ratio,
            "optimistic_exponent": exponent,
            "optimistic_predicted_damage": optimistic_damage,
            "saving_scalars": saving, "damage_per_saved_scalar": exchange,
            "exchange_ratio_vs_adopted_mlp04": exchange / mlp_exchange,
        }
    ratios = {rank: rows_out[str(rank)]["exchange_ratio_vs_adopted_mlp04"]
              for rank in RANKS}
    best_rank = min(ratios, key=ratios.get)
    non96 = {rank: ratio for rank, ratio in ratios.items() if rank != 96}
    best_non96 = min(non96, key=non96.get)
    identity = (value96["value_rank"] == 96 and value96["value_factorized_maps"] == MAPS
                and value96["literal_standalone_scalars"] == 522_539_318
                and abs(measured - value96["surcharge_vs_context_qk80"]) < 1e-9
                and FIT_SLICE == (72, 96) and LAYERS == tuple(range(2, 18)))
    pred_a = ordered and identity
    pred_b = min(ratios.values()) >= 2.0
    pred_c = non96[best_non96] >= 2.0 and identity
    null = min(ratios.values()) <= 1.0 or not ordered
    result = {
        "status": "value_context_tail_price_closure_complete",
        "rung": 379,
        "claim_level": "value_family_optimistic_tail_exchange_calibration_screen",
        "fit_cache": FIT_CACHE.name, "fit_rows_half_open": list(FIT_SLICE),
        "context_layers": list(LAYERS), "factorized_maps": MAPS,
        "ranks": list(RANKS), "exponent_envelope": [EXPONENT_LOW, EXPONENT_HIGH],
        "measured_value96_component_damage": measured,
        "adopted_mlp04_component_damage": mlp_surcharge,
        "adopted_mlp04_saving_scalars": mlp_saving,
        "adopted_mlp04_damage_per_saved_scalar": mlp_exchange,
        "arms": rows_out, "best_rank": best_rank,
        "best_exchange_ratio_vs_mlp04": ratios[best_rank],
        "best_noncalibration_rank": best_non96,
        "best_noncalibration_exchange_ratio_vs_mlp04": non96[best_non96],
        'pred_a_value_tail_and_calibration_identities_hold': bool(pred_a),
        'pred_b_no_value_rank_beats_two_x_mlp_exchange': bool(pred_b),
        'pred_c_noncalibration_value_ranks_remain_inefficient': bool(pred_c),
        "null_value_rank_can_compete_with_adopted_mlp": bool(null),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
