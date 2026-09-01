"""RUNG 358 -- MDL DEPLOYMENT CROSSOVER LAW FOR THE GATED QK FRONTIER.

Express semantic storage and future predictive loss in the same unit.  For a
program with scalar count S and census damage D nats/token, its incremental
description length relative native is

    L_b(N) = b (S-S_native) + N D / ln(2)  bits,

under a hypothetical uniform b-bit representation.  Literal tensor bytes are
audited separately.  Quantized b=16/8 prices are hypothetical: this rung does
not claim unchanged physical CE after quantization.

Frozen predictions
------------------
pred_a_every_gated_qk_point_has_an_mdl_interval:
    Native and all six gated QK ranks96..56 each occupy a nonempty exact lower-
    envelope interval for b in {32,16,8}; no adopted rung is skipped.
pred_b_break_even_schedule_is_coherently_convex:
    All adjacent token crossovers are positive and strictly increase while
    traversing rank56 -> ... -> rank96 -> native (equivalently they decrease
    toward the lower-rank edge).
pred_c_literal_bytes_match_uniform_fp32_and_r48_is_only_mapped:
    Literal-byte crossovers equal uniform32 within numerical tolerance, and
    rank48 is kept in a separately labeled mapped-tier envelope.

Null: any adopted point is never optimal or any adjacent crossover is <=0.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "qk_frontier_mdl_crossover_results.json"
LN2 = math.log(2.0)
BITS = (32, 16, 8)
NATIVE = {"name": "native", "rank": None, "scalars": 545_902_902,
          "raw_tensor_bytes": 2_067_669_612, "damage": 0.0,
          "certificates": 62, "status": "native"}
FILES = {
    96: "mixed96_context_metric_qk_split_ood_results.json",
    88: "mixed88_context_metric_qk_ood_results.json",
    80: "mixed80_context_metric_qk_ood_results.json",
    72: "mixed72_context_metric_qk_ood_results.json",
    64: "mixed64_context_metric_qk_ood_results.json",
    56: "mixed56_context_metric_qk_newcorpus_ood_results.json",
    48: "mixed48_context_metric_qk_newcorpus_ood_results.json",
}


def _line(point, n_tokens: float, bits_per_scalar: int | None):
    if bits_per_scalar is None:
        storage = 8.0 * (point["raw_tensor_bytes"] - NATIVE["raw_tensor_bytes"])
    else:
        storage = float(bits_per_scalar) * (point["scalars"] - NATIVE["scalars"])
    return storage + n_tokens * point["damage"] / LN2


def _crossing(smaller, larger, bits_per_scalar: int | None):
    """Token N where the smaller/higher-damage and larger/lower-damage lines meet."""
    if bits_per_scalar is None:
        saved_bits = 8.0 * (larger["raw_tensor_bytes"] - smaller["raw_tensor_bytes"])
    else:
        saved_bits = float(bits_per_scalar) * (larger["scalars"] - smaller["scalars"])
    damage_gap = smaller["damage"] - larger["damage"]
    assert saved_bits > 0 and damage_gap > 0
    return saved_bits * LN2 / damage_gap


def _schedule(points_small_to_large, bits_per_scalar: int | None):
    crossings = [_crossing(points_small_to_large[index], points_small_to_large[index + 1],
                           bits_per_scalar)
                 for index in range(len(points_small_to_large) - 1)]
    intervals = []
    for index, point in enumerate(points_small_to_large):
        lo = 0.0 if index == 0 else crossings[index - 1]
        hi = None if index == len(points_small_to_large) - 1 else crossings[index]
        probe = (lo + 1.0) * 2.0 if hi is None else ((lo + hi) / 2.0)
        values = [_line(candidate, probe, bits_per_scalar) for candidate in points_small_to_large]
        winner = points_small_to_large[min(range(len(values)), key=values.__getitem__)]["name"]
        intervals.append({"point": point["name"], "tokens_lo_inclusive": lo,
                          "tokens_hi_exclusive": hi, "midpoint_winner": winner,
                          "nonempty": hi is None or hi > lo})
    return crossings, intervals


def main() -> None:
    needed = [ROOT / name for name in FILES.values()]
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in needed)
        assert sorted(FILES) == [48, 56, 64, 72, 80, 88, 96]
        print("QK FRONTIER MDL | dry run: receipts, hypotheses, bars valid")
        return

    started = time.time()
    points = {}
    for rank, filename in FILES.items():
        receipt = json.loads((ROOT / filename).read_text())
        points[rank] = {
            "name": f"qk{rank}",
            "rank": rank,
            "scalars": int(receipt["literal_standalone_scalars"]),
            "raw_tensor_bytes": int(receipt["literal_raw_tensor_bytes"]),
            "damage": float(receipt["census_damage"]),
            "certificates": int(receipt["certificates_valid"]),
            "status": "fully_gated_adopted" if rank >= 56 else "mapped_certificate_ledge",
        }

    adopted_small_to_large = [points[rank] for rank in (56, 64, 72, 80, 88, 96)] + [NATIVE]
    including_mapped_small_to_large = [points[48]] + adopted_small_to_large
    schedules = {}
    all_nonempty = True
    all_increasing = True
    for bits in BITS:
        cross, intervals = _schedule(adopted_small_to_large, bits)
        schedules[str(bits)] = {"adjacent_crossovers_tokens": cross, "intervals": intervals}
        all_nonempty = all_nonempty and all(row["nonempty"] and row["midpoint_winner"] == row["point"]
                                             for row in intervals)
        all_increasing = all_increasing and all(cross[index + 1] > cross[index]
                                                for index in range(len(cross) - 1))

    raw_cross, raw_intervals = _schedule(adopted_small_to_large, None)
    schedules["literal_raw_bytes"] = {"adjacent_crossovers_tokens": raw_cross,
                                      "intervals": raw_intervals}
    fp32_cross = schedules["32"]["adjacent_crossovers_tokens"]
    raw_matches_fp32 = all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-3)
                           for a, b in zip(raw_cross, fp32_cross))

    mapped_schedules = {}
    for bits in BITS:
        cross, intervals = _schedule(including_mapped_small_to_large, bits)
        mapped_schedules[str(bits)] = {"adjacent_crossovers_tokens": cross,
                                      "intervals": intervals,
                                      "warning": "qk48 has only 29/62 certificates and is mapped, not adopted"}

    deployment_sizes = [1_000_000, 1_000_000_000, 10_000_000_000,
                        100_000_000_000, 1_000_000_000_000]
    selections = {}
    for bits in BITS:
        selections[str(bits)] = {}
        for n_tokens in deployment_sizes:
            values = {point["name"]: _line(point, n_tokens, bits)
                      for point in adopted_small_to_large}
            selections[str(bits)][str(n_tokens)] = min(values, key=values.get)

    pred_a = all_nonempty
    pred_b = all_increasing and all(value > 0 for bits in BITS
                                    for value in schedules[str(bits)]["adjacent_crossovers_tokens"])
    pred_c = raw_matches_fp32 and points[48]["certificates"] < 43 and points[48]["status"].startswith("mapped")
    null = (not all_nonempty) or any(value <= 0 for value in fp32_cross)
    result = {
        "status": "qk_frontier_mdl_crossover_complete",
        "rung": 358,
        "claim_level": "cpu_exact_storage_predictive_codelength_tradeoff",
        "formula": "incremental bits = b*(S-S_native) + N*damage_nats/ln(2)",
        "assumptions": [
            "census damage is the expected future per-token penalty",
            "uniform 32/16/8-bit cases assume quantization does not alter damage and are hypothetical",
            "literal-byte case uses measured tensor bytes and is physically priced storage only",
            "compute, latency, training cost, and certificate utility are not converted to bits",
        ],
        "native": NATIVE,
        "qk_points": [points[rank] for rank in sorted(points, reverse=True)],
        "fully_gated_schedules": schedules,
        "fixed_deployment_selections": selections,
        "mapped_rank48_schedules": mapped_schedules,
        "literal_bytes_equal_uniform_fp32": raw_matches_fp32,
        'pred_a_every_gated_qk_point_has_an_mdl_interval': bool(pred_a),
        'pred_b_break_even_schedule_is_coherently_convex': bool(pred_b),
        'pred_c_literal_bytes_match_uniform_fp32_and_r48_is_only_mapped': bool(pred_c),
        "null_adopted_point_is_mdl_dominated_or_crossing_nonpositive": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "uniform16_crossovers_tokens_rank56_to_native": schedules["16"]["adjacent_crossovers_tokens"],
        "uniform32_crossovers_tokens_rank56_to_native": schedules["32"]["adjacent_crossovers_tokens"],
        "uniform8_crossovers_tokens_rank56_to_native": schedules["8"]["adjacent_crossovers_tokens"],
        "selections": selections,
        "raw_matches_fp32": raw_matches_fp32,
        "predicates": [pred_a, pred_b, pred_c],
        "null": null,
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("QK FRONTIER MDL CROSSOVER DONE", flush=True)


if __name__ == "__main__":
    main()
