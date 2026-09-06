#!/usr/bin/env python3
# BQGATE: frozen predictions; head set (v9), cue pairs (v15), ranks, seeds, bars fixed before the run.
"""v22: does polarity's rank-4 selection (v21) replicate across DAS seeds?

v21 selected rank 4 for polarity_licensing because S + C on held-out reached 0.87 (bar 0.85) at rank 4 while
ranks 1, 2 AND 8 sat at 0.76-0.78 -- non-monotone in rank from a single seed, which is what a fit that
happened to land looks like. Same fit (complement weight 1.0, 120 steps, lr 0.05) at ranks 2, 4, 8 with
seeds 1 and 2 (seed 0 is v21's), fit on the three-pair pool even rows, S + C read on the pool's odd rows
and on the unseen fourth pair (scarcely/frequently). Bars unchanged.

REGISTERED BEFORE THE RUN
    pred_a_rank4_replicates     S + C >= 0.85 on the fourth pair at rank 4 for BOTH seeds 1 and 2.
                                Worked example: 0.91, 0.88 -> True; 0.91, 0.79 -> False.
    pred_b_rank2_insufficient   S + C < 0.85 on the fourth pair at rank 2 for BOTH seeds.
                                Worked example: 0.78, 0.80 -> True.
    pred_c_rank8_recovers       S + C >= 0.85 on the fourth pair at rank 8 for at least one seed (v21's rank-8
                                0.81 was a landing, not a ceiling). Worked example: 0.79, 0.88 -> True.
    pred_d_rank4_confirms       at rank 4 both seeds pass band [0.50, 1.20] + |complement| <= 0.30 on the fourth
                                pair. Worked example: 0.93/-0.04 and 0.97/0.02 -> True.
    pred_e_random_bar           rank-matched random <= 0.10 on the fourth pair at every rung and seed.
    Reading rule. a and b both True: polarity's variable is a rank-4 subspace and v21's selection stands.
    a False: v21's rank-4 pass was one seed landing; polarity's exact-set effect is not the additive sum of a
    low-rank subspace and its complement at any rank <= 8 -- a nonlinear read, to be characterised, not fitted.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_polarity_rank_seeds_v22_result.json"
NAME = "polarity_licensing"
RANKS, SEEDS = (2, 4, 8), (1, 2)
LO, HI, COMP_BAR, SUM_LO, SUM_HI, RAND_BAR = 0.50, 1.20, 0.30, 0.85, 1.15, 0.10
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 3000, 100000


def _plan():
    module, units, maps, fourth = v15.SETS[NAME]
    return {"candidate_id": "corpus.unit_polarity_rank_seeds_v22", "units": units, "fitted": maps, "fourth": fourth,
            "ranks": list(RANKS), "seeds": list(SEEDS), "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX, "model_backwards": len(RANKS) * len(SEEDS) * STEPS,
            "model_updates": 0, "fit_parameters": len(SEEDS) * 4 * 128 * sum(RANKS),
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    module, units, maps, fourth_map = v15.SETS[NAME]
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    a1 = g.rows_of(module, "A1")
    pairs = [a1] + [g.lexical_variant(a1, mp) for mp in maps]
    p_fit = g.prepare(backend, sum((p[0::2] for p in pairs), []), valid_only=True)
    p_sel = g.prepare(backend, sum((p[1::2] for p in pairs), []), valid_only=True)
    p_conf = g.prepare(backend, g.lexical_variant(a1, fourth_map), valid_only=True)
    grid = {}
    for seed in SEEDS:
        for r in RANKS:
            q, hist = g.fit_block_subspace(backend, p_fit, units, rank=r, steps=STEPS, lr=LR, seed=seed, complement_weight=CW)
            rnd = g.block_random_subspace(backend, units, rank=r, seed=1)
            grid[f"seed{seed}_rank{r}"] = {"seed": seed, "rank": r,
                                          "select": g.block_direction_battery(backend, p_sel, units, q, rnd),
                                          "confirm": g.block_direction_battery(backend, p_conf, units, q, rnd),
                                          "loss_history": hist}
            c = grid[f"seed{seed}_rank{r}"]["confirm"]
            print(seed, r, "sel S+C", round(grid[f"seed{seed}_rank{r}"]["select"]["linearity_sum"], 2),
                  "conf", round(c["subspace_fraction"], 2), round(c["complement_fraction"], 2), "S+C", round(c["linearity_sum"], 2),
                  "rnd", round(c["random_fraction"], 2), flush=True)

    def sums(r):
        return [grid[f"seed{s}_rank{r}"]["confirm"]["linearity_sum"] for s in SEEDS]

    def band(e):
        c = e["confirm"]
        return LO <= c["subspace_fraction"] <= HI and abs(c["complement_fraction"]) <= COMP_BAR

    predictions = {
        'pred_a_rank4_replicates': all(SUM_LO <= s <= SUM_HI for s in sums(4)),
        'pred_b_rank2_insufficient': all(s < SUM_LO for s in sums(2)),
        'pred_c_rank8_recovers': any(SUM_LO <= s <= SUM_HI for s in sums(8)),
        'pred_d_rank4_confirms': all(band(grid[f"seed{s}_rank4"]) for s in SEEDS),
        'pred_e_random_bar': all(abs(e["confirm"]["random_fraction"]) <= RAND_BAR for e in grid.values()),
    }
    reading = ("rank4_stands" if predictions['pred_a_rank4_replicates'] and predictions['pred_b_rank2_insufficient']
               else "rank4_was_a_landing_nonlinear_read" if not predictions['pred_a_rank4_replicates'] else "rank2_suffices_on_some_seed")
    result = {"predictions": predictions, "reading": reading, "schema": "circuit_unit_polarity_rank_seeds_result_v1",
              "candidate_id": "corpus.unit_polarity_rank_seeds_v22", "semantics": "block_live", "units": units,
              "bars": {"band": [LO, HI], "complement": COMP_BAR, "sum": [SUM_LO, SUM_HI], "random": RAND_BAR,
                       "das": {"steps": STEPS, "lr": LR, "complement_weight": CW, "seeds": list(SEEDS), "ranks": list(RANKS)}},
              "exact_fourth": next(iter(grid.values()))["confirm"]["exact_set"],
              "sums_by_rank": {r: sums(r) for r in RANKS}, "grid": grid,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "reading": reading, "sums_by_rank": result["sums_by_rank"],
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
