#!/usr/bin/env python3
# BQGATE: frozen predictions; head sets (v9 receipt), cue pairs, DAS rank 1 / steps / lr / seed / inertness weight, bars fixed before the run.
"""v18: does v17's transfer come from the ESTIMATOR or from fitting ACROSS cue pairs?

v17 (`unit_das_common_axis_v17_result.json`): block DAS rank 1 + complement inertness fit on three cue
pairs serves an unseen fourth pair in band on all four tested behaviours (dative 0.71, polarity 0.99,
quantifier 1.13, complementizer 0.73), where single-pair DIFF-IN-MEANS directions served it only
0.12-0.53 (v14/v15). Two readings: (i) the shared axis is there in every single pair and diff-in-means
is just a poor estimator of it (its delta carries a large cue-specific part); (ii) only fitting across
pairs finds the shared axis, because a single pair's margin is served equally well by many
directions and the optimizer picks a cue-specific one. This run fits the SAME DAS (rank 1 per block,
complement weight 1.0, 120 steps, lr 0.05, seed 0) on EACH single pair's even rows (three fits per
behaviour) and tests every fit on the unseen fourth pair; the v17 cross-pair fractions are read from
its receipt for the comparison, not refit.
  Band [0.50, 1.20]; complement <= 0.30; exact bar 0.50 (fourth pair untested below it); gain 0.15;
  cos bar 0.50.

REGISTERED BEFORE THE RUN
    pred_a_single_das_transfers   >= half of the single-pair DAS fits on tested behaviours serve the
                                  fourth pair in band with complement <= 0.30 (reading i).
                                  Worked example: 7 of 12 -> True; 4 of 12 -> False.
    pred_b_cross_pair_advantage   on >= 3 tested behaviours v17's cross-pair fraction on the fourth pair
                                  exceeds the BEST single-pair DAS fraction by >= 0.15 (reading ii).
                                  Worked example: 0.99 vs 0.71 (x3) -> True; 0.99 vs 0.95 -> False.
    pred_c_das_beats_dim          >= half of the single-pair DAS fits exceed the same pair's diff-in-means
                                  fraction on the fourth pair by >= 0.15 (the estimator was the limit).
                                  Worked example: 0.65 vs 0.30 on 8 of 12 -> True.
    pred_d_single_axes_agree      on >= 3 tested behaviours the median per-block |cos| among the three
                                  single-pair DAS axes is >= 0.50 (v14's diff-in-means pairs: 0.12-0.63).
                                  Worked example: medians 0.7, 0.6, 0.55 -> True; 0.4 -> False.
    pred_e_estimator_safe         no single-pair DAS fraction >= 1.20 on any fourth-pair battery.
                                  Worked example: max 1.10 -> True.
    Priors. Unsure between i and ii; v16's quantifier single-block gaps suggest single-pair fits can
    find steep directions, so e is the one I would not bet on. c expected. If a and c hold and b fails,
    v13/v14's "cue-keyed" is retracted as an estimator artifact outright; if b holds and a fails, the
    cross-pair fit is doing real work and v13/v14 stand as single-pair facts.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15   # SETS: units, fitted maps, fourth map (unchanged since v15)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_single_pair_das_v18_result.json"
V17 = ROOT / "circuits/followups/unit_das_common_axis_v17_result.json"
SETS = v15.SETS
LO, HI, COMP_BAR, EXACT_BAR, GAIN, COS_BAR, NEED = 0.50, 1.20, 0.30, 0.50, 0.15, 0.50, 3
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2400, 40000


def _plan():
    return {"candidate_id": "corpus.unit_single_pair_das_v18",
            "sets": {k: {"units": v[1], "fitted": v[2], "fourth": v[3]} for k, v in SETS.items()},
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 15 * STEPS, "model_updates": 0, "fit_parameters": 15 * 5 * 128,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _band(b):
    return b["subspace_fraction"] is not None and LO <= b["subspace_fraction"] <= HI \
        and abs(b["complement_fraction"]) <= COMP_BAR


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    v17 = json.loads(V17.read_text())["sets"]
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    report, fits = {}, []
    for name, (module, units, maps, fourth_map) in SETS.items():
        a1 = g.rows_of(module, "A1")
        fitted = {"orig": a1, **{f"v{k + 1}": g.lexical_variant(a1, mp) for k, mp in enumerate(maps)}}
        p_fit = {k: g.prepare(backend, rows[0::2], valid_only=True) for k, rows in fitted.items()}
        p_fourth = g.prepare(backend, g.lexical_variant(a1, fourth_map), valid_only=True)
        r1 = g.block_random_subspace(backend, units, rank=1, seed=1)
        q_das, dim, hist = {}, {}, {}
        for k in fitted:
            q_das[k], hist[k] = g.fit_block_subspace(backend, p_fit[k], units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW)
            dim[k] = g.block_diff_in_means(backend, p_fit[k], units)
        on_fourth = {k: {"das": g.block_direction_battery(backend, p_fourth, units, q_das[k], r1),
                         "dim": g.block_direction_battery(backend, p_fourth, units, dim[k], r1)} for k in fitted}
        exact = next(iter(on_fourth.values()))["das"]["exact_set"]
        tested = exact >= EXACT_BAR
        keys = list(fitted)
        cos = {f"{a}_{b}": g.block_cosines(q_das[a], q_das[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}
        med = statistics.median(x for c in cos.values() for x in c.values())
        cross = v17[name]["on_fourth"]["das"]["subspace_fraction"]
        best_single = max(on_fourth[k]["das"]["subspace_fraction"] or 0 for k in fitted)
        entry = {"units": units, "fitted": maps, "fourth": fourth_map,
                 "dropped": {"fourth": p_fourth.dropped, **{f"{k}_fit": v.dropped for k, v in p_fit.items()}},
                 "exact_fourth": exact, "tested": tested, "on_fourth": on_fourth,
                 "cross_pair_fraction_v17": cross, "best_single_das_fraction": best_single,
                 "cross_advantage": (cross or 0) - best_single, "das_axis_cos": cos, "median_das_cos": med,
                 "das_loss_history": hist}
        report[name] = entry
        for k in fitted:
            fits.append({"behaviour": name, "pair": k, "tested": tested,
                         "das_band": _band(on_fourth[k]["das"]),
                         "das_fraction": on_fourth[k]["das"]["subspace_fraction"],
                         "dim_fraction": on_fourth[k]["dim"]["subspace_fraction"],
                         "gain": (on_fourth[k]["das"]["subspace_fraction"] or 0) - (on_fourth[k]["dim"]["subspace_fraction"] or 0)})
        print(name, "exact4", round(exact, 2), "cross", round(cross or 0, 2),
              {k: (round(v["das"]["subspace_fraction"] or 0, 2), round(v["das"]["complement_fraction"] or 0, 2),
                   "dim", round(v["dim"]["subspace_fraction"] or 0, 2)) for k, v in on_fourth.items()},
              "median_cos", round(med, 2), flush=True)

    tf = [f for f in fits if f["tested"]]
    tb = [e for e in report.values() if e["tested"]]
    predictions = {
        'pred_a_single_das_transfers': bool(tf) and sum(f["das_band"] for f in tf) * 2 >= len(tf),
        'pred_b_cross_pair_advantage': sum(e["cross_advantage"] >= GAIN for e in tb) >= NEED,
        'pred_c_das_beats_dim': bool(tf) and sum(f["gain"] >= GAIN for f in tf) * 2 >= len(tf),
        'pred_d_single_axes_agree': sum(e["median_das_cos"] >= COS_BAR for e in tb) >= NEED,
        'pred_e_estimator_safe': bool(tf) and all((f["das_fraction"] or 0) < HI for f in tf),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_single_pair_das_result_v1",
              "candidate_id": "corpus.unit_single_pair_das_v18", "semantics": "block_live",
              "bars": {"band": [LO, HI], "complement": COMP_BAR, "exact": EXACT_BAR, "gain": GAIN, "cos": COS_BAR,
                       "need": NEED, "das": {"rank": 1, "steps": STEPS, "lr": LR, "seed": 0, "complement_weight": CW}},
              "tested": [n for n, e in report.items() if e["tested"]], "fits": fits, "sets": report,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "tested": result["tested"],
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
