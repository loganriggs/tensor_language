#!/usr/bin/env python3
# BQGATE: frozen predictions; head sets (v9 receipt), cue pairs, DAS rank 1 / steps / lr / seed / inertness weight, bars fixed before the run.
"""v19: is fourth-pair transfer MONOTONE in the number of fitted cue pairs (1 -> 2 -> 3)?

v18 (`unit_two_pair_das_v19_result.json`): a rank-1 block DAS + inertness axis fit on ONE cue pair serves
the unseen fourth pair on polarity and quantifier but leaves a non-inert complement (0.31-0.78) on dative and
complementizer; v17's THREE-pair fit is in band on all four. Open question: is the shared axis a limit the
fit approaches as pairs are added (complement falls at each step), or a step reached already at two pairs?
This run fits the SAME DAS (rank 1, complement weight 1.0, 120 steps, lr 0.05, seed 0) on each of the three
TWO-pair unions of even rows and tests on the fourth pair; single-pair numbers are read from v18's receipt
and three-pair numbers from v17's, not refit. Bars unchanged from v17/v18.

REGISTERED BEFORE THE RUN
    pred_a_complement_monotone    on >= 3 tested behaviours the median fourth-pair |complement| is
                                  non-increasing 1 -> 2 -> 3 pairs (median of 3 singles, median of 3
                                  two-pair fits, the v17 three-pair value).
                                  Worked example: 0.55 -> 0.30 -> 0.26 -> True; 0.55 -> 0.20 -> 0.26 -> False.
    pred_b_step_at_two            on BOTH dative and complementizer (single-pair failures in v18) >= 2 of 3
                                  two-pair fits are already in band with complement <= 0.30.
                                  Worked example: dative 3/3, complementizer 2/3 -> True; 1/3 -> False.
    pred_c_fraction_monotone      on >= 3 tested behaviours the median fourth-pair fraction is
                                  non-decreasing 1 -> 2 -> 3. Worked example: 0.41 -> 0.60 -> 0.71 -> True.
    pred_d_two_pair_axes_agree    on >= 3 tested behaviours the median per-block |cos| among the three
                                  two-pair axes exceeds v18's median among the single-pair axes.
                                  Worked example: 0.70 vs 0.56 -> True; 0.50 vs 0.56 -> False.
    pred_e_estimator_safe         no two-pair fraction >= 1.20 on any fourth-pair battery.
                                  Worked example: max 1.05 -> True.
    Priors. a and c expected (inertness is the objective; more pairs = less pair-specific direction to
    exploit); b is the informative one -- if it holds, two lexically distinct pairs suffice and the
    protocol can fit on two and hold out one; d expected weakly.
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
OUT = ROOT / "circuits/followups/unit_two_pair_das_v19_result.json"
V17 = ROOT / "circuits/followups/unit_das_common_axis_v17_result.json"
V18 = ROOT / "circuits/followups/unit_single_pair_das_v18_result.json"
SETS = v15.SETS
LO, HI, COMP_BAR, EXACT_BAR, COS_BAR, NEED = 0.50, 1.20, 0.30, 0.50, 0.50, 3
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2400, 40000


def _plan():
    return {"candidate_id": "corpus.unit_two_pair_das_v19",
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
    v18 = json.loads(V18.read_text())["sets"]
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    report, fits = {}, []
    for name, (module, units, maps, fourth_map) in SETS.items():
        a1 = g.rows_of(module, "A1")
        fitted = {"orig": a1, **{f"v{k + 1}": g.lexical_variant(a1, mp) for k, mp in enumerate(maps)}}
        keys = list(fitted)
        duos = {f"{x}+{y}": g.prepare(backend, fitted[x][0::2] + fitted[y][0::2], valid_only=True)
                for i, x in enumerate(keys) for y in keys[i + 1:]}
        p_fourth = g.prepare(backend, g.lexical_variant(a1, fourth_map), valid_only=True)
        r1 = g.block_random_subspace(backend, units, rank=1, seed=1)
        q_das, hist, on_fourth = {}, {}, {}
        for k, prep in duos.items():
            q_das[k], hist[k] = g.fit_block_subspace(backend, prep, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW)
            on_fourth[k] = g.block_direction_battery(backend, p_fourth, units, q_das[k], r1)
        exact = next(iter(on_fourth.values()))["exact_set"]
        tested = exact >= EXACT_BAR
        dk = list(duos)
        cos = {f"{x}|{y}": g.block_cosines(q_das[x], q_das[y]) for i, x in enumerate(dk) for y in dk[i + 1:]}
        med_cos = statistics.median(c for m in cos.values() for c in m.values())
        single = v18[name]["on_fourth"]
        frac = {1: statistics.median(single[k]["das"]["subspace_fraction"] or 0 for k in keys),
                2: statistics.median(on_fourth[k]["subspace_fraction"] or 0 for k in dk),
                3: v17[name]["on_fourth"]["das"]["subspace_fraction"] or 0}
        comp = {1: statistics.median(abs(single[k]["das"]["complement_fraction"] or 0) for k in keys),
                2: statistics.median(abs(on_fourth[k]["complement_fraction"] or 0) for k in dk),
                3: abs(v17[name]["on_fourth"]["das"]["complement_fraction"] or 0)}
        n_band = sum(_band(on_fourth[k]) for k in dk)
        entry = {"units": units, "fitted": maps, "fourth": fourth_map,
                 "dropped": {"fourth": p_fourth.dropped, **{f"{k}_fit": v.dropped for k, v in duos.items()}},
                 "exact_fourth": exact, "tested": tested, "on_fourth": on_fourth, "two_pair_in_band": n_band,
                 "median_fraction_by_pairs": frac, "median_abs_complement_by_pairs": comp,
                 "complement_monotone": comp[1] >= comp[2] >= comp[3], "fraction_monotone": frac[1] <= frac[2] <= frac[3],
                 "two_pair_axis_cos": cos, "median_two_pair_cos": med_cos, "median_single_cos_v18": v18[name]["median_das_cos"],
                 "das_loss_history": hist}
        report[name] = entry
        for k in dk:
            fits.append({"behaviour": name, "pairs": k, "tested": tested, "band": _band(on_fourth[k]),
                         "fraction": on_fourth[k]["subspace_fraction"], "complement": on_fourth[k]["complement_fraction"]})
        print(name, "exact4", round(exact, 2), "frac", {i: round(v, 2) for i, v in frac.items()},
              "comp", {i: round(v, 2) for i, v in comp.items()},
              {k: (round(v["subspace_fraction"] or 0, 2), round(v["complement_fraction"] or 0, 2)) for k, v in on_fourth.items()},
              "cos2", round(med_cos, 2), "cos1", round(entry["median_single_cos_v18"], 2), flush=True)

    tf = [f for f in fits if f["tested"]]
    tb = {n: e for n, e in report.items() if e["tested"]}
    predictions = {
        'pred_a_complement_monotone': sum(e["complement_monotone"] for e in tb.values()) >= NEED,
        'pred_b_step_at_two': all(n in tb and tb[n]["two_pair_in_band"] >= 2 for n in ("dative", "verb_complementizer")),
        'pred_c_fraction_monotone': sum(e["fraction_monotone"] for e in tb.values()) >= NEED,
        'pred_d_two_pair_axes_agree': sum(e["median_two_pair_cos"] > e["median_single_cos_v18"] for e in tb.values()) >= NEED,
        'pred_e_estimator_safe': bool(tf) and all((f["fraction"] or 0) < HI for f in tf),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_two_pair_das_result_v1",
              "candidate_id": "corpus.unit_two_pair_das_v19", "semantics": "block_live",
              "bars": {"band": [LO, HI], "complement": COMP_BAR, "exact": EXACT_BAR, "cos": COS_BAR,
                       "need": NEED, "das": {"rank": 1, "steps": STEPS, "lr": LR, "seed": 0, "complement_weight": CW}},
              "tested": list(tb), "fits": fits, "sets": report,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "tested": result["tested"],
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
