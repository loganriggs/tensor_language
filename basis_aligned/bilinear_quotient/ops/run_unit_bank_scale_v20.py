#!/usr/bin/env python3
# BQGATE: frozen predictions; head sets (v9 receipt), cue pairs, filler shifts, DAS rank/steps/lr/seed, bars fixed before the run.
"""v20: do the 16-row estimates survive a 4x bank?

Every direction result since v9 was fit on the 16 EVEN A1 rows of a 32-row bank (16 base/donor pairs) and
tested on the 16 odd rows, 32 A2 rows and, since v15, 32 rows of an unseen cue pair. A rank-1 fit over a
five-block set has 640 free parameters against 16 margins. The held-out / unseen-pair batteries are the
guard, but the v18/v19 verdicts turned on single fits and polarity's S + C 0.68-0.77 could be noise.
This run enlarges every bank 4x WITHOUT touching the cue: each row's filler (the subject noun; the
adjective for quantifier, whose noun is the final token) is replaced by the filler 8, 16 and 24 places
along the same 32-word vocabulary; for verb_preposition and polarity, whose template is a full noun x cue
product so a shift only reproduces existing rows, the determiner takes Our / This / That instead. 128 rows
per cue pair from the same spec, no duplicates (asserted). The cross-pair
DAS (rank 1, complement weight 1.0, 120 steps, lr 0.05, seed 0) is refit on the 4x three-pair pool and
tested on the 4x fourth pair; the ORIGINAL 16x3-row fit is redone first as the reproduce-the-old-digest
check and then applied unchanged to the 4x fourth pair. Bars unchanged from v17-v19.

REGISTERED BEFORE THE RUN
    pred_a_rows_exchangeable      instrument: on every tested behaviour the fourth-pair exact-set fraction
                                  differs by <= 0.10 between the 32-row and 128-row banks, and the redone
                                  16x3 fit reproduces v17's fourth-pair fraction within 0.05.
                                  Worked example: exact 0.76 vs 0.72, das 0.71 vs 0.69 -> True.
    pred_b_cross_axis_holds_4x    the cross-pair axis refit on the 4x pool serves the 4x fourth pair in band
                                  with complement <= 0.30 on >= 3 tested behaviours.
                                  Worked example: 0.68/0.22, 0.95/0.05, 1.05/0.08, 0.70/0.28 -> True.
    pred_c_small_fit_not_overfit  the 16x3-row axis applied unchanged to the 4x fourth pair is within 0.10
                                  of the 4x-fit axis's fraction on >= 3 tested behaviours.
                                  Worked example: 0.69 vs 0.66 -> counts; 0.69 vs 0.52 -> does not.
    pred_d_dim_not_noise_limited  per-pair diff-in-means fit on 64 even rows lands within 0.10 of v18's
                                  16-row fourth-pair fraction on >= half of the tested (behaviour, pair)
                                  cells. Worked example: 7 of 12 -> True.
    pred_e_polarity_sum_repairs   single-pair DAS on the 4x polarity banks (three fits) gives S + C >= 0.85
                                  on the 4x fourth pair in >= 2 of 3 fits (v18: 0.74, 0.68, 0.75 at 1x).
                                  Worked example: 0.88, 0.86, 0.80 -> True.
    Priors. a expected (fillers are the nuisance the spec already varies); b expected; c is the real
    question -- I expect it to hold, in which case the v9-v19 chain stands and the rank ladder runs on
    the 4x banks; if c fails, every direction result since v9 is re-run at 4x before anything else.
    e is the one I would not bet on: sub-additivity may be structural on polarity.
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
OUT = ROOT / "circuits/followups/unit_bank_scale_v20_result.json"
V17 = ROOT / "circuits/followups/unit_das_common_axis_v17_result.json"
V18 = ROOT / "circuits/followups/unit_single_pair_das_v18_result.json"
SETS = v15.SETS
LO, HI, COMP_BAR, EXACT_BAR, NEED, SUM_LO, SUM_HI = 0.50, 1.20, 0.30, 0.50, 3, 0.85, 1.15
TOL, REPRO_TOL, SHIFTS = 0.10, 0.05, (8, 16, 24)
# filler slot per behaviour: cyclic shift along the 32-word slot vocabulary, or -- where the template is a
# full noun x cue product so a shift only reproduces existing rows -- three determiner alternatives at slot 0
SLOT = {"quantifier_number": ("cycle", 3), "verb_preposition": ("alts", 0, ("Our", "This", "That")),
        "polarity_licensing": ("alts", 0, ("Our", "This", "That"))}
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 6000, 200000


def _plan():
    return {"candidate_id": "corpus.unit_bank_scale_v20",
            "sets": {k: {"units": v[1], "fitted": v[2], "fourth": v[3]} for k, v in SETS.items()},
            "shifts": list(SHIFTS), "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 13 * STEPS, "model_updates": 0, "fit_parameters": 13 * 5 * 128,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _vocab(rows, slot):
    return sorted({r[f"{s}_text"].split()[slot] for r in rows for s in ("base", "donor")})


def _scaled(rows, spec):
    """rows plus three filler-varied copies: the slot word moved 8 / 16 / 24 places along the slot vocabulary
    ("cycle"), or replaced by each of three alternatives ("alts"). Cue words are never touched."""
    out = list(rows)
    if spec[0] == "cycle":
        vocab = _vocab(rows, spec[1])
        assert len(vocab) == 32, len(vocab)
        for k in SHIFTS:
            out += g.lexical_variant(rows, {w: vocab[(i + k) % len(vocab)] for i, w in enumerate(vocab)})
    else:
        (word,) = _vocab(rows, spec[1])
        for alt in spec[2]:
            out += g.lexical_variant(rows, {word: alt})
    texts = [r[f"{s}_text"] for r in out for s in ("base", "donor")]
    assert len(set(texts)) == len(texts), "filler variation reproduced an existing row"
    return out


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
    report, dim_cells, pol_sums = {}, [], []
    for name, (module, units, maps, fourth_map) in SETS.items():
        slot = SLOT.get(name, ("cycle", 1))
        a1 = g.rows_of(module, "A1")
        pairs1 = {"orig": a1, **{f"v{k + 1}": g.lexical_variant(a1, mp) for k, mp in enumerate(maps)}}
        fourth1 = g.lexical_variant(a1, fourth_map)
        pairs4 = {k: _scaled(rows, slot) for k, rows in pairs1.items()}
        fourth4 = _scaled(fourth1, slot)
        assert all(len(r) == 4 * len(a1) for r in pairs4.values()) and len(fourth4) == 4 * len(a1)
        pool1 = g.prepare(backend, sum((rows[0::2] for rows in pairs1.values()), []), valid_only=True)
        pool4 = g.prepare(backend, sum((rows[0::2] for rows in pairs4.values()), []), valid_only=True)
        p4_1 = g.prepare(backend, fourth1, valid_only=True)
        p4_4 = g.prepare(backend, fourth4, valid_only=True)
        r1 = g.block_random_subspace(backend, units, rank=1, seed=1)

        q_small, _ = g.fit_block_subspace(backend, pool1, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW)
        q_big, hist = g.fit_block_subspace(backend, pool4, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW)
        small_on_1 = g.block_direction_battery(backend, p4_1, units, q_small, r1)
        small_on_4 = g.block_direction_battery(backend, p4_4, units, q_small, r1)
        big_on_4 = g.block_direction_battery(backend, p4_4, units, q_big, r1)
        exact1, exact4 = small_on_1["exact_set"], big_on_4["exact_set"]
        tested = exact4 >= EXACT_BAR
        v17_frac = v17[name]["on_fourth"]["das"]["subspace_fraction"]
        repro = abs(small_on_1["subspace_fraction"] - v17_frac)

        dim4 = {}
        for k, rows in pairs4.items():
            q = g.block_diff_in_means(backend, g.prepare(backend, rows[0::2], valid_only=True), units)
            dim4[k] = g.block_direction_battery(backend, p4_4, units, q, r1)
            d18 = v18[name]["on_fourth"][k]["dim"]["subspace_fraction"]
            dim_cells.append({"behaviour": name, "pair": k, "tested": tested, "dim_1x": d18,
                              "dim_4x": dim4[k]["subspace_fraction"], "close": abs(dim4[k]["subspace_fraction"] - d18) <= TOL})
        entry = {"units": units, "fitted": maps, "fourth": fourth_map, "slot": slot,
                 "rows": {"pair_1x": len(a1), "pair_4x": len(fourth4), "pool_1x_fit": len(pool1.rows), "pool_4x_fit": len(pool4.rows)},
                 "dropped": {"pool_1x": pool1.dropped, "pool_4x": pool4.dropped, "fourth_1x": p4_1.dropped, "fourth_4x": p4_4.dropped},
                 "exact_fourth_1x": exact1, "exact_fourth_4x": exact4, "exact_gap": abs(exact1 - exact4), "tested": tested,
                 "v17_fourth_fraction": v17_frac, "small_axis_on_1x": small_on_1, "repro_gap": repro,
                 "small_axis_on_4x": small_on_4, "big_axis_on_4x": big_on_4,
                 "small_vs_big_gap": abs(small_on_4["subspace_fraction"] - big_on_4["subspace_fraction"]),
                 "axis_cos_small_big": g.block_cosines(q_small, q_big), "dim_4x_on_fourth": dim4,
                 "das_loss_history_4x": hist}
        if name == "polarity_licensing":
            single = {}
            for k, rows in pairs4.items():
                q, _ = g.fit_block_subspace(backend, g.prepare(backend, rows[0::2], valid_only=True), units,
                                            rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW)
                single[k] = g.block_direction_battery(backend, p4_4, units, q, r1)
                pol_sums.append(single[k]["linearity_sum"])
            entry["single_pair_das_4x_on_fourth"] = single
        report[name] = entry
        print(name, "exact", round(exact1, 2), round(exact4, 2), "v17", round(v17_frac or 0, 2), "small@1x", round(small_on_1["subspace_fraction"], 2),
              "small@4x", round(small_on_4["subspace_fraction"], 2), round(small_on_4["complement_fraction"], 2),
              "big@4x", round(big_on_4["subspace_fraction"], 2), round(big_on_4["complement_fraction"], 2), "S+C", round(big_on_4["linearity_sum"], 2),
              "dim4x", {k: round(v["subspace_fraction"], 2) for k, v in dim4.items()}, "cos", {k: round(v, 2) for k, v in entry["axis_cos_small_big"].items()},
              "polS+C", [round(x, 2) for x in pol_sums] if name == "polarity_licensing" else "", flush=True)

    tb = {n: e for n, e in report.items() if e["tested"]}
    tc = [c for c in dim_cells if c["tested"]]
    predictions = {
        'pred_a_rows_exchangeable': bool(tb) and all(e["exact_gap"] <= TOL and e["repro_gap"] <= REPRO_TOL for e in tb.values()),
        'pred_b_cross_axis_holds_4x': sum(_band(e["big_axis_on_4x"]) for e in tb.values()) >= NEED,
        'pred_c_small_fit_not_overfit': sum(e["small_vs_big_gap"] <= TOL for e in tb.values()) >= NEED,
        'pred_d_dim_not_noise_limited': bool(tc) and sum(c["close"] for c in tc) * 2 >= len(tc),
        'pred_e_polarity_sum_repairs': sum(SUM_LO <= s <= SUM_HI for s in pol_sums) >= 2,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_bank_scale_result_v1",
              "candidate_id": "corpus.unit_bank_scale_v20", "semantics": "block_live",
              "bars": {"band": [LO, HI], "complement": COMP_BAR, "exact": EXACT_BAR, "tol": TOL, "repro_tol": REPRO_TOL,
                       "sum": [SUM_LO, SUM_HI], "need": NEED, "shifts": list(SHIFTS),
                       "das": {"rank": 1, "steps": STEPS, "lr": LR, "seed": 0, "complement_weight": CW}},
              "tested": list(tb), "dim_cells": dim_cells, "sets": report,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "tested": result["tested"], "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
