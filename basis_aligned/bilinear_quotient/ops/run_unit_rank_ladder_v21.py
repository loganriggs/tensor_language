#!/usr/bin/env python3
# BQGATE: frozen predictions; head sets (v9 receipt), cue pairs, rank ladder, DAS steps/lr/seed/inertness weight, selection rule and bars fixed before the run.
"""v21: the smallest rank that works -- a registered rank ladder per head set.

Every direction in this lane has been rank 1 because diff-in-means is rank 1 and most head-set deltas were
94-99% rank 1 (v4). That is not a rank test. This run fits block DAS (complement weight 1.0, 120 steps,
lr 0.05, seed 0) INDEPENDENTLY at ranks 1, 2, 4, 8 per block (not nested), with a rank-MATCHED random
subspace at every rung, and applies one selection rule fixed here: the SMALLEST rank whose held-out battery
passes band [0.50, 1.20], |complement| <= 0.30, S + C in [0.85, 1.15] and random <= 0.10. The selected rank
is then confirmed with no rank choice on a split the selection never saw.
  Cue-pair behaviours (dative, quantifier, polarity, complementizer; v15 SETS): fit on the even rows of the
  three-pair pool, select on the pool's odd rows, confirm on the unseen fourth pair.
  v9 head sets whose rank-1 direction was NOT in band and that have no cue pairs (interrogative, voice):
  fit on even A1, select on odd A1, confirm on A2.
Fractions are of the exact-set effect; a confirmation split with exact-set < 0.50 is UNTESTED.

REGISTERED BEFORE THE RUN
    pred_a_rank1_minimal_where_in_band   quantifier and complementizer (rank-1 in band since v17) select rank 1.
                                         Worked example: both 1 -> True; quantifier 2 -> False.
    pred_b_polarity_needs_rank2          polarity selects rank >= 2 AND at the selected rank S + C on the fourth
                                         pair is in [0.85, 1.15] (v18-v20: rank-1 S + C 0.67-0.83 is a second
                                         direction). Worked example: rank 2, S + C 0.96 -> True; rank 1 -> False.
    pred_c_ladder_finds_rank             interrogative and voice (rank-1 not in band in v9) both select some rank
                                         <= 8 on held-out A1. Worked example: 2 and 4 -> True; none for voice -> False.
    pred_d_selection_confirms            every behaviour with a selected rank passes band + complement at that
                                         rank on its confirmation split (fourth pair / A2), tested splits only.
                                         Worked example: 5 of 5 -> True; 4 of 5 -> False.
    pred_e_random_bar_meaningful         rank-matched random at rank 8 stays <= 0.10 on the confirmation split of
                                         every tested behaviour (8 of 128 dims is still nothing).
                                         Worked example: max 0.06 -> True; 0.14 -> False.
    Priors. a expected. b is the informative one and I am unsure -- the sub-additivity could be a nonlinear
    read of one direction, in which case no rank repairs S + C and polarity's variable is not a subspace.
    c unsure; voice's greedy set is weak (0.65 joint with four heads). e expected but must be shown.
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
import circuit_fast_screen_candidate_interrogative_licensing as m_interrogative
import circuit_fast_screen_candidate_voice_frame as m_voice

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_rank_ladder_v21_result.json"
V9 = ROOT / "circuits/followups/unit_corpus_battery_v9_result.json"
PAIRED = {k: v for k, v in v15.SETS.items() if k in ("dative", "quantifier_number", "polarity_licensing", "verb_complementizer")}
UNPAIRED = {"interrogative_licensing.question_vs_declarative": m_interrogative, "voice_frame.passive_vs_active": m_voice}
RANKS = (1, 2, 4, 8)
LO, HI, COMP_BAR, SUM_LO, SUM_HI, RAND_BAR, EXACT_BAR = 0.50, 1.20, 0.30, 0.85, 1.15, 0.10, 0.50
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 6000, 200000


def _plan():
    return {"candidate_id": "corpus.unit_rank_ladder_v21", "ranks": list(RANKS),
            "paired": {k: {"units": v[1], "fitted": v[2], "fourth": v[3]} for k, v in PAIRED.items()},
            "unpaired": list(UNPAIRED), "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 6 * len(RANKS) * STEPS, "model_updates": 0,
            "fit_parameters": 6 * 5 * 128 * sum(RANKS), "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _passes(b):
    f, c, s, r = b["subspace_fraction"], b["complement_fraction"], b["linearity_sum"], b["random_fraction"]
    return all(x is not None for x in (f, c, s, r)) and LO <= f <= HI and abs(c) <= COMP_BAR \
        and SUM_LO <= s <= SUM_HI and abs(r) <= RAND_BAR


def _band(b):
    return b["subspace_fraction"] is not None and LO <= b["subspace_fraction"] <= HI \
        and abs(b["complement_fraction"]) <= COMP_BAR


def _ladder(backend, units, p_fit, p_sel, p_conf):
    rungs, selected = {}, None
    for r in RANKS:
        q, hist = g.fit_block_subspace(backend, p_fit, units, rank=r, steps=STEPS, lr=LR, seed=0, complement_weight=CW)
        rnd = g.block_random_subspace(backend, units, rank=r, seed=1)
        sel = g.block_direction_battery(backend, p_sel, units, q, rnd)
        conf = g.block_direction_battery(backend, p_conf, units, q, rnd)
        rungs[r] = {"select": sel, "confirm": conf, "select_passes": _passes(sel), "loss_history": hist}
        if selected is None and _passes(sel):
            selected = r
    return rungs, selected


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    v9 = json.loads(V9.read_text())["behaviours"]
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    report = {}
    for name, (module, units, maps, fourth_map) in PAIRED.items():
        a1 = g.rows_of(module, "A1")
        pairs = [a1] + [g.lexical_variant(a1, mp) for mp in maps]
        p_fit = g.prepare(backend, sum((p[0::2] for p in pairs), []), valid_only=True)
        p_sel = g.prepare(backend, sum((p[1::2] for p in pairs), []), valid_only=True)
        p_conf = g.prepare(backend, g.lexical_variant(a1, fourth_map), valid_only=True)
        rungs, selected = _ladder(backend, units, p_fit, p_sel, p_conf)
        report[name] = {"kind": "paired", "units": units, "confirm_split": "fourth_pair", "rungs": rungs, "selected_rank": selected,
                        "exact_confirm": next(iter(rungs.values()))["confirm"]["exact_set"]}
    for name, module in UNPAIRED.items():
        units = v9[name]["greedy"]["chosen"]
        a1, a2 = g.rows_of(module, "A1"), g.rows_of(module, "A2")
        p_fit, p_sel = g.prepare(backend, a1[0::2], valid_only=True), g.prepare(backend, a1[1::2], valid_only=True)
        p_conf = g.prepare(backend, a2, valid_only=True)
        rungs, selected = _ladder(backend, units, p_fit, p_sel, p_conf)
        report[name] = {"kind": "unpaired", "units": units, "confirm_split": "A2", "rungs": rungs, "selected_rank": selected,
                        "exact_confirm": next(iter(rungs.values()))["confirm"]["exact_set"]}
    for name, e in report.items():
        e["tested"] = e["exact_confirm"] >= EXACT_BAR
        sr = e["selected_rank"]
        e["confirmed"] = bool(sr) and e["tested"] and _band(e["rungs"][sr]["confirm"])
        print(name, "exact_conf", round(e["exact_confirm"], 2), "selected", sr, "confirmed", e["confirmed"],
              {r: (round(v["select"]["subspace_fraction"] or 0, 2), round(v["select"]["complement_fraction"] or 0, 2),
                   round(v["select"]["linearity_sum"] or 0, 2), round(v["select"]["random_fraction"] or 0, 2),
                   "conf", round(v["confirm"]["subspace_fraction"] or 0, 2), round(v["confirm"]["complement_fraction"] or 0, 2),
                   round(v["confirm"]["linearity_sum"] or 0, 2)) for r, v in e["rungs"].items()}, flush=True)

    pol = report["polarity_licensing"]
    pol_sum = pol["rungs"][pol["selected_rank"]]["confirm"]["linearity_sum"] if pol["selected_rank"] else None
    with_sel = [e for e in report.values() if e["selected_rank"] and e["tested"]]
    tested = [e for e in report.values() if e["tested"]]
    predictions = {
        'pred_a_rank1_minimal_where_in_band': all(report[n]["selected_rank"] == 1 for n in ("quantifier_number", "verb_complementizer")),
        'pred_b_polarity_needs_rank2': bool(pol["selected_rank"]) and pol["selected_rank"] >= 2 and pol_sum is not None and SUM_LO <= pol_sum <= SUM_HI,
        'pred_c_ladder_finds_rank': all(report[n]["selected_rank"] is not None for n in UNPAIRED),
        'pred_d_selection_confirms': bool(with_sel) and all(e["confirmed"] for e in with_sel),
        'pred_e_random_bar_meaningful': bool(tested) and all(abs(e["rungs"][8]["confirm"]["random_fraction"] or 0) <= RAND_BAR for e in tested),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_rank_ladder_result_v1",
              "candidate_id": "corpus.unit_rank_ladder_v21", "semantics": "block_live",
              "bars": {"band": [LO, HI], "complement": COMP_BAR, "sum": [SUM_LO, SUM_HI], "random": RAND_BAR, "exact": EXACT_BAR,
                       "ranks": list(RANKS), "selection": "smallest rank passing band+complement+sum+random on the selection split",
                       "das": {"steps": STEPS, "lr": LR, "seed": 0, "complement_weight": CW, "nested": False}},
              "selected": {n: e["selected_rank"] for n, e in report.items()}, "confirmed": {n: e["confirmed"] for n, e in report.items()},
              "tested": [n for n, e in report.items() if e["tested"]], "sets": report,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "selected": result["selected"], "confirmed": result["confirmed"],
                      "tested": result["tested"], "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
