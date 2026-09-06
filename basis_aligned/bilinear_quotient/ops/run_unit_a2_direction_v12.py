#!/usr/bin/env python3
# BQGATE: frozen predictions; sets (from the terminal v9 receipt), ranks, seeds, steps and bars fixed before the run.
"""v12: the four v9 sets whose A1 direction did not serve A2 -- a second direction, or a nonlinearity?

v9 (`unit_corpus_battery_v9_result.json`) put 17 behaviours through greedy head sets and a block
diff-in-means direction fit on A1. Four sets missed the direction band:
  dative_v2               A1 held-out 1.01 / complement 0.00, but A2 0.39 / complement 0.56
  interrogative_licensing A1 1.01 / 0.00; A2 0.58 / 0.23, S + C 0.81
  polarity_licensing      A1 0.95 / 0.01; A2 0.69 / 0.11, S + C 0.80
  voice_frame             A1 0.81 / 0.00, S + C 0.81 already on held-out A1
Two mechanisms produce those numbers. (i) A DIFFERENT DIRECTION for A2: the same heads carry the
variable in the fresh construction along another axis, so the A1 direction's complement carries
A2 (dative's 0.56 says exactly this). The registered test is rank fixed IN ADVANCE: fit the A2
direction on A2's even rows, measure per-block |cos| to the A1 direction, and test the rank-2 union
per block (the span of both) on the held-out rows of BOTH families. (ii) A NONLINEAR route inside
the set: subspace and complement do not add (S + C < 0.85) even on the family the direction was fit
on -- voice_frame. For that the check is block DAS rank 1 (seed 0, 120 steps, lr 0.05, exact-set
objective): if a fitted direction cannot reach S + C >= 0.85 either, the deficit is the route, not
the estimator.

  per behaviour: A1 even / odd, A2 even / odd preps (valid_only, dropped counted);
  d1 = block dim on A1 even, d2 = block dim on A2 even; batteries (fraction, complement, S + C,
  random rank-matched seed 1) of d1, d2 and union(d1, d2) on A1 odd and A2 odd; per-block |cos|;
  voice_frame additionally: block DAS rank 1 on A1 even, battery on A1 odd.
  Band [0.50, 1.20]; complement <= 0.30; S + C in [0.85, 1.15].

REGISTERED BEFORE THE RUN
    pred_a_a2_own_direction_serves    dative, interrogative, polarity: d2 on A2 odd rows in band with
                                      complement <= 0.30. Worked example: 0.93 / 0.04 on all three -> True;
                                      dative 0.45 -> False.
    pred_b_dative_directions_differ   dative: median per-block |cos|(d1, d2) <= 0.50 (five blocks).
                                      Worked example: cosines 0.2, 0.3, 0.4, 0.7, 0.8 -> median 0.4 -> True.
    pred_c_union_serves_both          dative, interrogative, polarity: union(d1, d2) (rank 2 per block)
                                      in band with complement <= 0.30 on BOTH A1 odd and A2 odd.
                                      Worked example: dative A1 1.0 / A2 0.95 -> True; A2 0.48 -> False.
    pred_d_a2_deficit_is_direction    interrogative and polarity: d2 on A2 odd has S + C in [0.85, 1.15]
                                      (the 0.80 was direction mismatch, not a nonlinear route).
                                      Worked example: 0.99 and 1.02 -> True; polarity 0.79 -> False.
    pred_e_voice_route_nonlinear      voice_frame: BOTH d1 and block DAS rank 1 give S + C <= 0.85 on
                                      A1 odd rows. Worked example: 0.81 and 0.83 -> True; DAS 0.97 -> False.

    Priors. a expected. b expected for dative (complement 0.56 on A2 is a second axis or nothing).
    c expected if a and b. d unsure -- interrogative's A2 complement 0.23 could be either mechanism.
    e unsure; if it FAILS, voice_frame's deficit is an estimator problem and DAS is the direction to
    report for it. Rank 2 is used only where registered here (the union); no rank is raised on a null.
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

import circuit_fast_screen_candidate_dative as m_dative
import circuit_fast_screen_candidate_interrogative_licensing as m_interrogative
import circuit_fast_screen_candidate_polarity_licensing as m_polarity
import circuit_fast_screen_candidate_voice_frame as m_voice

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_a2_direction_v12_result.json"
SETS = {
    "dative_v2": (m_dative, ["attn:14:head:08", "attn:07:head:08", "attn:06:head:03", "attn:13:head:08", "attn:11:head:03"]),
    "interrogative_licensing": (m_interrogative, ["attn:09:head:07", "attn:07:head:08", "attn:03:head:00", "attn:02:head:06"]),
    "polarity_licensing": (m_polarity, ["attn:07:head:08", "attn:08:head:01", "attn:04:head:07", "attn:03:head:00"]),
    "voice_frame": (m_voice, ["attn:07:head:08", "attn:01:head:05", "attn:00:head:03", "attn:04:head:01"]),
}
TWO_DIR = ("dative_v2", "interrogative_licensing", "polarity_licensing")
LO, HI, COMP_BAR, SUM_LO, SUM_HI, COS_BAR = 0.50, 1.20, 0.30, 0.85, 1.15, 0.50
STEPS, LR = 120, 0.05
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 600, 20000


def _plan():
    return {"candidate_id": "corpus.unit_a2_direction_v12", "sets": {k: v[1] for k, v in SETS.items()},
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": 4 * 128,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _band(b):
    return b["subspace_fraction"] is not None and LO <= b["subspace_fraction"] <= HI \
        and abs(b["complement_fraction"]) <= COMP_BAR


def _sum_ok(b):
    return b["linearity_sum"] is not None and SUM_LO <= b["linearity_sum"] <= SUM_HI


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        a1, a2 = g.rows_of(module, "A1"), g.rows_of(module, "A2")
        p = {k: g.prepare(backend, rows, valid_only=True) for k, rows in
             (("a1_fit", a1[0::2]), ("a1_odd", a1[1::2]), ("a2_fit", a2[0::2]), ("a2_odd", a2[1::2]))}
        d1 = g.block_diff_in_means(backend, p["a1_fit"], units)
        d2 = g.block_diff_in_means(backend, p["a2_fit"], units)
        union = g.block_union(d1, d2)
        r1, r2 = g.block_random_subspace(backend, units, rank=1, seed=1), g.block_random_subspace(backend, units, rank=2, seed=1)
        cos = g.block_cosines(d1, d2)
        entry = {"units": units, "dropped": {k: v.dropped for k, v in p.items()},
                 "cos_d1_d2": cos, "median_cos": statistics.median(cos.values()), "batteries": {}}
        for dname, q, r in (("d1", d1, r1), ("d2", d2, r1), ("union", union, r2)):
            entry["batteries"][dname] = {ev: g.block_direction_battery(backend, p[ev], units, q, r)
                                         for ev in ("a1_odd", "a2_odd")}
        if name == "voice_frame":
            q_das, hist = g.fit_block_subspace(backend, p["a1_fit"], units, rank=1, steps=STEPS, lr=LR, seed=0)
            entry["batteries"]["das"] = {"a1_odd": g.block_direction_battery(backend, p["a1_odd"], units, q_das, r1)}
            entry["das_loss_history"] = hist
            entry["das_cos_to_d1"] = g.block_cosines(q_das, d1)
        report[name] = entry
        print(name, "cos", {k: round(v, 2) for k, v in cos.items()},
              {d: {ev: (round(b["subspace_fraction"] or 0, 2), round(b["complement_fraction"] or 0, 2),
                        round(b["linearity_sum"] or 0, 2)) for ev, b in bs.items()}
               for d, bs in entry["batteries"].items()}, flush=True)

    B = {n: report[n]["batteries"] for n in report}
    v = B["voice_frame"]
    predictions = {
        'pred_a_a2_own_direction_serves': all(_band(B[n]["d2"]["a2_odd"]) for n in TWO_DIR),
        'pred_b_dative_directions_differ': report["dative_v2"]["median_cos"] <= COS_BAR,
        'pred_c_union_serves_both': all(_band(B[n]["union"]["a1_odd"]) and _band(B[n]["union"]["a2_odd"]) for n in TWO_DIR),
        'pred_d_a2_deficit_is_direction': all(_sum_ok(B[n]["d2"]["a2_odd"]) for n in ("interrogative_licensing", "polarity_licensing")),
        'pred_e_voice_route_nonlinear': (v["d1"]["a1_odd"]["linearity_sum"] <= SUM_LO
                                         and v["das"]["a1_odd"]["linearity_sum"] <= SUM_LO),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_a2_direction_result_v1",
              "candidate_id": "corpus.unit_a2_direction_v12", "semantics": "block_live",
              "bars": {"band": [LO, HI], "complement": COMP_BAR, "linearity_sum": [SUM_LO, SUM_HI],
                       "cos": COS_BAR, "das_steps": STEPS, "das_lr": LR, "union_rank_per_block": 2},
              "sets": report, "seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
