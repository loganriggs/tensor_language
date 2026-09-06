#!/usr/bin/env python3
# BQGATE: frozen predictions; head set (v9 receipt), cue pairs, interpolation grid, DAS rank/steps/lr/seed, bars fixed before the run.
"""v16: anatomy of an over-band direction. Why does the POOLED quantifier axis deliver MORE than the exact set?

v15 (`unit_common_axis_v15_result.json`): on the quantifier_number set [07:08, 11:03] the diff-in-means
direction pooled over Each/All, Neither/Several, One/Many gives subspace fraction 1.57 on the
Neither/Several odd rows (S + C 1.52) and 1.36 on the unseen Either/Some (S + C 1.12), while every
single-pair direction on its own pair sits at 1.03-1.07 with S + C 1.04-1.06. The pooled axis is
|cos| 0.81-0.86 from Neither/Several's own axis. So a direction NEAR the carried axis, but rotated,
moves the margin further than replacing the whole block does: the readout is steeper along the
rotated direction than along the actual delta. That is the steering signature the lane retracted
in v4/v7, now appearing with NO optimizer -- averaging alone found it. Three registered
decompositions, rank fixed at 1 throughout (rank-3 union only where v15 already registered it):
  (1) per block: patch the pooled axis in ONE block only (07:08 alone; 11:03 alone), other block
      untouched, against that block's own exact patch. Which block overshoots?
  (2) interpolation: q(t) = normalize((1 - t) d_own + t d_pool) per block, t in {0, .25, .5, .75, 1};
      subspace fraction and S + C on Neither/Several odd rows. Does the overshoot grow with the
      rotation away from the carried axis?
  (3) estimator check (README protocol): block DAS rank 1 with complement inertness
      (complement_weight 1.0, 120 steps, lr 0.05, seed 0) fit on the POOLED even rows; battery on
      Neither/Several odd and Either/Some. Does a fitted common axis avoid the overshoot?
  Bars: band [0.50, 1.20]; complement <= 0.30; S + C in [0.85, 1.15]; over-band = fraction >= 1.20.

REGISTERED BEFORE THE RUN
    pred_a_overshoot_reproduces   pooled axis on Neither/Several odd rows: fraction >= 1.20.
                                  Worked example: 1.57 -> True; 1.15 -> False.
    pred_b_one_block_overshoots   with single-block patches, exactly ONE block's pooled-axis recovery
                                  exceeds that block's own exact recovery by >= 0.15 of the full exact
                                  set. Worked example: 11:03 axis 0.55 vs exact 0.30 (gap 0.25), 07:08
                                  axis 0.40 vs exact 0.38 -> True; both gaps >= 0.15 -> False.
    pred_c_overshoot_grows_with_t fraction on Neither/Several odd is non-decreasing in t (tolerance
                                  0.05) and fraction(t=1) - fraction(t=0) >= 0.30.
                                  Worked example: 1.04, 1.12, 1.25, 1.41, 1.57 -> True; 1.04, 1.30, 1.10, 1.45, 1.57 -> False.
    pred_d_das_common_axis_sane   block DAS rank 1 (complement inertness) fit on pooled rows: S + C in
                                  [0.85, 1.15] on Neither/Several odd AND in band with complement <= 0.30
                                  on Either/Some. Worked example: S + C 0.98, fourth 0.88 / 0.10 -> True;
                                  fourth 1.31 -> False.
    pred_e_union_sane             rank-3 union (v15's registered ceiling) has S + C in [0.85, 1.15] on
                                  Neither/Several odd AND Either/Some. Worked example: 1.02 / 1.04 -> True.
    Priors. a expected (same rows, same direction). b: I expect 11:03 (the number axis, v11) to be
    the steep one -- unsure. c expected. d unsure: the optimizer may find the same steep direction
    (v4's lesson) -- if d FAILS the pooled-axis route is a steering artifact and the common component
    cannot be estimated by either method on this set; if it HOLDS, diff-in-means pooling is the
    culprit and the DAS estimate is the one to report. e expected (v15: 1.04 on the fourth pair).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import circuit_fast_screen_candidate_quantifier_number as m_quantifier

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_quantifier_overshoot_v16_result.json"
UNITS = ["attn:07:head:08", "attn:11:head:03"]
MAPS = [{"Each": "Neither", "All": "Several"}, {"Each": "One", "All": "Many"}]
FOURTH = {"Each": "Either", "All": "Some"}
TS = (0.0, 0.25, 0.5, 0.75, 1.0)
LO, HI, COMP_BAR, SUM_LO, SUM_HI, OVER, BLOCK_GAP, TOL, GROW = 0.50, 1.20, 0.30, 0.85, 1.15, 1.20, 0.15, 0.05, 0.30
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 600, 12000


def _plan():
    return {"candidate_id": "quantifier_number.unit_overshoot_v16", "units": UNITS, "maps": MAPS, "fourth": FOURTH,
            "interpolation": list(TS), "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": 2 * 128,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _band(b):
    return b["subspace_fraction"] is not None and LO <= b["subspace_fraction"] <= HI \
        and abs(b["complement_fraction"]) <= COMP_BAR


def _sum_ok(b):
    return b["linearity_sum"] is not None and SUM_LO <= b["linearity_sum"] <= SUM_HI


def _interp(d_own, d_pool, t):
    out = {}
    for key in d_own:
        v = (1 - t) * d_own[key] + t * d_pool[key]
        out[key] = v / v.norm(dim=0, keepdim=True)
    return out


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    a1 = g.rows_of(m_quantifier, "A1")
    fitted = {"orig": a1, **{f"v{k + 1}": g.lexical_variant(a1, mp) for k, mp in enumerate(MAPS)}}
    fourth = g.lexical_variant(a1, FOURTH)
    p_fit = {k: g.prepare(backend, rows[0::2], valid_only=True) for k, rows in fitted.items()}
    p_v1_odd = g.prepare(backend, fitted["v1"][1::2], valid_only=True)
    p_fourth = g.prepare(backend, fourth, valid_only=True)
    p_pool = g.prepare(backend, [r for k in fitted for r in fitted[k][0::2]], valid_only=True)
    d = {k: g.block_diff_in_means(backend, p_fit[k], UNITS) for k in fitted}
    d_pool = g.block_diff_in_means(backend, p_pool, UNITS)
    union = g.block_union(*d.values())
    r1 = g.block_random_subspace(backend, UNITS, rank=1, seed=1)
    r3 = g.block_random_subspace(backend, UNITS, rank=3, seed=1)

    # (0) the overshoot itself, on both row sets
    pooled = {"v1_odd": g.block_direction_battery(backend, p_v1_odd, UNITS, d_pool, r1),
              "fourth": g.block_direction_battery(backend, p_fourth, UNITS, d_pool, r1)}
    exact_full = {"v1_odd": pooled["v1_odd"]["exact_set"], "fourth": pooled["fourth"]["exact_set"]}

    # (1) per-block: pooled axis in one block only vs that block's exact patch, other block untouched
    per_block = {}
    for u in UNITS:
        key = g.block_key(u)
        for ev, prep in (("v1_odd", p_v1_odd), ("fourth", p_fourth)):
            ex = g.recovery(prep, g.patched_axis(backend, prep, [u]))
            ax = g.recovery(prep, g.patched_axis(backend, prep, [u], q={key: d_pool[key]}))
            cp = g.recovery(prep, g.patched_axis(backend, prep, [u], q={key: d_pool[key]}, complement=True))
            per_block[f"{u}|{ev}"] = {"exact_block": ex, "axis_block": ax, "complement_block": cp,
                                      "exact_fraction": ex / exact_full[ev], "axis_fraction": ax / exact_full[ev],
                                      "gap": (ax - ex) / exact_full[ev]}

    # (2) interpolation own -> pooled on Neither/Several odd rows
    interp = {}
    for t in TS:
        b = g.block_direction_battery(backend, p_v1_odd, UNITS, _interp(d["v1"], d_pool, t), r1)
        interp[str(t)] = {"subspace_fraction": b["subspace_fraction"], "complement_fraction": b["complement_fraction"],
                          "linearity_sum": b["linearity_sum"]}
    fr = [interp[str(t)]["subspace_fraction"] for t in TS]

    # (3) estimator check: block DAS rank 1 with complement inertness on the pooled rows
    q_das, hist = g.fit_block_subspace(backend, p_pool, UNITS, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW)
    das = {"v1_odd": g.block_direction_battery(backend, p_v1_odd, UNITS, q_das, r1),
           "fourth": g.block_direction_battery(backend, p_fourth, UNITS, q_das, r1)}
    das_cos = {"to_pooled": g.block_cosines(q_das, d_pool), "to_v1": g.block_cosines(q_das, d["v1"])}
    un = {"v1_odd": g.block_direction_battery(backend, p_v1_odd, UNITS, union, r3),
          "fourth": g.block_direction_battery(backend, p_fourth, UNITS, union, r3)}

    gaps = {u: per_block[f"{u}|v1_odd"]["gap"] for u in UNITS}
    predictions = {
        'pred_a_overshoot_reproduces': (pooled["v1_odd"]["subspace_fraction"] or 0) >= OVER,
        'pred_b_one_block_overshoots': sum(v >= BLOCK_GAP for v in gaps.values()) == 1,
        'pred_c_overshoot_grows_with_t': all(fr[i + 1] >= fr[i] - TOL for i in range(len(fr) - 1)) and fr[-1] - fr[0] >= GROW,
        'pred_d_das_common_axis_sane': _sum_ok(das["v1_odd"]) and _band(das["fourth"]),
        'pred_e_union_sane': _sum_ok(un["v1_odd"]) and _sum_ok(un["fourth"]),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_overshoot_anatomy_result_v1",
              "candidate_id": "quantifier_number.unit_overshoot_v16", "semantics": "block_live", "units": UNITS,
              "bars": {"band": [LO, HI], "complement": COMP_BAR, "linearity_sum": [SUM_LO, SUM_HI], "over": OVER,
                       "block_gap": BLOCK_GAP, "tol": TOL, "grow": GROW, "das": {"rank": 1, "steps": STEPS, "lr": LR, "complement_weight": CW}},
              "dropped": {"v1_odd": p_v1_odd.dropped, "fourth": p_fourth.dropped, "pool": p_pool.dropped},
              "exact_full": exact_full, "pooled": pooled, "per_block": per_block, "block_gaps_v1_odd": gaps,
              "interpolation": interp, "das": das, "das_cos": das_cos, "das_loss_history": hist, "union": un,
              "pooled_cos_to_v1": g.block_cosines(d_pool, d["v1"]),
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions,
                      "pooled": {k: (round(b["subspace_fraction"], 2), round(b["complement_fraction"], 2), round(b["linearity_sum"], 2)) for k, b in pooled.items()},
                      "per_block": {k: (round(v["exact_fraction"], 2), round(v["axis_fraction"], 2), round(v["gap"], 2)) for k, v in per_block.items()},
                      "interp": [round(x, 2) for x in fr], "interp_sum": [round(interp[str(t)]["linearity_sum"], 2) for t in TS],
                      "das": {k: (round(b["subspace_fraction"], 2), round(b["complement_fraction"], 2), round(b["linearity_sum"], 2)) for k, b in das.items()},
                      "das_cos": das_cos,
                      "union": {k: (round(b["subspace_fraction"], 2), round(b["complement_fraction"], 2), round(b["linearity_sum"], 2)) for k, b in un.items()},
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
