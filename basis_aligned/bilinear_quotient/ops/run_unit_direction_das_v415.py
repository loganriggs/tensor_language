#!/usr/bin/env python3
# BQGATE: five frozen predictions; unit set, RANK, cells and bars all fixed before the run.
"""v415: the DAS pass on the shared direction -- rank fixed in advance, all four hypotheses, transfer across cells.

PRECONDITION AND WHAT IT SAYS. v413 fitted one greedy set per interchange direction on rows pooled across all five
cells of the determiner-number class (forty rows per direction, against the sixteen a single cell's set is chosen on).
The EACH-TO-SEVERAL direction came out carried by a FOUR-unit set -- attn:11:head:02, attn:07:head:08, attn:04:head:07,
attn:02:head:02 -- at 0.884 to 0.937 on every one of the five cells. The SEVERAL-TO-EACH direction was not carried at
all: thirteen units gave 0.399 to 0.630, worse than each cell's own sixteen-row set. So one direction has a SHARED
site set across five different determiner pairs and the other is cell-specific. That is a claim about SITES.
This rung is the standing DAS follow-up applied to it: interchange localises to sites, DAS asks which SUBSPACE inside
them carries the variable. RANK IS FIXED AT 1 BEFORE THE RUN and a null is not permission to raise it.
DESIGN. One rank-1 direction is fitted on the shared four-unit set over dirB rows POOLED across the five cells, with
the pooled own-C rows as the control, and is then evaluated PER CELL and on all four hypotheses:
    A1  the cell's own held dirB rows -- does the shared subspace carry this cell's effect?
    A2  the cell's second construction (the storm frame) -- does it carry it elsewhere?
    P   the answer-preserving lexical rewrite -- is it spared?
    C   the unrelated behaviour -- is it spared?
The dirA arm is fitted identically on the thirteen-unit set as the comparison: if the shared subspace exists only for
dirB, that is the subspace-level version of what v413 showed at the site level.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_shared_subspace_transfers  the dirB rank-1 direction, fitted on POOLED rows and never on this cell alone,
                          reaches extraction >= floor = 0.80 of the exact-set effect on at least k_cells = 4 of the 5
                          cells' own held dirB rows. Worked example: 0.93, 0.91, 0.88, 0.84, 0.61 gives four
                          clearing, TRUE.                                                                 prior 55%
  pred_b_A2_carries       the same direction reaches >= 0.80 on at least 3 of the 5 cells' A2 (storm-frame) rows.
                          A1 alone is not selectivity; this is the second hypothesis and it uses a construction the
                          direction was not fitted on.                                                    prior 45%
  pred_c_P_and_C_spared   own-C CE damage UB975 <= c_ub = 0.01 AND P damage UB975 <= 0.05 on at least 4 of 5 cells.
                          A subspace that carries the variable must spare the answer-preserving edit and the
                          unrelated behaviour; without this the transfer number means nothing.            prior 55%
  pred_d_dirA_does_not    the dirA direction, fitted the same way on its thirteen-unit set, reaches >= 0.80 on FEWER
                          cells than the dirB direction does. This is the subspace-level version of v413's site-level
                          result and it is registered as a COMPARISON so that a uniformly good or uniformly bad run
                          cannot be read as confirming it.                                                prior 65%
  pred_e_no_error         every cell returns a receipt for both directions; an error fails this predicate rather than
                          being dropped.                                                                  prior 90%
COUNTING. Nothing here is counted. The determiner class has one four-row pass and no family, so it has no entry to
add to; this rung is about what the sites contain.
Smoke: V415_SMOKE=<out.json> (CPU, V415_SMOKE_ROWS=4 per split).
"""
from __future__ import annotations

import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_direction_das_v415_result.json"
CELLS = ("determiner_number_crates", "determiner_number_ropes_d", "determiner_number_barrels_d",
         "determiner_number_ropes", "determiner_number_barrels")
UNITS = {"B": ["attn:11:head:02", "attn:07:head:08", "attn:04:head:07", "attn:02:head:02"],
         "A": ["attn:11:head:02", "attn:07:head:08", "attn:04:head:07", "attn:03:head:06", "attn:15:head:01",
               "attn:16:head:08", "attn:10:head:05", "attn:04:head:00", "attn:04:head:04", "attn:04:head:03",
               "attn:13:head:06", "attn:12:head:08", "attn:00:head:03"]}
RANK = 1                     # fixed in advance; a null is not permission to raise it
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"floor": 0.8, "c_ub": 0.01, "p_ub": 0.05, "k_cells": 4, "k_a2": 3, "rank": RANK}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_direction_das_v415", "cells": len(CELLS), "rank": RANK,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * STEPS, "model_updates": 0, "fit_parameters": 2 * len(UNITS["A"]) * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    D = R.get("directions", {})
    ok = bool(D) and all("error" not in c for d in D.values() for c in d.values())
    def clears(d, key):
        return [n for n, c in D.get(d, {}).items()
                if "error" not in c and c.get(key) is not None and c[key] >= B["floor"]]
    a = ok and len(clears("B", "A1_extraction")) >= B["k_cells"]
    b = ok and len(clears("B", "A2_extraction")) >= B["k_a2"]
    spared = [n for n, c in D.get("B", {}).items()
              if "error" not in c and c.get("C_ub975") is not None and c["C_ub975"] <= B["c_ub"]
              and c.get("P_ub975") is not None and c["P_ub975"] <= B["p_ub"]]
    c = ok and len(spared) >= B["k_cells"]
    d = ok and len(clears("A", "A1_extraction")) < len(clears("B", "A1_extraction"))
    return {"pred_a_shared_subspace_transfers": bool(a), "pred_b_A2_carries": bool(b),
            "pred_c_P_and_C_spared": bool(c), "pred_d_dirA_does_not": bool(d), "pred_e_no_error": bool(ok)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V415_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V415_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    V = dict(valid_only=True)
    ext = lambda p, units, q=None: round(g.recovery(p, g.patched_axis(backend, p, list(units), q=q)), 3)

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float()
                                for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0)
                for u in units}

    def dmg(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975")}

    rows_of = {}
    for name in CELLS:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{name}")
        a1, a2, pp, cc = (g.rows_of(m, "A1"), g.rows_of(m, "A2"), g.rows_of(m, "P"), g.rows_of(m, "C"))
        rows_of[name] = {"A_fit": cut(a1[0::4]), "A_held": cut(a1[2::4]),
                         "B_fit": cut(a1[1::4]), "B_held": cut(a1[3::4]),
                         "A2_held": cut(a2[2::4] + a2[3::4]), "P_held": cut(pp[2::4] + pp[3::4]),
                         "C_fit": cut(cc[0::4] + cc[1::4]), "C_held": cut(cc[2::4] + cc[3::4])}

    directions = {}
    for d in ("B", "A"):
        pooled = [r for name in CELLS for r in rows_of[name][f"{d}_fit"]]
        pooled_c = [r for name in CELLS for r in rows_of[name]["C_fit"]]
        P_fit, P_cfit = g.prepare(backend, pooled, **V), g.prepare(backend, pooled_c)
        units = UNITS[d]
        mu = mu_of(P_fit, units)
        q, _hist = g.fit_block_subspace_constrained(backend, P_fit, units, rank=RANK, steps=steps, lr=LR, seed=0,
                                                    complement_weight=CW, controls=(P_cfit,), control_weight=LAM, mu=mu)
        per = {}
        for name in CELLS:
            try:
                p1 = g.prepare(backend, rows_of[name][f"{d}_held"], **V)
                p2 = g.prepare(backend, rows_of[name]["A2_held"], **V)
                pp_ = g.prepare(backend, rows_of[name]["P_held"])
                pc = g.prepare(backend, rows_of[name]["C_held"])
                e1, e2 = ext(p1, units), ext(p2, units)
                cd, pd = dmg(pc, units, q, mu), dmg(pp_, units, q, mu)
                per[name] = {"exact_A1": e1, "exact_A2": e2,
                             "A1_extraction": round(ext(p1, units, q=q) / e1, 3) if abs(e1) > 1e-6 else None,
                             "A2_extraction": round(ext(p2, units, q=q) / e2, 3) if abs(e2) > 1e-6 else None,
                             "C_damage": cd, "C_ub975": cd["ce_ub975"], "P_damage": pd, "P_ub975": pd["ce_ub975"]}
            except Exception as err:                               # noqa: BLE001 - recorded, never silently dropped
                per[name] = {"error": f"{type(err).__name__}: {err}"}
        directions[d] = per
        print(f"[dir{d}] {len(units)} units, {round(time.perf_counter() - t0, 1)}s", flush=True)

    predictions = PREDS({"directions": directions})
    result = {"predictions": predictions, "schema": "unit_direction_das_v415",
              "candidate_id": "corpus.unit_direction_das_v415", "bars": BARS, "rank": RANK,
              "units": UNITS, "directions": directions, "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    target = Path(smoke) if smoke else OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions,
                      "dirB": {k: {kk: v.get(kk) for kk in ("A1_extraction", "A2_extraction", "C_ub975", "P_ub975")}
                               for k, v in directions["B"].items()},
                      "dirA": {k: v.get("A1_extraction") for k, v in directions["A"].items()},
                      "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
