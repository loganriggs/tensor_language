#!/usr/bin/env python3
# BQGATE: five frozen predictions; cells, unit budget and bars fixed before the run.
"""v413: the direction question again, with the sample-size confound removed.

WHY v411 COULD NOT ANSWER IT. The determiner class shows a systematic split at row 2 -- dirB 0.968 to 0.987 on every
cell, dirA 0.768 to 0.819 -- and row 2 takes the MIN, so dirA decides all five verdicts and the one cell that passed
did so by 0.019. v411 asked whether a greedy set chosen on dirA rows ALONE would reach dirA's effect, which would make
the asymmetry a selection artefact. It came back 3/5 with dirA mostly COLLAPSING (0.787 -> 0.257, 0.768 -> 0.394), and
the dirB control I had registered explained why: dirB fell too (0.978 -> 0.926, 0.987 -> 0.904) despite having no
headroom. Splitting by direction halves the fit rows from sixteen to eight, and greedy selection on eight rows is
worse than on sixteen whichever direction they carry. The rung measured sample size.
WHAT THIS ONE CHANGES. The dirA rows are POOLED ACROSS ALL FIVE CELLS of the class -- five cells x eight dirA fit rows
is forty, against the sixteen a single cell's joint set is chosen on -- and ONE greedy set is fitted on that pool, then
evaluated on each cell's own dirA held rows. The dirB arm is built identically as the control. Now the direction-only
set has MORE data than the joint set rather than less, so if dirA still cannot be carried it is not for want of rows.
This also uses the one instrument the corpus has tested most today: pooling across shapes has run eleven times, and the
five cells here share a construction, a frame and a P control, differing only in the determiner pair and the noun.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_dirA_selection   in at least k_cells = 3 of the 5 cells, the POOLED dirA set reaches recovery >= floor = 0.80
                          on that cell's dirA held rows. Worked example: 0.88, 0.85, 0.83, 0.81, 0.40 gives four
                          clearing, TRUE. This is the corpus's own row-2 bar asked of a set with forty rows behind it.
                                                                                                          prior 55%
  pred_b_dirA_improves    the pooled dirA set beats the joint set's dirA recovery by >= gain = 0.03 on at least 3 of
                          5 cells. Improvement, not merely clearing, is what separates SELECTION from MECHANISM: a
                          cell already at 0.819 could clear 0.80 without the pooling doing anything.      prior 50%
  pred_c_dirB_control     the pooled dirB set does NOT improve dirB by 0.03 on a majority of cells. dirB is already
                          0.968-0.987 so there is nothing to gain; if pooling lifted it anyway, the dirA gain would be
                          about data volume rather than about direction. This is the same control that caught v411 and
                          it is the reason that rung is filed inconclusive rather than as a finding.      prior 75%
  pred_d_no_error         every cell returns a receipt; an error fails this predicate rather than being dropped.
                                                                                                          prior 90%
  pred_e_outlier_stays_low  determiner_number_barrels -- weak on BOTH directions at 0.329 / 0.528 -- does NOT reach
                          0.80 on dirA even with the pooled set. It is the one cell I expect to be a real failure
                          rather than a fitting artefact, and registering it separately stops a uniform pass from
                          hiding it.                                                                      prior 60%
COUNTING. Nothing here is counted; this is an instrument reading. If pred_a and pred_b both hold, the determiner
class's row-2 verdicts are a selection artefact of a sixteen-row fit and the corpus's single determiner "pass" is not
qualitatively different from its four "failures" -- which is a statement about the battery, not about the model, and
I would take it to the board as proposal (xiii) evidence rather than act on it alone.
Smoke: V413_SMOKE=<out.json> (CPU, V413_SMOKE_ROWS=4 per split).
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_direction_split_v413_result.json"
CELLS = (("determiner_number_crates", 0.819, 0.978),
         ("determiner_number_ropes_d", 0.796, 0.981),
         ("determiner_number_barrels_d", 0.787, 0.968),
         ("determiner_number_ropes", 0.768, 0.987),
         ("determiner_number_barrels", 0.329, 0.528))
OUTLIER = "determiner_number_barrels"
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
BARS = {"floor": 0.8, "gain": 0.03, "k_cells": 3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_direction_split_v413", "cells": len(CELLS),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    C = R.get("cells", {})
    ok = bool(C) and all("error" not in c for c in C.values())
    clears = [n for n, c in C.items() if "error" not in c and (c.get("dirA_only") or 0) >= B["floor"]]
    gains = [n for n, c in C.items() if "error" not in c
             and c.get("dirA_only") is not None and c["dirA_only"] - c["dirA_both"] >= B["gain"]]
    bgains = [n for n, c in C.items() if "error" not in c
              and c.get("dirB_only") is not None and c["dirB_only"] - c["dirB_both"] >= B["gain"]]
    out = C.get(OUTLIER, {})
    return {"pred_a_dirA_selection": bool(ok and len(clears) >= B["k_cells"]),
            "pred_b_dirA_improves": bool(ok and len(gains) >= B["k_cells"]),
            "pred_c_dirB_control": bool(ok and len(bgains) <= len(C) // 2),
            "pred_d_no_error": bool(ok),
            "pred_e_outlier_stays_low": bool(ok and (out.get("dirA_only") or 0) < B["floor"])}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V413_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V413_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units = (3, 3) if smoke else (POOL, MAX_UNITS)
    V = dict(valid_only=True)
    ext = lambda p, units: round(g.recovery(p, g.patched_axis(backend, p, list(units))), 3)

    pooled_fit = {"A": [], "B": []}
    per_cell_rows = {}
    for name, a_both, b_both in CELLS:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{name}")
        rows = g.rows_of(m, "A1")
        per_cell_rows[name] = {"A_held": cut(rows[2::4]), "B_held": cut(rows[3::4])}
        pooled_fit["A"] += cut(rows[0::4])
        pooled_fit["B"] += cut(rows[1::4])
    sets = {}
    for d in ("A", "B"):
        P = g.prepare(backend, pooled_fit[d], **V)
        _s, _r, greedy = g.greedy_heads(backend, P, pool=pool, target=TARGET, min_gain=MIN_GAIN, max_units=max_units)
        sets[d] = list(greedy["chosen"])
        print(f"[pooled dir{d}] {len(sets[d])} units on {len(P.base_batch.row_ids)} rows, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    cells = {}
    for name, a_both, b_both in CELLS:
        try:
            res = {"dirA_both": a_both, "dirB_both": b_both,
                   "dirA_n_units": len(sets["A"]), "dirB_n_units": len(sets["B"])}
            for d in ("A", "B"):
                p = g.prepare(backend, per_cell_rows[name][f"{d}_held"], **V)
                res[f"dir{d}_only"] = ext(p, sets[d])
            cells[name] = res
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            cells[name] = {"error": f"{type(err).__name__}: {err}"}
        print(f"[{name}] {round(time.perf_counter() - t0, 1)}s", flush=True)

    predictions = PREDS({"cells": cells})
    result = {"predictions": predictions, "schema": "unit_direction_split_v413",
              "candidate_id": "corpus.unit_direction_split_v413", "bars": BARS,
              "recipe": {"pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units,
                         "pooled_dirA_units": sets["A"], "pooled_dirB_units": sets["B"]},
              "cells": cells, "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    target = Path(smoke) if smoke else OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions,
                      "cells": {k: {kk: v[kk] for kk in ("dirA_both", "dirA_only", "dirB_both", "dirB_only")
                                    if kk in v} for k, v in cells.items()},
                      "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
