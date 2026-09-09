#!/usr/bin/env python3
# BQGATE: five frozen predictions; cells, unit budget and bars fixed before the run.
"""v411: is the row-2 asymmetry a SELECTION failure or a different mechanism?

WHAT PROMPTED IT. Five cells of the determiner-number class have now been screened. Splitting row 2 by interchange
DIRECTION: dirB runs 0.968 to 0.987 on every cell while dirA runs 0.768 to 0.819, and row 2 takes the MIN, so dirA
decides all five verdicts. The one cell that passed cleared the bar by 0.019 and one that failed missed it by 0.004.
The battery chooses ONE greedy unit set on rows from BOTH directions and then reports recovery per direction. So a
uniform dirB and a weak dirA is consistent with two quite different stories:
  SELECTION  -- the both-direction greedy set is dominated by whatever serves dirB, and dirA has units of its own that
                are never chosen. If so, a set chosen on dirA rows ALONE should reach dirA's own effect.
  MECHANISM  -- dirA's effect is genuinely distributed and no small set carries it, in which case a dirA-only set does
                no better and the asymmetry is a property of the model rather than of the selection step.
This runner chooses a greedy set on dirA FIT rows only (rows[0::4]) and evaluates it on dirA HELD rows (rows[2::4]),
against the both-direction set's dirA recovery already on disk, and does the same for dirB as a control -- if the
dirB-only set does not also improve, the dirA change is about dirA and not about halving the fit data.
CELLS (all five of the class; the numbers in brackets are the both-direction set's dirA / dirB recovery):
    determiner_number_crates      [0.819 / 0.978]   the only four-row pass
    determiner_number_ropes_d     [0.796 / 0.981]
    determiner_number_barrels_d   [0.787 / 0.968]
    determiner_number_ropes       [0.768 / 0.987]
    determiner_number_barrels     [0.329 / 0.528]   the outlier: weak on BOTH directions
NOTHING HERE CHANGES THE BATTERY. It does not touch a bar, a predicate or a verdict; it is a diagnostic that reads the
same rows a different way, and its output is a number to put beside row 2 rather than a replacement for it.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_dirA_selection   in at least k_cells = 3 of the 5 cells, a greedy set chosen on dirA rows alone reaches
                          recovery >= floor = 0.80 on dirA held rows, where the both-direction set reached 0.768 to
                          0.819. Worked example: dirA-only recoveries of 0.88, 0.85, 0.83, 0.81 and 0.40 give four
                          clearing the floor, TRUE; 0.80 is the corpus's own bar so this is the same standard row 2
                          applies, asked of a set that was allowed to see only dirA.                     prior 55%
  pred_b_dirA_improves    the dirA-only set beats the both-direction set's dirA recovery by at least gain = 0.03 on
                          at least 3 of 5 cells. Registered separately from pred_a because a cell already at 0.819
                          could clear 0.80 without improving at all, and improvement is what distinguishes SELECTION
                          from MECHANISM.                                                                 prior 50%
  pred_c_dirB_control     the dirB-only set does NOT improve dirB by 0.03 on a majority of cells -- dirB is already
                          0.968 to 0.987, so there is no headroom, and if halving the fit data improved it anyway the
                          comparison would be meaningless. This is the control that makes pred_b interpretable.
                                                                                                          prior 75%
  pred_d_no_error         every cell returns a receipt; an error is recorded per cell and fails this predicate.
                                                                                                          prior 90%
  pred_e_outlier_stays_low  determiner_number_barrels, the cell that is weak on BOTH directions (0.329 / 0.528),
                          does NOT reach 0.80 on dirA even with a dirA-only set. It is the one cell whose failure I
                          expect to be real rather than a selection artefact, and registering it separately keeps a
                          uniform pass from hiding it.                                                    prior 65%
COUNTING. Nothing here is counted; this is an instrument reading, not a circuit.
Smoke: V411_SMOKE=<out.json> (CPU, V411_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_direction_split_v411_result.json"
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
    return {"candidate_id": "corpus.unit_direction_split_v411", "cells": len(CELLS),
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
    smoke = os.environ.get("V411_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V411_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units = (3, 3) if smoke else (POOL, MAX_UNITS)
    V = dict(valid_only=True)
    ext = lambda p, units: round(g.recovery(p, g.patched_axis(backend, p, list(units))), 3)

    cells = {}
    for name, a_both, b_both in CELLS:
        try:
            m = importlib.import_module(f"circuit_fast_screen_candidate_{name}")
            rows = g.rows_of(m, "A1")
            # the battery's own split: fit = rows[0::4] + rows[1::4], held = rows[2::4] + rows[3::4],
            # and within those, index 0/2 are one interchange direction and 1/3 are the other.
            pieces = {"A_fit": cut(rows[0::4]), "A_held": cut(rows[2::4]),
                      "B_fit": cut(rows[1::4]), "B_held": cut(rows[3::4])}
            P = {k: g.prepare(backend, v, **V) for k, v in pieces.items()}
            res = {"dirA_both": a_both, "dirB_both": b_both}
            for d in ("A", "B"):
                _s, _r, greedy = g.greedy_heads(backend, P[f"{d}_fit"], pool=pool, target=TARGET,
                                                min_gain=MIN_GAIN, max_units=max_units)
                units = list(greedy["chosen"])
                res[f"dir{d}_only"] = ext(P[f"{d}_held"], units)
                res[f"dir{d}_units"] = units
                res[f"dir{d}_n_units"] = len(units)
            cells[name] = res
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            cells[name] = {"error": f"{type(err).__name__}: {err}"}
        print(f"[{name}] {round(time.perf_counter() - t0, 1)}s", flush=True)

    predictions = PREDS({"cells": cells})
    result = {"predictions": predictions, "schema": "unit_direction_split_v411",
              "candidate_id": "corpus.unit_direction_split_v411", "bars": BARS,
              "recipe": {"pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units},
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
