#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the fifty-ninth two spec-authored behaviours, plus one instrument.
"""v587: row 2 is a DIRECTION-SYMMETRY gate -- is the weak direction rescuable, or genuinely not carried?

WHAT THE LAST THREE RUNGS ESTABLISHED. v575, v581 and v585 lifted 25 backlog cells. Rows 3, 4 and 5 pass almost
everywhere (19 of 25, 24 of 25, 25 of 25); ROW 2 passes only 6 of 25 and is the single thing gating countable
behaviours. I then read the row definition rather than assuming it: row 2 is min(extraction_dirA, extraction_dirB)
at or above 0.80, so it is the INTERCHANGE-DIRECTION SYMMETRY row. Every one of the 25 cells has a STRONG direction
(max 0.75 to 1.13); only the weaker one varies, and the mean absolute gap is 0.117 for the five amid cells against
0.247 for the other twenty. That also retires the story I was tempted by after v581 -- that the countable behaviours
are picked out by a rare readout token. Two non-amid cells pass row 2, the token merely correlates with symmetry,
and the thing being measured is direction.
THE QUESTION. A joint fit uses both directions rows at once, so a set chosen to satisfy the strong direction may
simply not be the set the weak direction needs. If that is what is happening, fitting each direction ON ITS OWN
should rescue the weak one. If instead the weak direction is not carried by any small head set, it stays weak no
matter how it is fitted -- and the corpus note that row-2 misses are direction asymmetry becomes a statement about
the model rather than about my fitting.
THE CONFOUND, AND THE CONTROL THAT CAN FAIL. A within-direction fit uses HALF the fit rows a joint fit uses, so a
rescue could be an artifact of the smaller fit set rather than of direction. Every cell therefore also gets a SIZE
CONTROL: a random half of the joint fit rows, the SAME SIZE as one direction half, evaluated on both held
directions. If the direction-matched fit does no better than that random half, the effect is sample size and there
is no direction story. That control has room to fail in both directions and is the reason this rung is worth GPU.
SIX CELLS, chosen as the widest asymmetries with a strong direction already in hand: off_with (1.099/0.581),
from_with (0.672/1.102), adjective in_with (0.700/1.063), adjective with_for (1.132/0.722), adjective under_for
(1.301/0.791), and in_amid (1.087/0.772) -- the one amid cell that FAILED row 2, which is an internal check that the
token is not doing the work.
WHAT A RESCUE WOULD AND WOULD NOT BUY. A site that carries the variable in ONE direction is a NARROWER object than a
symmetric one, and I am not going to present a rescued direction as equivalent to a row-2 pass. This rung tests a
different, weaker claim on purpose, and says so.
NOTHING IS COUNTED. The count stays 139.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_weak_direction_recovers at least k_weak = 3 of the six cells have their WEAK direction reach
                       ext_min = 0.80 when fitted on that direction alone.                            prior 45%
  pred_b_strong_direction_holds at least k_strong = 5 of the six have their STRONG direction still reach 0.80 under
                       its own fit -- a sanity bar, since halving the fit rows should not break what already
                       worked.                                                                        prior 85%
  pred_c_beats_size_control at least k_beats = 3 cells have the direction-matched fit beat the equal-sized RANDOM
                       half on the weak direction by at least margin = 0.05. This is what separates a direction
                       effect from a fit-set-size effect, and it fails if halving is what matters. prior 45%
  pred_d_unit_sets_differ at least k_differ = 4 of the six have their two per-direction head sets differ at a
                       Jaccard of jac_max = 0.7 or below, so the asymmetry is about WHICH heads carry each
                       direction rather than about one direction being harder everywhere.             prior 60%
  pred_e_all_measured  zero cells error.                                                              prior 90%
HOW IT READS. a and c both true: the weak direction is carried, the joint fit was the problem, and 19 row-2 failures
become candidates through per-direction fits -- the largest single expansion of countable inventory available, at
the cost of a narrower claim per circuit. a true and c FALSE: the rescue is sample size, not direction, and I would
be manufacturing passes by halving the fit set -- the most important thing this rung can catch. a FALSE with b true:
the weak direction is not carried by any small set while the strong one plainly is, row 2 is measuring something
real about the model, and the 19 failures are honest failures. d FALSE with a true: both directions need the same
heads and the asymmetry is about strength rather than location.
SCOPE. Six cells, one fit per direction plus one size control each, held-out extraction only -- no removal rows, no
counting.
Smoke: V587_SMOKE=<out.json> (CPU, V587_SMOKE_ROWS=4, V587_SMOKE_NAMES=<cell>).
"""
from __future__ import annotations

import importlib
import json
import pathlib
import random
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_direction_split_v587_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
CELLS = ("verb_preposition_off_with", "verb_preposition_from_with", "adjective_preposition_in_with",
         "adjective_preposition_with_for", "adjective_preposition_under_for", "verb_preposition_in_amid")
PARENT_DIR = {"verb_preposition_off_with": (1.099, 0.581), "verb_preposition_from_with": (0.672, 1.102),
              "adjective_preposition_in_with": (0.700, 1.063), "adjective_preposition_with_for": (1.132, 0.722),
              "adjective_preposition_under_for": (1.301, 0.791), "verb_preposition_in_amid": (1.087, 0.772)}
NAMES = {c: c for c in CELLS}
NEW = CELLS
INSTR = ()
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "k_weak": 3, "k_strong": 5, "k_beats": 3, "margin": 0.05, "jac_max": 0.7, "k_differ": 4, "tol": 0.05}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000




def _plan():
    return {"candidate_id": "corpus.unit_direction_split_v587", "behaviours": len(CELLS),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "error" not in r}
    ok = bool(good) and set(good) == set(CELLS)
    weak = sum(1 for n in good if good[n]["weak_own_fit"] >= B["ext_min"])
    strong = sum(1 for n in good if good[n]["strong_own_fit"] >= B["ext_min"])
    beats = sum(1 for n in good
                if good[n]["weak_own_fit"] - good[n]["weak_size_control"] >= B["margin"])
    differ = sum(1 for n in good if good[n]["jaccard"] <= B["jac_max"])
    a = ok and weak >= B["k_weak"]
    b = ok and strong >= B["k_strong"]
    c = ok and beats >= B["k_beats"]
    d = ok and differ >= B["k_differ"]
    e = ok
    return {"pred_a_weak_direction_recovers": bool(a), "pred_b_strong_direction_holds": bool(b),
            "pred_c_beats_size_control": bool(c), "pred_d_unit_sets_differ": bool(d),
            "pred_e_all_measured": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V587_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V587_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units = (3, 3) if smoke else (POOL, MAX_UNITS)
    which = [n for n in CELLS if not smoke or n in os.environ.get("V587_SMOKE_NAMES", CELLS[0]).split(",")]
    ext = lambda p, units: round(g.recovery(p, g.patched_axis(backend, p, list(units))), 3)
    R = {}
    for n in which:
        t1 = time.perf_counter()
        try:
            m = importlib.import_module(f"circuit_fast_screen_candidate_{NAMES[n]}")
            rows = g.rows_of(m, "A1")
            V = dict(valid_only=True)
            # rows[0::4] and rows[1::4] are the two FIT directions; rows[2::4] and rows[3::4] the two HELD directions
            fit_a, fit_b = cut(rows[0::4]), cut(rows[1::4])
            held_a, held_b = cut(rows[2::4]), cut(rows[3::4])
            # SIZE CONTROL: a random half of the JOINT fit rows, the same size as one direction half,
            # so a rescue that is really about halving the fit set cannot be read as a direction effect.
            joint = cut(rows[0::4] + rows[1::4])
            rng = random.Random(0)
            ctrl_rows = rng.sample(joint, min(len(fit_a), len(joint)))
            P = {"fit_a": g.prepare(backend, fit_a, **V), "fit_b": g.prepare(backend, fit_b, **V),
                 "held_a": g.prepare(backend, held_a, **V), "held_b": g.prepare(backend, held_b, **V),
                 "fit_ctrl": g.prepare(backend, ctrl_rows, **V)}
            fits = {}
            for key in ("fit_a", "fit_b", "fit_ctrl"):
                _s, _r, greedy = g.greedy_heads(backend, P[key], pool=pool, target=TARGET,
                                                min_gain=MIN_GAIN, max_units=max_units)
                fits[key] = list(greedy["chosen"])
            a_own, b_own = ext(P["held_a"], fits["fit_a"]), ext(P["held_b"], fits["fit_b"])
            a_ctrl, b_ctrl = ext(P["held_a"], fits["fit_ctrl"]), ext(P["held_b"], fits["fit_ctrl"])
            pa, pb = PARENT_DIR[n]
            weak_is_a = pa <= pb
            sa, sb = set(fits["fit_a"]), set(fits["fit_b"])
            jac = round(len(sa & sb) / len(sa | sb), 3) if (sa | sb) else None
            R[n] = {"parent_dirA": pa, "parent_dirB": pb, "weak_is_dirA": bool(weak_is_a),
                    "weak_own_fit": a_own if weak_is_a else b_own,
                    "strong_own_fit": b_own if weak_is_a else a_own,
                    "weak_size_control": a_ctrl if weak_is_a else b_ctrl,
                    "strong_size_control": b_ctrl if weak_is_a else a_ctrl,
                    "n_units_a": len(fits["fit_a"]), "n_units_b": len(fits["fit_b"]),
                    "n_units_ctrl": len(fits["fit_ctrl"]), "jaccard": jac,
                    "seconds": round(time.perf_counter() - t1, 1)}
        except Exception as err:  # noqa: BLE001 - one cell must not lose the batch
            R[n] = {"error": f"{type(err).__name__}: {err}", "seconds": round(time.perf_counter() - t1, 1)}
        print(n, json.dumps(R[n]), round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "cells": R, "schema": "unit_direction_split_v587",
              "candidate_id": "corpus.unit_direction_split_v587", "bars": BARS,
              "recipe": {"pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units},
              "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result["predictions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
