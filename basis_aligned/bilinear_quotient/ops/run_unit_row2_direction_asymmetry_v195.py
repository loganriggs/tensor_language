#!/usr/bin/env python3
# BQGATE: five frozen predictions; EVEN sets copied from the v113 receipt; ODD-fit greedy with the v113 parameters; bars fixed before the run.
"""v195: the battery's row-2 misses -- is 'EVEN fit / ODD eval' a split test or a DIRECTION test?

Rows alternate interchange direction, so parity == direction on every family (checked: degree_result EVEN = infinitival_to_finite
x16, ODD = finite_to_infinitival x16; finiteness EVEN = nonfinite_to_finite, ODD = finite_to_nonfinite). Row 2 ('ODD exact-set
recovery >= 0.80 for a set chosen on EVEN') therefore asks whether the direction-A head set carries direction B. The v113 misses:
degree_result 0.856 -> 0.639, finiteness_selection 0.844 -> 0.641, correlative_both_either 0.889 -> 0.751, polarity_state
0.865 -> 0.790 (quantifier_number 0.834 passes; control).

Magnitudes printed BEFORE writing (CPU probes; disclosed -- preds a-c are partly pre-measured on 8-16 rows):
    greedy fitted ON ODD (pool 12, v113 target/min_gain/max_units): degree_result plateaus at 0.619 in-sample with 5 units and
    gives 0.823 on EVEN; finiteness 0.683 in-sample (9 units), 0.844 on EVEN; Jaccard with the EVEN set 0.57 / 0.55.
    -> direction B is harder for ANY <=14-head set, whichever direction chose it. Where is the rest? exact patch of ALL 162 heads
    recovers 1.0 both directions; ALL 18 MLP outputs 0.69 / 0.60 on ODD (0.78 / 0.61 EVEN); the 90 heads of layers 4-13 give
    0.899 / 0.873 on ODD (0.974 / 0.912 EVEN). So the direction-B remainder sits in MANY heads below the greedy's min_gain
    (0.02), not in the MLPs. ODD single-head recoveries fall off faster (degree 0.30, 0.11, 0.04, 0.03, 0.02) than EVEN's
    (0.45, 0.19, 0.07, 0.07, 0.06 from the v113 receipt).
Behaviours: the four row-2 misses + quantifier_number (control, both directions >= 0.80 expected).

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_direction_not_split -> on degree_result and finiteness_selection the ODD-fitted greedy set (pool 20, target 0.88,
                          min_gain 0.02, max 14 -- the v113 parameters) recovers < 0.80 IN-SAMPLE on ODD and within [0.75, 1.10]
                          on EVEN (out-of-sample, the other direction): the miss is the direction, not the split.
  pred_b_remainder_is_heads -> on all four misses, the exact patch of the 90 heads of layers 4-13 recovers >= 0.85 on ODD, and
                          the exact patch of all 18 MLP outputs recovers <= 0.75 on ODD.
  pred_c_distributed     -> on all four misses, a relaxed ODD-fitted greedy (pool 40, min_gain 0.005, max 30 units, target 0.88)
                          reaches >= 0.80 in-sample on ODD (direction B is carried by a long tail of small heads, not absent),
                          using <= 30 units. Unprobed: this is the run's own outcome.
  pred_d_control_symmetric -> quantifier_number: the ODD-fitted v113-parameter greedy recovers >= 0.80 in-sample AND >= 0.80
                          on EVEN, and the ODD greedy's Jaccard with the v113 EVEN set >= 0.4.
  pred_e_instrument      -> the v113 EVEN sets reproduce their receipts' EVEN/ODD recoveries within +-0.03 on all five.
Reading if a-c hold: row 2 of the battery is a direction-asymmetry statement -- the marked->unmarked (or vice versa) direction is
carried by a distributed head tail that a 14-head greedy with min_gain 0.02 cannot collect; the 'misses' are one circuit each with an
asymmetric head concentration between directions. Protocol amendment for future lifts: fit and evaluate row 2 WITHIN direction
(interleaved split [0::4]+[1::4] vs [2::4]+[3::4]) and report the per-direction concentration separately.
Smoke: V195_SMOKE=<out.json> (CPU, V195_SMOKE_ROWS=4 per split, V195_SMOKE_NAMES=degree_result, pools 3/4).
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
OUT = ROOT / "circuits/followups/unit_row2_direction_asymmetry_v195_result.json"
V113 = ROOT / "circuits/followups/unit_tier3_batch_row2_v113_result.json"
NAMES = {"degree_result": "degree_result", "finiteness_selection": "finiteness", "correlative_both_either": "both_either",
         "polarity_state": "polarity_state", "quantifier_number": "quantifier_number"}
MISSES = ("degree_result", "finiteness_selection", "correlative_both_either", "polarity_state")
PROBED = ("degree_result", "finiteness_selection")
CONTROL = "quantifier_number"
POOL, TARGET, MIN_GAIN, MAX_UNITS = 20, 0.88, 0.02, 14
RELAX_POOL, RELAX_MIN_GAIN, RELAX_MAX = 40, 0.005, 30
BARS = {"odd_fit_in_sample_max": 0.80, "odd_fit_even_band": [0.75, 1.10], "mid_heads_odd_min": 0.85, "mlps_odd_max": 0.75,
        "relaxed_odd_min": 0.80, "relaxed_max_units": 30, "control_min": 0.80, "control_jaccard_min": 0.4, "repro_tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    return {"candidate_id": "corpus.unit_row2_direction_asymmetry_v195", "behaviours": 5, "constructions": 1,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    inb = lambda x, b: x is not None and b[0] <= x <= b[1]
    have = lambda ns: [n for n in ns if n in R]
    ok = bool(have(NAMES))  # an empty receipt fails everything (all() over nothing would pass)
    a = ok and bool(have(PROBED)) and all(R[n]["odd_fit_in_sample"] < B["odd_fit_in_sample_max"] and inb(R[n]["odd_fit_on_even"], B["odd_fit_even_band"]) for n in have(PROBED))
    b = ok and bool(have(MISSES)) and all(R[n]["mid_heads_odd"] >= B["mid_heads_odd_min"] and R[n]["mlps_odd"] <= B["mlps_odd_max"] for n in have(MISSES))
    c = ok and bool(have(MISSES)) and all(R[n]["relaxed_odd_in_sample"] >= B["relaxed_odd_min"] and R[n]["relaxed_n_units"] <= B["relaxed_max_units"] for n in have(MISSES))
    d = ok and CONTROL in R and R[CONTROL]["odd_fit_in_sample"] >= B["control_min"] and R[CONTROL]["odd_fit_on_even"] >= B["control_min"] and R[CONTROL]["jaccard_odd_vs_even_set"] >= B["control_jaccard_min"]
    e = ok and all(abs(R[n]["even_set_ext_even"] - R[n]["parent_even"]) <= B["repro_tol"] and abs(R[n]["even_set_ext_odd"] - R[n]["parent_odd"]) <= B["repro_tol"] for n in have(NAMES))
    return {"pred_a_direction_not_split": bool(a), "pred_b_remainder_is_heads": bool(b), "pred_c_distributed": bool(c),
            "pred_d_control_symmetric": bool(d), "pred_e_instrument": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V195_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V195_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, relax_pool, max_units, relax_max = (3, 4, 3, 4) if smoke else (POOL, RELAX_POOL, MAX_UNITS, RELAX_MAX)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V195_SMOKE_NAMES", "degree_result").split(",")]
    parent = json.loads(V113.read_text())["behaviours"]
    all_heads = g.all_head_units()
    mid_heads = [u for u in all_heads if 4 <= int(u[5:7]) <= 13]
    all_mlps = [f"mlp:{l:02d}" for l in range(18)]
    ext = lambda p, units: round(g.recovery(p, g.patched_axis(backend, p, list(units))), 3)
    jac = lambda a, b: round(len(set(a) & set(b)) / len(set(a) | set(b)), 3) if (set(a) | set(b)) else None

    R = {}
    for name in which:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{NAMES[name]}")
        rows = g.rows_of(m, "A1")
        P = {"even": g.prepare(backend, cut(rows[0::2])), "odd": g.prepare(backend, cut(rows[1::2]))}
        u_even = list(parent[name]["units"])
        singles, ranked, greedy = g.greedy_heads(backend, P["odd"], pool=pool, target=TARGET, min_gain=MIN_GAIN, max_units=max_units)
        u_odd = list(greedy["chosen"])
        _, ranked_r, greedy_r = g.greedy_heads(backend, P["odd"], pool=relax_pool, target=TARGET, min_gain=RELAX_MIN_GAIN, max_units=relax_max, units=ranked[:relax_pool])
        u_relax = list(greedy_r["chosen"])
        R[name] = {"direction_even": str(P["even"].rows[0].get("direction_id")), "direction_odd": str(P["odd"].rows[0].get("direction_id")),
                   "even_set": u_even, "parent_even": parent[name]["extraction_even"], "parent_odd": parent[name]["extraction_odd"],
                   "even_set_ext_even": ext(P["even"], u_even), "even_set_ext_odd": ext(P["odd"], u_even),
                   "odd_set": u_odd, "odd_fit_in_sample": ext(P["odd"], u_odd), "odd_fit_on_even": ext(P["even"], u_odd), "jaccard_odd_vs_even_set": jac(u_odd, u_even),
                   "relaxed_set": u_relax, "relaxed_n_units": len(u_relax), "relaxed_odd_in_sample": ext(P["odd"], u_relax), "relaxed_on_even": ext(P["even"], u_relax),
                   "mid_heads_odd": ext(P["odd"], mid_heads), "mid_heads_even": ext(P["even"], mid_heads), "mlps_odd": ext(P["odd"], all_mlps), "mlps_even": ext(P["even"], all_mlps),
                   "all_heads_odd": ext(P["odd"], all_heads),
                   "odd_singles_top10": {u: round(singles[u], 3) for u in ranked[:10]}, "odd_n_singles_ge_0p02": sum(v >= 0.02 for v in singles.values()), "odd_n_singles_ge_0p05": sum(v >= 0.05 for v in singles.values()),
                   "n_rows": {sp: len(P[sp].rows) for sp in ("even", "odd")}}
        print(name, json.dumps({k: v for k, v in R[name].items() if k not in ("even_set", "odd_set", "relaxed_set", "odd_singles_top10")}), round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_row2_direction_asymmetry_v195", "candidate_id": "corpus.unit_row2_direction_asymmetry_v195",
              "bars": BARS, "protocol": {"pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units, "relaxed": [relax_pool, RELAX_MIN_GAIN, relax_max]},
              "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
