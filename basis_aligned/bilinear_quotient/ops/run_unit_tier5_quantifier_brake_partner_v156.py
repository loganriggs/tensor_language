#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: which patched partner makes the complement of 11:03's quantifier write oppose the margin?

v155: on quantifier's seven, 11:03 alone is linear (rank-1 0.289 vs exact 0.285) but 11:03 rank-1 WITH the other six
blocks exact-patched gives 0.901 vs exact 0.844 (+0.057); mlp:08 is the only MLP whose own complement is negative
(-0.022). Lexical's seven is flat everywhere. Test the partner: (i) pairs -- units = {11:03, block j}; 11:03 rank-1
(own diff-in-means from parity 0) + block j exact, minus the pair's exact; (ii) leave-one-out -- 11:03 rank-1 + the
five other blocks exact, dropping block j, minus that set's exact. Quantifier seven (v152 order); lexical seven's
11:03 + mlp:08 pair as the control. q from parity 0, evaluated on held-out parity 1.
Row population (ops/row_population.py): lexical A1 p0 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 15}',
p1 'rows 16 ... {1: 16}'; quantifier A1 p0 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}', p1 same.

Registered before the run:
  pred_a_instrument   11:03 rank-1 + six exact reproduces v155's 0.901 within 0.02; 11:03 alone rank-1 reproduces 0.289 within 0.02
  pred_b_mlp08_pair   the 11:03 + mlp:08 pair overshoot >= 0.6 x 0.057 = 0.034
  pred_c_other_pairs  every other pair overshoot (11:03 with mlp:09, mlp:10, 07:08, 08:01, mlp:07) < 0.02
  pred_d_leave_out    dropping mlp:08 cuts the six-block overshoot to <= 0.029 (half); dropping any other block changes it by < 0.02
  pred_e_lex_control  lexical seven's 11:03 + mlp:08 pair overshoot within +-0.02
Reported, unregistered: every pair's exact, rank-1 and complement recovery; leave-one-out exacts.
Prior: b/c/d are one hypothesis (a 11:03 x mlp:08 interaction); if the overshoot only appears with several partners together
(b fails, d fails) the interaction is with the SUMMED patched state, which the next rung would dose.
Smoke: V156_SMOKE=<out.json> -> CPU, 4 rows. The 4-row smoke gave every pair 0.016-0.046 (largest with 07:08, the biggest
partner) and leave-one-out flat at 0.06 -- the summed-state alternative; bars kept as registered.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_batch_v112 as v112
import run_unit_tier5_carrier_relay_v120 as v120
import run_unit_tier5_near_carrier_heads_v123 as v123
import run_unit_tier5_near_value_source_v131 as v131
import run_unit_tier5_near_value_mid_remainder_v132 as v132

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_quantifier_brake_partner_v156_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
REPRO_TOL, PAIR_MIN, OTHER_MAX, DROP_MAX, DROP_OTHER, LEX_TOL = 0.02, 0.034, 0.02, 0.029, 0.02, 0.02
V155_SIX, V155_ALONE, HUB = 0.901, 0.289, "attn:11:head:03"
SETS = {"quant_seven": ("quantifier_number", ("mlp:10", "attn:11:head:03", "mlp:09", "mlp:08", "attn:07:head:08", "attn:08:head:01", "mlp:07"), 0.969),
        "shared_four": ("quantifier_number", ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10"), 0.673),
        "lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_quantifier_brake_partner_v156", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V156_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:4]) if smoke else (lambda rows: rows)
    P = {}
    for n in ("lexical_number_pp", "quantifier_number"):
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")
        P[n] = {"p0": g.prepare(backend, cut(a1[0::2])), "p1": g.prepare(backend, cut(a1[1::2]))}
    R = {}
    for sname, (n, units, _) in SETS.items():
        if sname == "shared_four":
            continue
        units = list(units)
        hub_q = g.block_diff_in_means(backend, P[n]["p0"], [HUB])   # 128-d: the 11:03 slice alone (lexical's 11:heads block also holds 11:02)
        prep = P[n]["p1"]
        def rec(us, q=None, complement=False):
            return round(g.recovery(prep, g.patched_axis(backend, prep, us, q=q, complement=complement)), 3)
        others = [u for u in units if u != HUB]
        S = {"alone": {"exact": rec([HUB]), "dim": rec([HUB], hub_q), "comp": rec([HUB], hub_q, True)},
             "six": {"exact": rec(units), "dim": rec(units, hub_q)}, "pairs": {}, "leave_out": {}}
        S["six"]["overshoot"] = round(S["six"]["dim"] - S["six"]["exact"], 3)
        for j in others:
            us = [HUB, j]
            e, d, c = rec(us), rec(us, hub_q), rec(us, hub_q, True)
            S["pairs"][j] = {"exact": e, "dim": d, "comp": c, "overshoot": round(d - e, 3)}
            if sname == "quant_seven":
                us = [u for u in units if u != j]
                e, d = rec(us), rec(us, hub_q)
                S["leave_out"][j] = {"exact": e, "dim": d, "overshoot": round(d - e, 3)}
        S["rows"] = len(prep.base_batch.row_ids)
        R[sname] = S
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    Qs = R["quant_seven"]
    pred_a = abs(Qs["six"]["dim"] - V155_SIX) <= REPRO_TOL and abs(Qs["alone"]["dim"] - V155_ALONE) <= REPRO_TOL
    pred_b = Qs["pairs"]["mlp:08"]["overshoot"] >= PAIR_MIN
    pred_c = all(v["overshoot"] < OTHER_MAX for j, v in Qs["pairs"].items() if j != "mlp:08")
    six_over = Qs["six"]["overshoot"]
    pred_d = Qs["leave_out"]["mlp:08"]["overshoot"] <= DROP_MAX and all(abs(v["overshoot"] - six_over) < DROP_OTHER for j, v in Qs["leave_out"].items() if j != "mlp:08")
    pred_e = abs(R["lex_seven"]["pairs"]["mlp:08"]["overshoot"]) <= LEX_TOL
    predictions = {"pred_a_instrument": pred_a, "pred_b_mlp08_pair": pred_b, "pred_c_other_pairs": pred_c, "pred_d_leave_out": pred_d, "pred_e_lex_control": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_quantifier_brake_partner_v156", "candidate_id": "corpus.unit_tier5_quantifier_brake_partner_v156",
              "bars": {"repro_tol": REPRO_TOL, "pair_min": PAIR_MIN, "other_max": OTHER_MAX, "drop_max": DROP_MAX, "drop_other": DROP_OTHER, "lex_tol": LEX_TOL, "v155": [V155_SIX, V155_ALONE]},
              "sets": {k: {"family": v[0], "units": list(v[1])} for k, v in SETS.items() if k != "shared_four"}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
