#!/usr/bin/env python3
# BQGATE: frozen predictions; sets (v9), low alpha grid, stacks (v27/v28) and the fit fixed before the run.
"""v29: exponent of the converter law -- is the MLP-stack conversion quadratic in the set's write?

v27/v28: conversion(0.5)/conversion(1) = 0.37 (quantifier), 0.25 (polarity), 0.20 (voice). A Bilinear MLP
on x = base + alpha w gives Left(x)*Right(x) = L(b)R(b) + alpha [L(b)R(w) + L(w)R(b)] + alpha^2 L(w)R(w):
the cross-terms are linear in alpha, the self-term quadratic. Ratio 0.25 at alpha 0.5 is the self-term's
signature. Low grid alpha in {0.25, 0.5, 0.75, 1.0}, stack conversion = rec_live - rec_stack_frozen
(stacks: quantifier mlp 11-14, polarity mlp 08-11 + attn 09:07, voice mlp 07-11), log-log least-squares
slope over the four points, and R^2 of through-origin fits c alpha vs c alpha^2.

REGISTERED BEFORE THE RUN
    pred_a_instrument         rec_live(1.0) equals the recorded exact-set recovery (quantifier 0.635, polarity
                              0.585, voice 0.645) within 0.005.
    pred_b_early_quadratic    slope in [1.6, 2.4] for polarity AND voice. Worked: 1.95, 2.1 -> True; 1.3 -> False.
    pred_c_quantifier_quadratic  slope in [1.6, 2.4] for quantifier (v27's 0.37 suggests ~1.4; registered as the
                              same hypothesis so a miss is a recorded difference). Worked: 1.45 -> False.
    pred_d_quadratic_fits_better  R^2(c alpha^2) > R^2(c alpha) on all three. Worked: 0.99 vs 0.90 -> True.
    pred_e_stack_holds_low    stack conversion / total conversion >= 0.80 at every alpha for polarity and voice
                              (the stack definition does not change with the write's size). Worked: 0.83..0.94 -> True.
    Reading rule. b True: the early sets' conversion is the write's quadratic self-term inside mlp 7/8-11 -- Tier 4
    becomes an exact weight statement (mlp_subspace_tensor restricted to the write axis, executable sufficiency
    next). b False with slope near 1: the write is read against context (cross-terms); Tier 4 needs the context
    factor identified first.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier2_characterization_v23 as v23
import run_unit_tier3_readers_v25 as v25
import run_unit_damper_law_v27 as v27
import run_unit_converter_law_v28 as v28

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_converter_exponent_v29_result.json"
SETS = {k: v23.SETS[k] for k in ("quantifier_number", "polarity_licensing", "voice_frame")}
STACK = dict(v28.STACK); STACK["quantifier_number"] = [f"mlp:{l:02d}" for l in range(11, 15)]
ALPHAS = (0.25, 0.5, 0.75, 1.0)
EXPECTED = {"quantifier_number": 0.635, "polarity_licensing": 0.585, "voice_frame": 0.645}
INSTR_TOL, SLOPE_LO, SLOPE_HI, STACK_BAR = 0.005, 1.6, 2.4, 0.80
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 60, 2000


def _plan():
    return {"candidate_id": "corpus.unit_converter_exponent_v29", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "alphas": list(ALPHAS), "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _slope(xs, ys):
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def _r2_origin(xs, ys, power):
    f = [x ** power for x in xs]
    c = sum(a * b for a, b in zip(f, ys)) / sum(a * a for a in f)
    ss_res = sum((y - c * a) ** 2 for a, y in zip(f, ys))
    my = sum(ys) / len(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    return 1 - ss_res / ss_tot if ss_tot else None, c


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        downstream, stack = v25._downstream(units), STACK[name]
        assert all(m in downstream for m in stack), (name, stack)
        cache = v25._merged_cache(prep, units)

        def rec(q_set, frozen):
            q = dict(q_set)
            for key, qb in v28._eye_blocks(backend, frozen).items():
                assert key not in q, key
                q[key] = qb
            out = g.forward_units(backend, prep.base_batch, units=list(units) + list(frozen), donor_cache=cache,
                                  base_cache=prep.base_cache, q=q)
            return v25._rec(prep, [-(float(x) - float(f)) for x, f in out.tolist()])

        curve = {}
        for a in ALPHAS:
            q_set = v27._scaled_q(backend, units, a)
            live, direct, sf = rec(q_set, []), rec(q_set, downstream), rec(q_set, stack)
            curve[a] = {"rec_live": live, "rec_direct": direct, "rec_stack_frozen": sf,
                        "total_conversion": live - direct, "stack_conversion": live - sf,
                        "stack_share": (live - sf) / (live - direct) if live != direct else None}
            print(name, a, {k: round(v, 3) for k, v in curve[a].items() if v is not None}, flush=True)
        ys = [curve[a]["stack_conversion"] for a in ALPHAS]
        ok = all(y > 0 for y in ys)
        r2_lin, c_lin = _r2_origin(list(ALPHAS), ys, 1)
        r2_quad, c_quad = _r2_origin(list(ALPHAS), ys, 2)
        report[name] = {"units": list(units), "stack": stack, "rows": len(prep.rows), "curve": curve,
                        "stack_conversion": ys, "slope_loglog": _slope(list(ALPHAS), ys) if ok else None,
                        "r2_linear": r2_lin, "r2_quadratic": r2_quad, "coef_linear": c_lin, "coef_quadratic": c_quad,
                        "total_conversion": [curve[a]["total_conversion"] for a in ALPHAS]}
        print(name, "slope %.2f r2 lin %.3f quad %.3f" % (report[name]["slope_loglog"] or -9, r2_lin, r2_quad), flush=True)

    predictions = {
        'pred_a_instrument': all(abs(report[n]["curve"][1.0]["rec_live"] - EXPECTED[n]) <= INSTR_TOL for n in SETS),
        'pred_b_early_quadratic': all(report[n]["slope_loglog"] is not None and SLOPE_LO <= report[n]["slope_loglog"] <= SLOPE_HI
                                      for n in ("polarity_licensing", "voice_frame")),
        'pred_c_quantifier_quadratic': report["quantifier_number"]["slope_loglog"] is not None
                                       and SLOPE_LO <= report["quantifier_number"]["slope_loglog"] <= SLOPE_HI,
        'pred_d_quadratic_fits_better': all(report[n]["r2_quadratic"] is not None and report[n]["r2_linear"] is not None
                                            and report[n]["r2_quadratic"] > report[n]["r2_linear"] for n in SETS),
        'pred_e_stack_holds_low': all(report[n]["curve"][a]["stack_share"] is not None and report[n]["curve"][a]["stack_share"] >= STACK_BAR
                                      for n in ("polarity_licensing", "voice_frame") for a in ALPHAS),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_converter_exponent_result_v1",
              "candidate_id": "corpus.unit_converter_exponent_v29", "semantics": "block_live_scaled_write_plus_base_freeze",
              "alphas": list(ALPHAS), "bars": {"instrument": INSTR_TOL, "slope": [SLOPE_LO, SLOPE_HI], "stack": STACK_BAR},
              "slopes": {n: report[n]["slope_loglog"] for n in SETS}, "behaviours": report,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "slopes": result["slopes"],
                      "r2": {n: (report[n]["r2_linear"], report[n]["r2_quadratic"]) for n in SETS}, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
