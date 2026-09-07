#!/usr/bin/env python3
# BQGATE: five frozen predictions; recipe, clamp groups and bars fixed before the run; the fits are the v99/v80 recipe.
"""v107: do the fitted rank-1 directions ROUTE like the exact sets they summarise?

v105/v106: each exact head set acts partly directly and partly through the shared mlp7-10 band, with a measured band
survival (exact effect with the band clamped to base / exact effect) of voice 0.32 ... modal 0.83 and a direct share
(all non-set units clamped) of 0.09 ... 1.07. A rank-1 per-block direction that matches the set's MARGIN could still
reach it by a different route (lean on the band more or less than the set does); a direction that routes differently
from the set is a steering direction in disguise. Instrument: `g.forward_units(q=per-block dict, units=set+clamp)` --
blocks absent from q (the clamped MLPs / non-set heads) are patched at full rank to their BASE values (new code path;
control = the exact set through the same call reproduces v105's survivals). Directions: v99/v80 full-specificity recipe
(rank 1 per block, pooled EVEN A1, own C EVEN + six A1 EVEN controls at 30 each, complement 1.0, 120 steps, lr 0.05,
seed 0, mu = pooled EVEN mean); evaluated on ODD A1. survival(group) = clamped effect / unclamped effect, per instrument.

REGISTERED BEFORE THE RUN
    pred_a_instrument      exact-set band and all survivals through the new path equal v105's within 0.02 on 7/7.
    pred_b_band_same       |band survival(direction) - band survival(exact)| <= 0.10 on >= 5 of 7. Worked: 0.55 vs 0.50 True; 0.30 vs 0.50 False.
    pred_c_direct_same     |all survival(direction) - all survival(exact)| <= 0.10 on >= 5 of 7.
    pred_d_attention_inert non-set attention clamped: direction survival >= 0.90 on >= 5 of 7 (as for exact, 0.78-1.00).
    pred_e_direction_leans direction band survival < exact band survival on >= 5 of 7 (the 1-d readout leans on the increment band).
    Prior: a 80%; b 55%; c 55%; d 75%; e 40%.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_candidate_modal_remoteness as m_modal
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_tier2_characterization_v23 as v23

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_seven_direction_routing_v107_result.json"
V80 = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
V97 = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
V105 = ROOT / "circuits/followups/unit_seven_band_readers_v105_result.json"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
INSTR_TOL, SAME_TOL, ATTN_MIN, K = 0.02, 0.10, 0.90, 5
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2000, 120000


def _plan():
    return {"candidate_id": "corpus.unit_seven_direction_routing_v107", "lambda": LAM,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 7 * 2 * STEPS, "model_updates": 0, "fit_parameters": 7 * 13 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")},
               "modal_remoteness": m_modal}
    sets = {n: s["units"] for n, s in json.loads(V80.read_text())["sets"].items()}
    sets["modal_remoteness"] = json.loads(V97.read_text())["final"]
    v105 = json.loads(V105.read_text())["summary"]
    even = {n: g.prepare(backend, g.rows_of(m, "A1")[0::2]) for n, m in modules.items()}
    even_c = {n: g.prepare(backend, g.rows_of(m, "C")[0::2]) for n, m in modules.items()}
    heads_all = [f"attn:{L:02d}:head:{h:02d}" for L in range(g.N_LAYERS) for h in range(g.N_HEADS)]
    mlps = lambda lo, hi: [f"mlp:{L:02d}" for L in range(lo, hi + 1)]

    def effect(O, units, clamp, q):
        merged = dict(O.base_cache)
        for rid in O.base_batch.row_ids:
            for u in units:
                merged[(rid, u)] = O.donor_cache[(rid, u)]
        out = g.forward_units(backend, O.base_batch, units=list(units) + list(clamp), donor_cache=merged, base_cache=O.base_cache, q=q)
        return sum(-(float(a) - float(f)) - b for (a, f), b in zip(out.tolist(), O.base_axis))

    report = {}
    for n, units in sets.items():
        pool = even[n]
        mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (pool.base_cache, pool.donor_cache)
                              for rid in pool.base_batch.row_ids]).mean(0) for u in units}
        controls = (even_c[n],) + tuple(p for k, p in even.items() if k != n)
        q, _ = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0,
                                                complement_weight=CW, controls=controls, control_weight=LAM * len(controls), mu=mu)
        O = g.prepare(backend, g.rows_of(modules[n], "A1")[1::2])
        others = [u for u in heads_all if u not in units]
        groups = {"none": [], "mlp_band": mlps(7, 10), "attn_other": others, "all": mlps(0, 17) + others}
        ex = {k: effect(O, units, c, None) for k, c in groups.items()}
        di = {k: effect(O, units, c, q) for k, c in groups.items()}
        ref = sum(e - b for e, b in zip(g.patched_axis(backend, O, units, q=q), O.base_axis))
        assert abs(di["none"] - ref) <= 1e-3 * max(abs(ref), 1.0), (di["none"], ref)
        S_ex = {k: round(ex[k] / ex["none"], 3) for k in groups if k != "none"}
        S_di = {k: round(di[k] / di["none"], 3) for k in groups if k != "none"}
        report[n] = {"units": units, "exact_effect": round(ex["none"], 3), "direction_effect": round(di["none"], 3),
                     "extraction": round(di["none"] / ex["none"], 3), "exact": S_ex, "direction": S_di,
                     "v105": {k: v105[n][k] for k in ("mlp_band", "attn_other", "all")}}
        print(n, "ext", report[n]["extraction"], "exact", S_ex, "dir", S_di, flush=True)

    R = report
    predictions = {
        'pred_a_instrument': all(abs(R[n]["exact"][k] - R[n]["v105"][k]) <= INSTR_TOL for n in R for k in ("mlp_band", "all")),
        'pred_b_band_same': sum(abs(R[n]["direction"]["mlp_band"] - R[n]["exact"]["mlp_band"]) <= SAME_TOL for n in R) >= K,
        'pred_c_direct_same': sum(abs(R[n]["direction"]["all"] - R[n]["exact"]["all"]) <= SAME_TOL for n in R) >= K,
        'pred_d_attention_inert': sum(R[n]["direction"]["attn_other"] >= ATTN_MIN for n in R) >= K,
        'pred_e_direction_leans': sum(R[n]["direction"]["mlp_band"] < R[n]["exact"]["mlp_band"] for n in R) >= K,
    }
    summary = {n: {"extraction": R[n]["extraction"], "exact": R[n]["exact"], "direction": R[n]["direction"]} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_direction_routing_result_v1",
              "candidate_id": "corpus.unit_seven_direction_routing_v107", "summary": summary, "sets": report,
              "recipe": {"lambda": LAM, "steps": STEPS, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0},
              "bars": {"instr_tol": INSTR_TOL, "same_tol": SAME_TOL, "attn_min": ATTN_MIN, "k": K},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
