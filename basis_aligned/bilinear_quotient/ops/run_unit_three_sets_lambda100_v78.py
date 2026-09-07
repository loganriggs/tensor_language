#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, rank (1 per block), lambdas and bars fixed before the run.
"""v78: does a stronger C-inertness penalty (lambda 100) close row 4 for the three sets that miss it, without cost?

v75 (lambda 30) left own-C residuals on held-out C rows: verb_complementizer 0.029 (UB 0.046), dative 0.006 (UB 0.021),
voice_frame 0.007 (UB 0.013); the rubric's row 4 is UB <= 0.01. Same recipe (pooled A1 + verb-variant even rows,
complement term, C even rows as the constraint; rank 1 per block) at lambda 30 (reproduction) and lambda 100 (PRIMARY),
evaluated on ODD rows: A1, A2, own C, unseen pair (where a map exists), cross-collateral, random.

REGISTERED BEFORE THE RUN (CE removal damage in nat on odd rows; primary = lambda 100; the three sets)
    pred_a_reproduce      lambda 30 own-C odd within 0.005 of v75 for all three. Worked: 0.029 vs 0.0285 True.
    pred_b_row4_closed    primary own-C odd ub975 <= 0.01 for at least two of three. Worked: UB 0.008, 0.009, 0.02 True; one only False.
    pred_c_a1_kept        primary A1 >= 0.85 x lambda-30 A1 for all three. Worked: 0.95 vs 1.025 True; 0.80 vs 1.025 False.
    pred_d_row5_kept      primary A2 >= 0.50 x primary A1 AND lb975 > 0 for all three. Worked: 0.6 vs 1.0 True; 0.4 vs 1.0 False.
    pred_e_cross_random   primary cross-collateral max <= 0.05 for all three AND random rank-1 on A1 <= 0.05 x primary A1.
                          Worked: 0.03, -0.01 True; 0.08 False.
    Prior: a True; b uncertain (bootstrap half-width on 16 docs is ~0.015, so UB <= 0.01 needs a point near -0.005);
    c, d True (lambda 30 cost nothing); e True. b False with c True: the residual is a width problem, not a direction
    problem -- report it as such, do not add C rows to the fit.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_tier2_characterization_v23 as v23
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_three_sets_lambda100_v78_result.json"
SRC = [ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json", ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"]
LAMBDAS, PRIMARY = (30.0, 100.0), 100.0
THREE = ("verb_complementizer", "dative", "voice_frame")
PREV = ROOT / "circuits/followups/unit_six_sets_constrained_das_v75_result.json"
C_BAND, KEEP, TRANSFER, CROSS_MAX, RAND_FRAC, C_INERT, EXCEPT = 0.02, 0.70, 0.50, 0.05, 0.05, 0.05, "verb_complementizer"
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 600, 40000


def _plan():
    return {"candidate_id": "corpus.unit_three_sets_lambda100_v78", "lambdas": list(LAMBDAS), "primary": PRIMARY,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 6 * 4 * len(LAMBDAS) * STEPS, "model_updates": 0, "fit_parameters": 6 * len(LAMBDAS) * 13 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    sets = {n: r["final"] for p in SRC for n, r in json.loads(p.read_text())["sets"].items() if n in THREE}
    prev = json.loads(PREV.read_text())["sets"]
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    cross_odd = {n: g.prepare(backend, g.rows_of(m, "A1")[1::2]) for n, m in modules.items()}

    def mu_of(prep, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    R = {}
    for name, units in sets.items():
        m = modules[name]
        a1 = g.rows_of(m, "A1")
        maps, fourth = (v15.SETS[name][2], v15.SETS[name][3]) if name in v15.SETS else ((), None)
        pool_rows = a1[0::2] + [r for mp in maps for r in g.lexical_variant(a1, mp)[0::2]]
        pool = g.prepare(backend, pool_rows)
        even_c = g.prepare(backend, g.rows_of(m, "C")[0::2])
        odd = {"A1": cross_odd[name], "A2": g.prepare(backend, g.rows_of(m, "A2")[1::2]), "C": g.prepare(backend, g.rows_of(m, "C")[1::2])}
        if fourth is not None:
            odd["unseen"] = g.prepare(backend, g.lexical_variant(a1, fourth)[1::2])
        mu = mu_of(pool, units)
        q, hist, dmg = {}, {}, {}
        for lam in LAMBDAS:
            q[lam], hist[lam] = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW,
                                                                 controls=(even_c,), control_weight=lam, mu=mu)
            dmg[str(lam)] = {t: v51.summary(torch, v51.removal(backend, p, units, q[lam], mu)) for t, p in odd.items()}
            dmg[str(lam)]["cross"] = {n: v51.summary(torch, v51.removal(backend, p, units, q[lam], mu))["ce_damage"] for n, p in cross_odd.items() if n != name}
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        dmg["random"] = {"A1": v51.summary(torch, v51.removal(backend, odd["A1"], units, q_rand, mu))}
        R[name] = {"units": units, "pool_rows": len(pool.rows), "maps": list(maps), "fourth": fourth, "damage": dmg,
                   "history": {str(k): v for k, v in hist.items()}, "block_cos_30_vs_primary": g.block_cosines(q[30.0], q[PRIMARY])}
        print(name, {lam: {t: round(v["ce_damage"], 3) for t, v in dmg[str(lam)].items() if t != "cross"} for lam in LAMBDAS},
              "cross_max", round(max(dmg[str(PRIMARY)]["cross"].values()), 3), "rand", round(dmg["random"]["A1"]["ce_damage"], 3),
              round(time.perf_counter() - t0), "s", flush=True)

    P = {n: R[n]["damage"][str(PRIMARY)] for n in R}
    Z = {n: R[n]["damage"]["30.0"] for n in R}
    ce = lambda n, lam, t: R[n]["damage"][str(lam)][t]["ce_damage"]
    predictions = {
        'pred_a_reproduce': all(abs(ce(n, 30.0, "C") - prev[n]["damage"]["30.0"]["C"]["ce_damage"]) <= 0.005 for n in R),
        'pred_b_row4_closed': sum(P[n]["C"]["ce_ub975"] <= 0.01 for n in R) >= 2,
        'pred_c_a1_kept': all(ce(n, PRIMARY, "A1") >= 0.85 * ce(n, 30.0, "A1") for n in R),
        'pred_d_row5_kept': all(ce(n, PRIMARY, "A2") >= TRANSFER * ce(n, PRIMARY, "A1") and P[n]["A2"]["ce_lb975"] > 0 for n in R),
        'pred_e_cross_random': all(max(P[n]["cross"].values()) <= CROSS_MAX for n in R) and all(R[n]["damage"]["random"]["A1"]["ce_damage"] <= RAND_FRAC * ce(n, PRIMARY, "A1") for n in R),
    }
    per_set = {n: {"A1_30": round(ce(n, 30.0, "A1"), 3), "A1_p": round(ce(n, PRIMARY, "A1"), 3), "A2_p": round(ce(n, PRIMARY, "A2"), 3), "A2_lb": round(P[n]["A2"]["ce_lb975"], 3),
                   "C_30": round(ce(n, 30.0, "C"), 3), "C_p": round(ce(n, PRIMARY, "C"), 3), "C_p_ub": round(P[n]["C"]["ce_ub975"], 3),
                   "unseen_p": round(ce(n, PRIMARY, "unseen"), 3) if "unseen" in P[n] else None, "cross_max_p": round(max(P[n]["cross"].values()), 3)} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_three_sets_lambda100_result_v1", "candidate_id": "corpus.unit_three_sets_lambda100_v78",
              "bars": {"c_band": C_BAND, "keep": KEEP, "transfer": TRANSFER, "cross_max": CROSS_MAX, "rand_frac": RAND_FRAC, "c_inert": C_INERT, "except": EXCEPT,
                       "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW, "lambdas": list(LAMBDAS), "primary": PRIMARY}},
              "summary": per_set, "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": per_set, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
