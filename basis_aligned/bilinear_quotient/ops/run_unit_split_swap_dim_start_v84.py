#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, recipe, split and bars fixed before the run.
"""v84: split swap -- fit on the ODD half, evaluate on the EVEN half. Does the dim-start direction's row-4 closure replicate?

v82 (registered for extraction) produced an unregistered observation: the full-specificity direction started from the
diff-in-means (xdim) has own-C ODD removal UB 0.000 on dative and quantifier, where the random-start direction (xdas) has
UB 0.026 / 0.018, and dative's A2 removal is 0.422 (0.83x A1) instead of 0.323. Those numbers were read before this test
was written, so they cannot be claimed from that run. Here everything is mirrored: pooled fit on ODD A1 + verb-variant
rows, inertness controls = ODD own C + ODD other-five A1 (30 per control), dim start from ODD A1; evaluation on EVEN rows
only (A1, A2, own C, unseen fourth pair, other five A1). Arms: xdas (random start), xdim (dim start), random rank-1.

REGISTERED BEFORE THE RUN (CE removal damage in nat on EVEN rows)
    pred_a_recipe_replicates  xdas A1 even >= 0.80 x v80's xdas A1 ev for all six. Worked: 0.60 vs 0.731 True; 0.50 False.
    pred_b_row4_dat_quant     xdim own-C even ub975 <= 0.01 on BOTH dative and quantifier. Worked: 0.004, 0.008 True; 0.004, 0.015 False.
    pred_c_dim_start_inerter  xdim own-C even ub975 <= xdas own-C even ub975 on at least four of six. Worked: 4 True; 3 False.
    pred_d_dative_a2          xdim dative A2 even >= 0.70 x xdim dative A1 even with lb975 > 0. Worked: 0.40 vs 0.50 True; 0.30 vs 0.50 False.
    pred_e_rows_kept          xdim A1 even >= 0.85 x xdas A1 even AND max |cross| <= 0.05 AND random on A1 <= 0.05 x xdim A1, all six.
    Prior: a True; b is the confirmation and I put it near 50/50 -- a UB of 0.000 on 16 documents is a small-sample
    artefact as often as not; c likely (the dim start begins inert and the penalty keeps it there); d unsure; e True.
    b False: the v82 observation is recorded as not replicated and the tier list keeps rows 4 open for both sets.
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
OUT = ROOT / "circuits/followups/unit_split_swap_dim_start_v84_result.json"
SRC = [ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json", ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"]
ARMS = ("xdas", "xdim")
LAM_PER_CONTROL = 30.0
REPLICATE, C_UB, N_INERTER, DAT_A2, KEEP, CROSS_MAX, RAND_FRAC = 0.80, 0.01, 4, 0.70, 0.85, 0.05, 0.05
PREV = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 600, 40000


def _plan():
    return {"candidate_id": "corpus.unit_split_swap_dim_start_v84", "arms": list(ARMS), "lambda_per_control": LAM_PER_CONTROL,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 6 * 4 * 2 * STEPS, "model_updates": 0, "fit_parameters": 6 * 2 * 13 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    sets = {n: r["final"] for p in SRC for n, r in json.loads(p.read_text())["sets"].items()}
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    eval_a1 = {n: g.prepare(backend, g.rows_of(m, "A1")[0::2]) for n, m in modules.items()}
    fit_a1 = {n: g.prepare(backend, g.rows_of(m, "A1")[1::2]) for n, m in modules.items()}
    prev = json.loads(PREV.read_text())["sets"]

    def mu_of(prep, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    R = {}
    for name, units in sets.items():
        m = modules[name]
        a1 = g.rows_of(m, "A1")
        maps, fourth = (v15.SETS[name][2], v15.SETS[name][3]) if name in v15.SETS else ((), None)
        pool_rows = a1[1::2] + [r for mp in maps for r in g.lexical_variant(a1, mp)[1::2]]
        pool = g.prepare(backend, pool_rows)
        fit_c = g.prepare(backend, g.rows_of(m, "C")[1::2])
        ev = {"A1": eval_a1[name], "A2": g.prepare(backend, g.rows_of(m, "A2")[0::2]), "C": g.prepare(backend, g.rows_of(m, "C")[0::2])}
        if fourth is not None:
            ev["unseen"] = g.prepare(backend, g.lexical_variant(a1, fourth)[0::2])
        mu = mu_of(pool, units)
        q, hist, dmg = {}, {}, {}
        xc = (fit_c,) + tuple(fit_a1[n] for n in modules if n != name)
        inits = {"xdas": None, "xdim": g.block_diff_in_means(backend, fit_a1[name], units)}
        for arm in ARMS:
            q[arm], hist[arm] = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW,
                                                                 controls=xc, control_weight=LAM_PER_CONTROL * len(xc), mu=mu, init=inits[arm])
            dmg[arm] = {t: v51.summary(torch, v51.removal(backend, p, units, q[arm], mu)) for t, p in ev.items()}
            dmg[arm]["cross"] = {n: v51.summary(torch, v51.removal(backend, p, units, q[arm], mu))["ce_damage"] for n, p in eval_a1.items() if n != name}
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        dmg["random"] = {"A1": v51.summary(torch, v51.removal(backend, ev["A1"], units, q_rand, mu))}
        R[name] = {"units": units, "pool_rows": len(pool.rows), "maps": list(maps), "fourth": fourth, "damage": dmg,
                   "history": hist, "block_cos_xdas_vs_xdim": g.block_cosines(q["xdas"], q["xdim"])}
        print(name, {arm: {t: round(v["ce_damage"], 3) for t, v in dmg[arm].items() if t != "cross"} for arm in ARMS},
              "cross_abs_max", round(max(abs(v) for v in dmg["xdim"]["cross"].values()), 3), "rand", round(dmg["random"]["A1"]["ce_damage"], 3),
              round(time.perf_counter() - t0), "s", flush=True)

    P = {n: R[n]["damage"]["xdim"] for n in R}
    O = {n: R[n]["damage"]["xdas"] for n in R}
    ce = lambda n, arm, t: R[n]["damage"][arm][t]["ce_damage"]
    v80 = lambda n: prev[n]["damage"]["xctl"]["A1"]["ce_damage"]
    inerter = [n for n in R if P[n]["C"]["ce_ub975"] <= O[n]["C"]["ce_ub975"]]
    predictions = {
        'pred_a_recipe_replicates': all(ce(n, "xdas", "A1") >= REPLICATE * v80(n) for n in R),
        'pred_b_row4_dat_quant': all(P[n]["C"]["ce_ub975"] <= C_UB for n in ("dative", "quantifier_number")),
        'pred_c_dim_start_inerter': len(inerter) >= N_INERTER,
        'pred_d_dative_a2': ce("dative", "xdim", "A2") >= DAT_A2 * ce("dative", "xdim", "A1") and P["dative"]["A2"]["ce_lb975"] > 0,
        'pred_e_rows_kept': all(ce(n, "xdim", "A1") >= KEEP * ce(n, "xdas", "A1") and max(abs(v) for v in P[n]["cross"].values()) <= CROSS_MAX
                               and R[n]["damage"]["random"]["A1"]["ce_damage"] <= RAND_FRAC * ce(n, "xdim", "A1") for n in R),
    }
    per_set = {n: {"A1_xdas": round(ce(n, "xdas", "A1"), 3), "A1_xdim": round(ce(n, "xdim", "A1"), 3), "A1_v80_odd": round(v80(n), 3),
                   "A2_xdim": round(ce(n, "xdim", "A2"), 3), "A2_lb": round(P[n]["A2"]["ce_lb975"], 3),
                   "C_xdas": round(ce(n, "xdas", "C"), 3), "C_xdas_ub": round(O[n]["C"]["ce_ub975"], 3), "C_xdim": round(ce(n, "xdim", "C"), 3), "C_xdim_ub": round(P[n]["C"]["ce_ub975"], 3),
                   "unseen_xdim": round(ce(n, "xdim", "unseen"), 3) if "unseen" in P[n] else None,
                   "cross_abs_max_xdim": round(max(abs(v) for v in P[n]["cross"].values()), 3)} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_split_swap_dim_start_result_v1", "candidate_id": "corpus.unit_split_swap_dim_start_v84",
              "bars": {"replicate": REPLICATE, "c_ub": C_UB, "n_inerter": N_INERTER, "dat_a2": DAT_A2, "keep": KEEP, "cross_max": CROSS_MAX, "rand_frac": RAND_FRAC,
                       "split": "fit ODD, evaluate EVEN", "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW, "lambda_per_control": LAM_PER_CONTROL, "arms": list(ARMS)}},
              "summary": per_set, "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": per_set, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
