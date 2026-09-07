#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, controls, rank (1 per block), lambda and bars fixed before the run.
"""v80: full-specificity constrained DAS -- own C AND the other five behaviours' A1 rows as inertness controls, all six hub+8 sets.

v79: verb_preposition's cross-collateral was a separable component (two behaviours as controls; A1 0.98x kept). This runs
the same idea as one recipe on all six sets: controls = own C even + the five other behaviours' A1 EVEN rows, control_weight
180 over six controls = 30 per control (own C keeps its v75 weight). Arm `own` = v75's lambda 30 (reproduction); arm
`xctl` = full specificity. Evaluated on ODD rows only: A1, A2, own C, unseen fourth pair, other five A1 (signed AND absolute:
v75's dative direction on quantifier was -0.093, a helpful removal is still non-inert).

REGISTERED BEFORE THE RUN (CE removal damage in nat on odd rows, hub+8)
    pred_a_reproduce      own A1 and own C within 0.005 of v75 for all six. Worked: 0.906 vs 0.906 True.
    pred_b_cross_abs      xctl max |cross| <= 0.05 for all six (v75 fails: preposition 0.077, dative |-0.093|). Worked: 0.03 True; 0.06 False.
    pred_c_a1_kept        xctl A1 >= 0.85 x own A1 for all six. Worked: 0.887 vs 0.906 True; 0.70 vs 0.906 False.
    pred_d_row4_kept      xctl own-C odd ub975 <= own's ub975 + 0.01 for all six (no set loses row 4 status). Worked: 0.005 vs 0.003 True; 0.03 vs 0.003 False.
    pred_e_row5_kept      xctl A2 >= 0.50 x xctl A1 with lb975 > 0 for all six AND unseen >= 0.80 x own's unseen for the five with maps
                          AND random rank-1 on A1 <= 0.05 x xctl A1. Worked: 0.64 vs 0.89, 0.27 vs 0.30 True; 0.20 vs 0.30 False.
    Prior: a True; b, c True together is the hypothesis (collateral is separable everywhere, as for preposition); b True
    with c False on a set names that set's cross-collateral as part of its on-target direction (report which); d, e True.
    Complementizer's own-C residual (0.029 at lambda 30) is recorded, not predicted.
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
OUT = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
SRC = [ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json", ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"]
ARMS = ("own", "xctl")
LAM_PER_CONTROL = 30.0
REPRO, CROSS_MAX, KEEP, C_UB_SLACK, TRANSFER, UNSEEN_KEEP, RAND_FRAC = 0.005, 0.05, 0.85, 0.01, 0.50, 0.80, 0.05
PREV = ROOT / "circuits/followups/unit_six_sets_constrained_das_v75_result.json"
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 600, 40000


def _plan():
    return {"candidate_id": "corpus.unit_six_sets_cross_inert_v80", "arms": list(ARMS), "lambda_per_control": LAM_PER_CONTROL,
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
    cross_odd = {n: g.prepare(backend, g.rows_of(m, "A1")[1::2]) for n, m in modules.items()}
    cross_even = {n: g.prepare(backend, g.rows_of(m, "A1")[0::2]) for n, m in modules.items()}
    prev = json.loads(PREV.read_text())["sets"]

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
        controls = {"own": (even_c,), "xctl": (even_c,) + tuple(cross_even[n] for n in modules if n != name)}
        for arm in ARMS:
            q[arm], hist[arm] = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW,
                                                                 controls=controls[arm], control_weight=LAM_PER_CONTROL * len(controls[arm]), mu=mu)
            dmg[arm] = {t: v51.summary(torch, v51.removal(backend, p, units, q[arm], mu)) for t, p in odd.items()}
            dmg[arm]["cross"] = {n: v51.summary(torch, v51.removal(backend, p, units, q[arm], mu))["ce_damage"] for n, p in cross_odd.items() if n != name}
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        dmg["random"] = {"A1": v51.summary(torch, v51.removal(backend, odd["A1"], units, q_rand, mu))}
        R[name] = {"units": units, "pool_rows": len(pool.rows), "maps": list(maps), "fourth": fourth, "damage": dmg,
                   "history": hist, "block_cos_own_vs_xctl": g.block_cosines(q["own"], q["xctl"])}
        print(name, {arm: {t: round(v["ce_damage"], 3) for t, v in dmg[arm].items() if t != "cross"} for arm in ARMS},
              "cross_abs_max", round(max(abs(v) for v in dmg["xctl"]["cross"].values()), 3), "rand", round(dmg["random"]["A1"]["ce_damage"], 3),
              round(time.perf_counter() - t0), "s", flush=True)

    P = {n: R[n]["damage"]["xctl"] for n in R}
    O = {n: R[n]["damage"]["own"] for n in R}
    ce = lambda n, arm, t: R[n]["damage"][arm][t]["ce_damage"]
    predictions = {
        'pred_a_reproduce': all(abs(ce(n, "own", t) - prev[n]["damage"]["30.0"][t]["ce_damage"]) <= REPRO for n in R for t in ("A1", "C")),
        'pred_b_cross_abs': all(max(abs(v) for v in P[n]["cross"].values()) <= CROSS_MAX for n in R),
        'pred_c_a1_kept': all(ce(n, "xctl", "A1") >= KEEP * ce(n, "own", "A1") for n in R),
        'pred_d_row4_kept': all(P[n]["C"]["ce_ub975"] <= O[n]["C"]["ce_ub975"] + C_UB_SLACK for n in R),
        'pred_e_row5_kept': all(ce(n, "xctl", "A2") >= TRANSFER * ce(n, "xctl", "A1") and P[n]["A2"]["ce_lb975"] > 0 for n in R)
                            and all(ce(n, "xctl", "unseen") >= UNSEEN_KEEP * ce(n, "own", "unseen") for n in R if "unseen" in P[n])
                            and all(R[n]["damage"]["random"]["A1"]["ce_damage"] <= RAND_FRAC * ce(n, "xctl", "A1") for n in R),
    }
    per_set = {n: {"A1_own": round(ce(n, "own", "A1"), 3), "A1_x": round(ce(n, "xctl", "A1"), 3), "A2_x": round(ce(n, "xctl", "A2"), 3), "A2_lb": round(P[n]["A2"]["ce_lb975"], 3),
                   "C_own": round(ce(n, "own", "C"), 3), "C_x": round(ce(n, "xctl", "C"), 3), "C_x_ub": round(P[n]["C"]["ce_ub975"], 3), "C_own_ub": round(O[n]["C"]["ce_ub975"], 3),
                   "unseen_own": round(ce(n, "own", "unseen"), 3) if "unseen" in P[n] else None, "unseen_x": round(ce(n, "xctl", "unseen"), 3) if "unseen" in P[n] else None,
                   "cross_abs_max_own": round(max(abs(v) for v in O[n]["cross"].values()), 3), "cross_abs_max_x": round(max(abs(v) for v in P[n]["cross"].values()), 3)} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_six_sets_cross_inert_result_v1", "candidate_id": "corpus.unit_six_sets_cross_inert_v80",
              "bars": {"repro": REPRO, "cross_max_abs": CROSS_MAX, "keep": KEEP, "c_ub_slack": C_UB_SLACK, "transfer": TRANSFER, "unseen_keep": UNSEEN_KEEP, "rand_frac": RAND_FRAC,
                       "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW, "lambda_per_control": LAM_PER_CONTROL, "arms": list(ARMS)}},
              "summary": per_set, "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": per_set, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
