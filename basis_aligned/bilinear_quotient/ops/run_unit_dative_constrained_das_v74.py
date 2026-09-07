#!/usr/bin/env python3
# BQGATE: five frozen predictions; set, pairs, rank (1 per block), lambda and bars fixed before the run.
"""v74: constrained DAS -- can one rank-1 direction per block hold dative's row 5 AND row 4 together on hub+8?

v73: a DAS direction pooled over A1 + two verb variants (A2 never seen) removes A2 at 0.320 nat = 0.70x its A1 damage
(row 5 met) but damages own C by 0.125 (UB 0.144; row 4 lost). The single-pair A1 direction has the opposite profile
(A2 0.218 = 0.43x, C -0.159). The user's proposal: constrained DAS with the complement term and a regularizer. The
regularizer here is a differentiable copy of the v51 removal statistic on the control family -- mean squared answer-CE
change when the units are projected onto the pooled background along the subspace, on C EVEN rows
(g.fit_block_subspace_constrained; mu passed as the removal "donor" exactly as v51 does). Everything is evaluated on
ODD rows the fit never saw: A1, A2, the unseen fourth verb pair, own C, and the other five behaviours' A1 rows.
Fits (rank 1 per block, complement_weight 1.0, 120 steps, seed 0): lambda 0 (reproduction control of v73's das),
lambda 10 (information), lambda 30 (PRIMARY, registered).

REGISTERED BEFORE THE RUN (CE removal damage in nat on odd rows, hub+8, primary = lambda 30)
    pred_a_reproduce      lambda 0 reproduces v73's das on A2 odd and on C odd within 0.02 nat each (0.320, 0.125).
                          Worked: 0.318 / 0.127 True; 0.28 / 0.13 False.
    pred_b_c_repaired     primary own-C odd CE ub975 <= 0.01 nat. Worked: UB -0.010 True; UB 0.050 False.
    pred_c_row5_kept      primary on A2 odd >= 0.50 x primary on A1 odd AND lb975 > 0. Worked: 0.25 vs 0.40 True; 0.15 vs 0.40 False.
    pred_d_a1_kept        primary on A1 odd >= 0.70 x lambda-0 on A1 odd (v73 0.459 -> >= 0.32). Worked: 0.40 True; 0.25 False.
    pred_e_cross_clean    primary on each of the other five behaviours' A1 odd rows: CE damage <= 0.05 nat, AND random rank-1
                          on A2 odd <= 0.05 x A2 own refit (0.629). Worked: max 0.02, 0.01 True; 0.10 False.
    Prior: a True (same code path at lambda 0); b, c, d True together is the hypothesis -- C damage is a separable
    component of the pooled direction. b True with c or d False means the C-damaging component IS the transferable
    component (the price of transfer is specificity); b False means the penalty cannot find an inert direction at this
    rank -- report, do not raise the rank.
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
OUT = ROOT / "circuits/followups/unit_dative_constrained_das_v74_result.json"
SRC = ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"
PREV = ROOT / "circuits/followups/unit_dative_pooled_das_row5_v73_result.json"
NAME = "dative"
LAMBDAS, PRIMARY = (0.0, 10.0, 30.0), 30.0
TRANSFER, KEEP, C_UB, CROSS_MAX, RAND_FRAC, REPRO = 0.50, 0.70, 0.01, 0.05, 0.05, 0.02
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 240, 16000


def _plan():
    return {"candidate_id": "corpus.unit_dative_constrained_das_v74", "lambdas": list(LAMBDAS), "primary": PRIMARY,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 4 * len(LAMBDAS) * STEPS, "model_updates": 0, "fit_parameters": len(LAMBDAS) * 13 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    module, _, maps, fourth = v15.SETS[NAME]
    units = json.loads(SRC.read_text())["sets"][NAME]["final"]
    prev = json.loads(PREV.read_text())["sets"]["hub8"]["damage"]
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    a1 = g.rows_of(module, "A1")
    pairs = {"orig": a1, "v1": g.lexical_variant(a1, maps[0]), "v2": g.lexical_variant(a1, maps[1]), "v3": g.lexical_variant(a1, fourth)}
    odd = {k: g.prepare(backend, r[1::2]) for k, r in pairs.items()}
    a2, c_rows = g.rows_of(module, "A2"), g.rows_of(module, "C")
    odd["A2"], odd["C"] = g.prepare(backend, a2[1::2]), g.prepare(backend, c_rows[1::2])
    even_a2, even_c = g.prepare(backend, a2[0::2]), g.prepare(backend, c_rows[0::2])
    cross = {n: g.prepare(backend, g.rows_of(m, "A1")[1::2]) for n, m in modules.items() if n != NAME}
    pool = g.prepare(backend, [r for k in ("orig", "v1", "v2") for r in pairs[k][0::2]])

    def mu_of(prep):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    mu_pool = mu_of(pool)
    q, hist = {}, {}
    for lam in LAMBDAS:
        q[lam], hist[lam] = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW,
                                                             controls=(even_c,), control_weight=lam, mu=mu_pool)
        print("lambda", lam, "history", hist[lam], flush=True)
    q["random"] = g.block_random_subspace(backend, units, rank=1, seed=1)
    q["A2_refit"] = g.block_diff_in_means(backend, even_a2, units)
    mu_a2 = mu_of(even_a2)
    targets = ("orig", "v1", "v2", "v3", "A2", "C")
    dmg = {}
    for lam in LAMBDAS:
        dmg[str(lam)] = {t: v51.summary(torch, v51.removal(backend, odd[t], units, q[lam], mu_pool)) for t in targets}
        dmg[str(lam)]["cross"] = {n: v51.summary(torch, v51.removal(backend, p, units, q[lam], mu_pool)) for n, p in cross.items()}
        print("lambda", lam, json.dumps({t: round(dmg[str(lam)][t]["ce_damage"], 3) for t in targets}),
              "cross", json.dumps({n: round(v["ce_damage"], 3) for n, v in dmg[str(lam)]["cross"].items()}), flush=True)
    dmg["random"] = {"A2": v51.summary(torch, v51.removal(backend, odd["A2"], units, q["random"], mu_pool))}
    dmg["A2_refit"] = {"A2": v51.summary(torch, v51.removal(backend, odd["A2"], units, q["A2_refit"], mu_a2))}
    cos = {f"{a}|{b}": g.block_cosines(q[a], q[b]) for a, b in ((0.0, PRIMARY), (0.0, 10.0), (PRIMARY, "A2_refit"), (0.0, "A2_refit"))}

    P, Z = dmg[str(PRIMARY)], dmg["0.0"]
    predictions = {
        'pred_a_reproduce': abs(Z["A2"]["ce_damage"] - prev["das->A2"]["ce_damage"]) <= REPRO and abs(Z["C"]["ce_damage"] - prev["das->C"]["ce_damage"]) <= REPRO,
        'pred_b_c_repaired': P["C"]["ce_ub975"] <= C_UB,
        'pred_c_row5_kept': P["A2"]["ce_damage"] >= TRANSFER * P["orig"]["ce_damage"] and P["A2"]["ce_lb975"] > 0,
        'pred_d_a1_kept': P["orig"]["ce_damage"] >= KEEP * Z["orig"]["ce_damage"],
        'pred_e_cross_clean': all(v["ce_damage"] <= CROSS_MAX for v in P["cross"].values()) and dmg["random"]["A2"]["ce_damage"] <= RAND_FRAC * dmg["A2_refit"]["A2"]["ce_damage"],
    }
    result = {"predictions": predictions, "schema": "circuit_unit_dative_constrained_das_result_v1", "candidate_id": "corpus.unit_dative_constrained_das_v74",
              "units": units, "pairs": {"v1": maps[0], "v2": maps[1], "v3": fourth}, "pool_rows": len(pool.rows), "control_rows_even": len(even_c.rows),
              "rows_odd": {k: len(p.rows) for k, p in odd.items()},
              "bars": {"transfer": TRANSFER, "keep": KEEP, "c_ub": C_UB, "cross_max": CROSS_MAX, "rand_frac": RAND_FRAC, "repro": REPRO,
                       "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW, "lambdas": list(LAMBDAS), "primary": PRIMARY}},
              "damage": dmg, "history": {str(k): v for k, v in hist.items()}, "block_cosines": cos, "v73_reference": {k: prev[k]["ce_damage"] for k in ("das->A2", "das->C", "das->orig")},
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
