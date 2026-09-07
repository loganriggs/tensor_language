#!/usr/bin/env python3
# BQGATE: five frozen predictions; set, controls, rank (1 per block), lambdas and bars fixed before the run.
"""v79: verb_preposition's cross-collateral -- is it a separable component of the rank-1 direction? (one licensed repair of v75 pred_d)

v75 (pooled constrained DAS, lambda 30 on own C): verb_preposition meets rows 2/3/4/5 on hub+8 but its direction damages
dative A1 odd by 0.077 nat and verb_complementizer A1 odd by 0.061 (bar 0.05) -- the only set failing pred_d. The repair
adds those two behaviours' A1 EVEN rows as further removal-inertness controls (control_weight 90 over three controls =
30 per control, so own C keeps its v75 weight). Arms: `own` (controls = C even; lambda 30; reproduction of v75) and
`xctl` (controls = C even + dative A1 even + complementizer A1 even). Everything is evaluated on ODD rows the fit never
saw: A1, the two verb variants, the unseen fourth pair, A2, own C, the two control behaviours, and the three behaviours
never used as controls (quantifier, polarity, voice).

REGISTERED BEFORE THE RUN (CE removal damage in nat on odd rows, hub+8)
    pred_a_reproduce     own on A1 odd and C odd within 0.005 of v75 (0.906, -0.001). Worked: 0.904 / 0.001 True; 0.89 False.
    pred_b_cross_repaired xctl on dative A1 odd <= 0.05 AND on complementizer A1 odd <= 0.05. Worked: 0.02, 0.03 True; 0.06, 0.03 False.
    pred_c_a1_kept       xctl on A1 odd >= 0.85 x own on A1 odd. Worked: 0.80 vs 0.906 True; 0.70 False.
    pred_d_rows_kept     xctl own-C odd ub975 <= 0.01 AND A2 odd >= 0.50 x A1 odd with lb975 > 0. Worked: UB 0.004, A2 0.60 vs 0.90 True.
    pred_e_held_out      xctl on each of quantifier / polarity / voice A1 odd <= 0.05 AND xctl on the unseen fourth pair >= 0.50 x own's
                         AND random rank-1 on A1 odd <= 0.05 x xctl A1. Worked: 0.02/0.01/0.03, 0.40 vs 0.60, 0.01 True; 0.07 False.
    Prior: a True; b True with c True is the hypothesis (collateral is separable). b True with c False: the collateral
    component IS part of the on-target direction (the price of specificity). b False: the penalty cannot remove it at
    this rank -- report, do not raise the rank and do not add a second repair.
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
OUT = ROOT / "circuits/followups/unit_preposition_cross_inert_v79_result.json"
SRC = ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json"
PREV = ROOT / "circuits/followups/unit_six_sets_constrained_das_v75_result.json"
NAME = "verb_preposition"
CONTROL_NAMES = ("dative", "verb_complementizer")
HELD_OUT = ("quantifier_number", "polarity_licensing", "voice_frame")
LAM_OWN, LAM_X = 30.0, 90.0    # 90 over three controls = 30 per control
STEPS, LR, CW = 120, 0.05, 1.0
REPRO, CROSS_MAX, KEEP, C_UB, TRANSFER, RAND_FRAC = 0.005, 0.05, 0.85, 0.01, 0.50, 0.05


def _plan():
    return {"candidate_id": "corpus.unit_preposition_cross_inert_v79", "arms": ["own", "xctl"], "lambdas": [LAM_OWN, LAM_X],
            "model_forwards_max": 6000, "example_evaluations_max": 40000,
            "model_backwards": 8 * STEPS, "model_updates": 0, "fit_parameters": 2 * 11 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    module, _, maps, fourth = v15.SETS[NAME]
    units = json.loads(SRC.read_text())["sets"][NAME]["final"]
    prev = json.loads(PREV.read_text())["sets"][NAME]["damage"]["30.0"]
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    a1 = g.rows_of(module, "A1")
    pairs = {"orig": a1, "v1": g.lexical_variant(a1, maps[0]), "v2": g.lexical_variant(a1, maps[1]), "v3": g.lexical_variant(a1, fourth)}
    odd = {k: g.prepare(backend, r[1::2]) for k, r in pairs.items()}
    a2, c_rows = g.rows_of(module, "A2"), g.rows_of(module, "C")
    odd["A2"], odd["C"] = g.prepare(backend, a2[1::2]), g.prepare(backend, c_rows[1::2])
    even_c = g.prepare(backend, c_rows[0::2])
    cross = {n: g.prepare(backend, g.rows_of(m, "A1")[1::2]) for n, m in modules.items() if n != NAME}
    even_ctl = {n: g.prepare(backend, g.rows_of(modules[n], "A1")[0::2]) for n in CONTROL_NAMES}
    pool = g.prepare(backend, [r for k in ("orig", "v1", "v2") for r in pairs[k][0::2]])

    def mu_of(prep):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    mu_pool = mu_of(pool)
    q, hist = {}, {}
    q["own"], hist["own"] = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW,
                                                             controls=(even_c,), control_weight=LAM_OWN, mu=mu_pool)
    q["xctl"], hist["xctl"] = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW,
                                                               controls=(even_c,) + tuple(even_ctl[n] for n in CONTROL_NAMES), control_weight=LAM_X, mu=mu_pool)
    q["random"] = g.block_random_subspace(backend, units, rank=1, seed=1)
    targets = ("orig", "v1", "v2", "v3", "A2", "C")
    dmg = {}
    for arm in ("own", "xctl", "random"):
        dmg[arm] = {t: v51.summary(torch, v51.removal(backend, odd[t], units, q[arm], mu_pool)) for t in targets}
        dmg[arm]["cross"] = {n: v51.summary(torch, v51.removal(backend, p, units, q[arm], mu_pool)) for n, p in cross.items()}
        print(arm, json.dumps({t: round(dmg[arm][t]["ce_damage"], 3) for t in targets}),
              "cross", json.dumps({n: round(v["ce_damage"], 3) for n, v in dmg[arm]["cross"].items()}), flush=True)
    cos = {"own|xctl": g.block_cosines(q["own"], q["xctl"])}

    O, X = dmg["own"], dmg["xctl"]
    ce = lambda d, t: d[t]["ce_damage"]
    predictions = {
        'pred_a_reproduce': abs(ce(O, "orig") - prev["A1"]["ce_damage"]) <= REPRO and abs(ce(O, "C") - prev["C"]["ce_damage"]) <= REPRO,
        'pred_b_cross_repaired': all(X["cross"][n]["ce_damage"] <= CROSS_MAX for n in CONTROL_NAMES),
        'pred_c_a1_kept': ce(X, "orig") >= KEEP * ce(O, "orig"),
        'pred_d_rows_kept': X["C"]["ce_ub975"] <= C_UB and ce(X, "A2") >= TRANSFER * ce(X, "orig") and X["A2"]["ce_lb975"] > 0,
        'pred_e_held_out': all(X["cross"][n]["ce_damage"] <= CROSS_MAX for n in HELD_OUT) and ce(X, "v3") >= TRANSFER * ce(O, "v3")
                           and ce(dmg["random"], "orig") <= RAND_FRAC * ce(X, "orig"),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_preposition_cross_inert_result_v1", "candidate_id": "corpus.unit_preposition_cross_inert_v79",
              "units": units, "pairs": {"v1": maps[0], "v2": maps[1], "v3": fourth}, "pool_rows": len(pool.rows),
              "control_rows_even": {"C": len(even_c.rows), **{n: len(p.rows) for n, p in even_ctl.items()}},
              "rows_odd": {k: len(p.rows) for k, p in odd.items()},
              "bars": {"repro": REPRO, "cross_max": CROSS_MAX, "keep": KEEP, "c_ub": C_UB, "transfer": TRANSFER, "rand_frac": RAND_FRAC,
                       "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW, "lambda_own": LAM_OWN, "lambda_xctl_total": LAM_X, "controls_xctl": ["C", *CONTROL_NAMES]}},
              "damage": dmg, "history": hist, "block_cosines": cos,
              "v75_reference": {k: prev[k]["ce_damage"] for k in ("A1", "A2", "C")},
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
