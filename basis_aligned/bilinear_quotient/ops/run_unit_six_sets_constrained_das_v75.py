#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, rank (1 per block), lambda and bars fixed before the run.
"""v75: the v74 constrained-DAS recipe on all six hub+8 sets -- rows 3/4/5 and cross-collateral on odd rows.

v74 (dative hub+8): a rank-1-per-block DAS direction fitted on pooled A1 + verb variants with the complement term and a
C-removal-inertness regularizer (lambda 30, C EVEN rows) held own C at 0.006 while improving A1, A2 and unseen-pair
damage. Same fixed recipe here for every hub+8 set (v66/v67 finals): pool = A1 even rows + the two v15 variant maps
(voice_frame has no maps: A1 even only); complement_weight 1.0; lambda 0 (comparison) and 30 (PRIMARY). Evaluated by v51
removal on ODD rows: A1, A2, own C, the unseen fourth pair where a map exists, the other five behaviours' A1 rows, and a
random rank-1 direction. The diagnostic: verb_complementizer's C (noted/replied -> that, foil whether) reads out the very
that/whether margin the direction writes (v69/v72). If that is right the regularizer cannot make C inert without
destroying the A1 effect -- the constraint should be FORCED to trade; for the other five it should not.

REGISTERED BEFORE THE RUN (CE removal damage in nat on odd rows; primary = lambda 30; "all" = the five sets other than
verb_complementizer unless stated)
    pred_a_c_inert_five     primary own-C point within [-0.02, +0.02] for all five. Worked: 0.006, -0.01, 0.015 True; 0.03 False.
    pred_b_a1_kept_five     primary A1 >= 0.70 x lambda-0 A1 for all five. Worked: 0.565 vs 0.459 True; 0.30 vs 0.46 False.
    pred_c_row5_all_six     primary A2 >= 0.50 x primary A1 AND A2 lb975 > 0, for all six. Worked: 0.313 vs 0.565 True; 0.2 vs 0.565 False.
    pred_d_cross_all_six    primary direction on each other behaviour's A1 odd rows: CE damage <= 0.05, for all six sets;
                            AND random rank-1 on A1 <= 0.05 x primary A1 for all six. Worked: max 0.042, -0.012 True; 0.08 False.
    pred_e_complementizer_forced_tradeoff  for verb_complementizer, primary C point <= 0.05 IMPLIES primary A1 <= 0.70 x
                            lambda-0 A1 (inertness bought only by giving up the effect); if primary C stays > 0.05 the
                            penalty could not buy inertness at all and the prediction is also True. It is False only if C
                            becomes inert AND A1 is kept -- which would refute the shared-axis reading of that C.
                            Worked: C 0.02 with A1 0.5 vs 1.1 True; C 0.30, A1 1.0 True; C 0.01 with A1 1.0 False.
    Prior: a-d True; e True (shared-axis reading). Note pred_e is the one prediction whose False would change a stated
    conclusion; a False on a/b for some set localizes where the recipe does not transfer.
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
OUT = ROOT / "circuits/followups/unit_six_sets_constrained_das_v75_result.json"
SRC = [ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json", ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"]
LAMBDAS, PRIMARY = (0.0, 30.0), 30.0
C_BAND, KEEP, TRANSFER, CROSS_MAX, RAND_FRAC, C_INERT, EXCEPT = 0.02, 0.70, 0.50, 0.05, 0.05, 0.05, "verb_complementizer"
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 600, 40000


def _plan():
    return {"candidate_id": "corpus.unit_six_sets_constrained_das_v75", "lambdas": list(LAMBDAS), "primary": PRIMARY,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 6 * 4 * len(LAMBDAS) * STEPS, "model_updates": 0, "fit_parameters": 6 * len(LAMBDAS) * 13 * 128, "gpu_accessed": False,
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
                   "history": {str(k): v for k, v in hist.items()}, "block_cos_0_vs_primary": g.block_cosines(q[0.0], q[PRIMARY])}
        print(name, {lam: {t: round(v["ce_damage"], 3) for t, v in dmg[str(lam)].items() if t != "cross"} for lam in LAMBDAS},
              "cross_max", round(max(dmg[str(PRIMARY)]["cross"].values()), 3), "rand", round(dmg["random"]["A1"]["ce_damage"], 3),
              round(time.perf_counter() - t0), "s", flush=True)

    P = {n: R[n]["damage"][str(PRIMARY)] for n in R}
    Z = {n: R[n]["damage"]["0.0"] for n in R}
    five = [n for n in R if n != EXCEPT]
    ce = lambda n, lam, t: R[n]["damage"][str(lam)][t]["ce_damage"]
    forced = (P[EXCEPT]["C"]["ce_damage"] > C_INERT) or (ce(EXCEPT, PRIMARY, "A1") <= KEEP * ce(EXCEPT, 0.0, "A1"))
    predictions = {
        'pred_a_c_inert_five': all(abs(P[n]["C"]["ce_damage"]) <= C_BAND for n in five),
        'pred_b_a1_kept_five': all(ce(n, PRIMARY, "A1") >= KEEP * ce(n, 0.0, "A1") for n in five),
        'pred_c_row5_all_six': all(ce(n, PRIMARY, "A2") >= TRANSFER * ce(n, PRIMARY, "A1") and P[n]["A2"]["ce_lb975"] > 0 for n in R),
        'pred_d_cross_all_six': all(max(P[n]["cross"].values()) <= CROSS_MAX for n in R) and all(R[n]["damage"]["random"]["A1"]["ce_damage"] <= RAND_FRAC * ce(n, PRIMARY, "A1") for n in R),
        'pred_e_complementizer_forced_tradeoff': bool(forced),
    }
    per_set = {n: {"A1_0": round(ce(n, 0.0, "A1"), 3), "A1_p": round(ce(n, PRIMARY, "A1"), 3), "A2_p": round(ce(n, PRIMARY, "A2"), 3), "A2_lb": round(P[n]["A2"]["ce_lb975"], 3),
                   "C_0": round(ce(n, 0.0, "C"), 3), "C_p": round(ce(n, PRIMARY, "C"), 3), "C_p_ub": round(P[n]["C"]["ce_ub975"], 3),
                   "unseen_p": round(ce(n, PRIMARY, "unseen"), 3) if "unseen" in P[n] else None, "cross_max_p": round(max(P[n]["cross"].values()), 3)} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_six_sets_constrained_das_result_v1", "candidate_id": "corpus.unit_six_sets_constrained_das_v75",
              "bars": {"c_band": C_BAND, "keep": KEEP, "transfer": TRANSFER, "cross_max": CROSS_MAX, "rand_frac": RAND_FRAC, "c_inert": C_INERT, "except": EXCEPT,
                       "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW, "lambdas": list(LAMBDAS), "primary": PRIMARY}},
              "summary": per_set, "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": per_set, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
