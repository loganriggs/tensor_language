#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, recipe, split and bars fixed before the run.
"""v86: cross-fitted row 4 -- two directions per set, each evaluated only on the half it never saw, pooled to 32 documents.

v80 (fit EVEN, evaluate ODD) and v84 (fit ODD, evaluate EVEN) give opposite row-4 verdicts on the same hub+8 sets:
own-C removal ub975 dative 0.026 / 0.004, voice 0.0102 / -0.002, quantifier 0.018 / 0.022. Sixteen evaluation documents
per half put the interval width at about the 0.01 bar itself. This run fits the standard full-specificity direction
(rank 1 per block, complement 1.0, own C + other five A1 as inertness controls at 30 each, 120 steps, lr 0.05, seed 0)
once per half and evaluates each direction on the OTHER half only, then concatenates the per-document CE damages
(32 documents per family) before bootstrapping. Nothing is evaluated in-sample; nothing is chosen after seeing a half.

REGISTERED BEFORE THE RUN (pooled out-of-sample CE removal damage in nat)
    pred_a_reproduce     each half's xdas A1 matches its earlier run within 0.005: even-fit/odd-eval vs v80 xctl, odd-fit/even-eval vs v84 xdas, all six.
    pred_b_row4_pooled   pooled own-C ub975 <= 0.01 on at least FOUR of six sets. Worked: 4 True; 3 False.
                         Named prior: verb_preposition, polarity_licensing yes; voice_frame likely; dative ~50%; quantifier_number, verb_complementizer no.
    pred_c_rows35_pooled pooled A1 lb975 > 0 AND pooled A2 >= 0.50 x pooled A1 with A2 lb975 > 0, all six. Worked: 0.30 vs 0.55 True; 0.25 vs 0.55 False.
    pred_d_specific      pooled max |cross| <= 0.05 AND pooled random rank-1 A1 <= 0.05 x pooled xdas A1, all six.
    pred_e_narrows       pooled own-C half-width (ub975 - point) <= 0.80 x the mean of the two halves' half-widths, all six (sqrt-2 narrowing gives 0.71).
                         Worked: 0.010 vs 0.015 True; 0.014 vs 0.015 False.
    Prior: a True (seed-fixed fits); b ~60%; c ~70% (dative A2 sits at 0.51-0.57x); d True; e ~85%.
    b False: the tier list keeps rows 4 open for the sets that miss, at 32 documents this time, and no further width is available
    from this C set -- the next step would be a second C sibling per behaviour, not another split.
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
OUT = ROOT / "circuits/followups/unit_cross_fit_row4_v86_result.json"
SRC = [ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json", ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"]
V80 = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
V84 = ROOT / "circuits/followups/unit_split_swap_dim_start_v84_result.json"
LAM_PER_CONTROL, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
REPRO, C_UB, N_ROW4, TRANSFER, CROSS_MAX, RAND_FRAC, NARROW = 0.005, 0.01, 4, 0.50, 0.05, 0.05, 0.80
HALVES = {"even": slice(0, None, 2), "odd": slice(1, None, 2)}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 900, 60000


def _plan():
    return {"candidate_id": "corpus.unit_cross_fit_row4_v86", "lambda_per_control": LAM_PER_CONTROL,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 6 * 2 * 2 * STEPS, "model_updates": 0, "fit_parameters": 6 * 2 * 13 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _pool(ds):
    return {k: [x for d in ds for x in d[k]] for k in ("ce", "margin", "kl", "top1_change")}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    sets = {n: r["final"] for p in SRC for n, r in json.loads(p.read_text())["sets"].items()}
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    prev = {"even": {n: v["damage"]["xctl"]["A1"]["ce_damage"] for n, v in json.loads(V80.read_text())["sets"].items()},
            "odd": {n: v["damage"]["xdas"]["A1"]["ce_damage"] for n, v in json.loads(V84.read_text())["sets"].items()}}
    a1 = {(n, h): g.prepare(backend, g.rows_of(m, "A1")[sl]) for n, m in modules.items() for h, sl in HALVES.items()}

    def mu_of(prep, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    R = {}
    for name, units in sets.items():
        m = modules[name]
        rows_a1 = g.rows_of(m, "A1")
        maps, fourth = (v15.SETS[name][2], v15.SETS[name][3]) if name in v15.SETS else ((), None)
        raw, halves = {"xdas": {}, "random": {}}, {}
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        for fit_h, ev_h in (("even", "odd"), ("odd", "even")):
            fs, es = HALVES[fit_h], HALVES[ev_h]
            pool = g.prepare(backend, rows_a1[fs] + [r for mp in maps for r in g.lexical_variant(rows_a1, mp)[fs]])
            fit_c = g.prepare(backend, g.rows_of(m, "C")[fs])
            ev = {"A1": a1[(name, ev_h)], "A2": g.prepare(backend, g.rows_of(m, "A2")[es]), "C": g.prepare(backend, g.rows_of(m, "C")[es])}
            if fourth is not None:
                ev["unseen"] = g.prepare(backend, g.lexical_variant(rows_a1, fourth)[es])
            ev.update({f"cross:{n}": a1[(n, ev_h)] for n in modules if n != name})
            mu = mu_of(pool, units)
            xc = (fit_c,) + tuple(a1[(n, fit_h)] for n in modules if n != name)
            q, hist = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW,
                                                       controls=xc, control_weight=LAM_PER_CONTROL * len(xc), mu=mu)
            for arm, qq in (("xdas", q), ("random", q_rand)):
                for t, p in ev.items():
                    raw[arm].setdefault(t, []).append(v51.removal(backend, p, units, qq, mu))
            halves[f"fit_{fit_h}"] = {t: v51.summary(torch, raw["xdas"][t][-1]) for t in ("A1", "A2", "C")} | {"history_tail": hist[-1] if isinstance(hist, list) else hist}
        pooled = {arm: {t: v51.summary(torch, _pool(ds)) for t, ds in raw[arm].items()} for arm in raw}
        R[name] = {"units": units, "halves": halves, "pooled": pooled, "prev": {h: prev[h][name] for h in HALVES}}
        print(name, "pooled xdas", {t: (round(pooled["xdas"][t]["ce_damage"], 3), round(pooled["xdas"][t]["ce_ub975"], 3)) for t in ("A1", "A2", "C")},
              "cross", round(max(abs(pooled["xdas"][t]["ce_damage"]) for t in pooled["xdas"] if t.startswith("cross:")), 3),
              "rand", round(pooled["random"]["A1"]["ce_damage"], 3), round(time.perf_counter() - t0), "s", flush=True)

    P = {n: R[n]["pooled"]["xdas"] for n in R}
    hw = lambda s: s["ce_ub975"] - s["ce_damage"]
    row4 = [n for n in R if P[n]["C"]["ce_ub975"] <= C_UB]
    predictions = {
        'pred_a_reproduce': all(abs(R[n]["halves"][f"fit_{h}"]["A1"]["ce_damage"] - R[n]["prev"][h]) <= REPRO for n in R for h in HALVES),
        'pred_b_row4_pooled': len(row4) >= N_ROW4,
        'pred_c_rows35_pooled': all(P[n]["A1"]["ce_lb975"] > 0 and P[n]["A2"]["ce_damage"] >= TRANSFER * P[n]["A1"]["ce_damage"] and P[n]["A2"]["ce_lb975"] > 0 for n in R),
        'pred_d_specific': all(max(abs(P[n][t]["ce_damage"]) for t in P[n] if t.startswith("cross:")) <= CROSS_MAX
                               and R[n]["pooled"]["random"]["A1"]["ce_damage"] <= RAND_FRAC * P[n]["A1"]["ce_damage"] for n in R),
        'pred_e_narrows': all(hw(P[n]["C"]) <= NARROW * 0.5 * (hw(R[n]["halves"]["fit_even"]["C"]) + hw(R[n]["halves"]["fit_odd"]["C"])) for n in R),
    }
    summary = {n: {"A1": round(P[n]["A1"]["ce_damage"], 3), "A1_lb": round(P[n]["A1"]["ce_lb975"], 3), "A2": round(P[n]["A2"]["ce_damage"], 3),
                   "C": round(P[n]["C"]["ce_damage"], 3), "C_ub": round(P[n]["C"]["ce_ub975"], 3),
                   "C_ub_halves": [round(R[n]["halves"][f"fit_{h}"]["C"]["ce_ub975"], 3) for h in ("even", "odd")],
                   "unseen": round(P[n]["unseen"]["ce_damage"], 3) if "unseen" in P[n] else None,
                   "cross_abs_max": round(max(abs(P[n][t]["ce_damage"]) for t in P[n] if t.startswith("cross:")), 3),
                   "random_A1": round(R[n]["pooled"]["random"]["A1"]["ce_damage"], 3)} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_cross_fit_row4_result_v1", "candidate_id": "corpus.unit_cross_fit_row4_v86",
              "row4_met_pooled": row4, "summary": summary,
              "bars": {"repro": REPRO, "c_ub": C_UB, "n_row4": N_ROW4, "transfer": TRANSFER, "cross_max": CROSS_MAX, "rand_frac": RAND_FRAC, "narrow": NARROW,
                       "protocol": "cross-fitted: fit EVEN->eval ODD and fit ODD->eval EVEN, per-document damages pooled (32 per family)",
                       "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW, "lambda_per_control": LAM_PER_CONTROL}},
              "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "row4": row4, "summary": summary, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
