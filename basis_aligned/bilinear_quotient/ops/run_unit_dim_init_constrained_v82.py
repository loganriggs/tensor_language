#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, recipe and bars fixed before the run.
"""v82: is the constrained-DAS optimum init-dependent? Full-specificity fit started from the diff-in-means direction, six sets.

Every fitted direction so far starts from seeded random raws (0.02 * randn). On quantifier the DAS objective extracts 0.757
where the per-block diff-in-means extracts 0.824 (v76/v81), and v77 showed pooling was not the cause. If the margin-matching
objective has several rank-1 optima, the start point decides which one we report. Arms (all hub+8, evaluated on ODD rows):
exact; xdas = v80/v81 recipe from random init (reproduction control = v80's xctl own-C odd removal); xdim = the same recipe
started from block_diff_in_means on A1 even (g.fit_block_subspace_constrained(init=...)); dim; random rank-1. For xdim the
removal rows are measured too (A1, A2, own C, other five A1), since a different optimum must re-earn rows 3/4/5.

REGISTERED BEFORE THE RUN (extraction = (margin(keep) - margin(null)) / (natural - null); damage in nat)
    pred_a_reproduce      xdas own-C odd removal within 0.005 of v80 for all six. Worked: 0.0134 vs 0.0134 True.
    pred_b_keeps_better   xdim extraction on A1 odd >= max(xdas, dim) - 0.03 for all six. Worked: 0.82 vs 0.824 True; 0.76 vs 0.824 False.
    pred_c_specific       xdim own-C odd ub975 <= xdas ub975 + 0.01 AND max |cross| <= 0.05 for all six. Worked: 0.020 vs 0.018, 0.02 True; 0.08 False.
    pred_d_rows35         xdim removal A1 odd >= 0.85 x xdas A1 AND A2 >= 0.50 x A1 with lb975 > 0, for all six. Worked: 0.70 vs 0.73, 0.85 True; 0.55 vs 0.73 False.
    pred_e_init_matters   random rank-1 extraction <= 0.10 x xdim for all six AND on at least one set the mean block |cos(xdim, xdas)| < 0.95
                          (the two starts reach different directions). Worked: 0.01, cos 0.88 True; all cos >= 0.97 False.
    Prior: a True; e True (v76: cos(cdas, dim) was 0.4-0.8 on some blocks); b is the hypothesis -- if the objective is
    flat between the two optima, starting at dim keeps dim's sufficiency and the constraint adds specificity. b False with
    c, d True: the objective actively moves away from the dim optimum (DAS extraction shortfall is a property of the
    objective, not the start). Reported either way; no second start point will be tried.
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
import run_unit_extraction_four_sets_v52 as v52

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_dim_init_constrained_v82_result.json"
SRC = [ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json", ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"]
PREV = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
REPRO, BETTER_SLACK, C_UB_SLACK, CROSS_MAX, KEEP, TRANSFER, RAND_FRAC, COS_DIFF = 0.005, 0.03, 0.01, 0.05, 0.85, 0.50, 0.10, 0.95
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 700, 45000


def _plan():
    return {"candidate_id": "corpus.unit_dim_init_constrained_v82", "lambda": LAM,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 6 * 4 * 2 * STEPS, "model_updates": 0, "fit_parameters": 6 * 2 * 13 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    all_heads = g.all_head_units()
    sets = {n: r["final"] for p in SRC for n, r in json.loads(p.read_text())["sets"].items()}
    prev = json.loads(PREV.read_text())["sets"]
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    cross_even = {n: g.prepare(backend, g.rows_of(m, "A1")[0::2]) for n, m in modules.items()}
    cross_odd = {n: g.prepare(backend, g.rows_of(m, "A1")[1::2]) for n, m in modules.items()}

    def mu_of(prep, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    R = {}
    for name, units in sets.items():
        m = modules[name]
        a1 = g.rows_of(m, "A1")
        maps = v15.SETS[name][2] if name in v15.SETS else ()
        pool = g.prepare(backend, a1[0::2] + [r for mp in maps for r in g.lexical_variant(a1, mp)[0::2]])
        even_c = g.prepare(backend, g.rows_of(m, "C")[0::2])
        odd_c = g.prepare(backend, g.rows_of(m, "C")[1::2])
        even_a1 = g.prepare(backend, a1[0::2])
        odd = {"A1": g.prepare(backend, a1[1::2]), "A2": g.prepare(backend, g.rows_of(m, "A2")[1::2])}
        mu = mu_of(pool, units)
        qd = {}
        xc = (even_c,) + tuple(cross_even[n] for n in modules if n != name)
        qd["xdas"], _ = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW, controls=xc, control_weight=LAM * len(xc), mu=mu)
        qd["dim"] = g.block_diff_in_means(backend, even_a1, units)
        qd["xdim"], hist_xdim = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW, controls=xc, control_weight=LAM * len(xc), mu=mu, init=qd["dim"])
        qd["random"] = g.block_random_subspace(backend, units, rank=1, seed=1)
        repro = v51.summary(torch, v51.removal(backend, odd_c, units, qd["xdas"], mu))["ce_damage"]
        rem = {}
        for arm in ("xdas", "xdim"):
            rem[arm] = {"A1": v51.summary(torch, v51.removal(backend, odd["A1"], units, qd[arm], mu)), "A2": v51.summary(torch, v51.removal(backend, odd["A2"], units, qd[arm], mu)),
                        "C": v51.summary(torch, v51.removal(backend, odd_c, units, qd[arm], mu)),
                        "cross": {n: v51.summary(torch, v51.removal(backend, p, units, qd[arm], mu))["ce_damage"] for n, p in cross_odd.items() if n != name}}

        ext = {}
        for fam, prep in odd.items():
            mu_all = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in all_heads}

            def margins(qs):
                order = v52.ordered_units(units)
                bq = v52.block_q(torch, backend.device, units, qs)
                nat, arm = [], []
                for side in ("base", "donor"):
                    batch = prep.base_batch if side == "base" else prep.donor_batch
                    cache = prep.base_cache if side == "base" else prep.donor_cache
                    bg = dict(cache)
                    for rid in batch.row_ids:
                        for u in all_heads:
                            bg[(rid, u)] = mu_all[u]
                    af_n = g.forward_units(backend, batch, units=[])
                    af_a = g.forward_units(backend, batch, units=order, donor_cache=bg, base_cache=cache, q=bq, complement=True)
                    nat += (af_n[:, 0] - af_n[:, 1]).tolist(); arm += (af_a[:, 0] - af_a[:, 1]).tolist()
                return nat, arm
            nat, m_null = margins(None)
            den = [a - b for a, b in zip(nat, m_null)]
            ext[fam] = {}
            for k, qs in (("exact", g.block_identity(backend, units)), ("xdas", qd["xdas"]), ("xdim", qd["xdim"]), ("dim", qd["dim"]), ("random", qd["random"])):
                _, m_arm = margins(qs)
                ext[fam][k] = v52._boot_ratio(torch, [x - y for x, y in zip(m_arm, m_null)], den)
        R[name] = {"units": units, "extraction": ext, "repro_c_odd": repro, "v80_c_odd": prev[name]["damage"]["xctl"]["C"]["ce_damage"],
                   "removal": rem, "history_xdim": hist_xdim,
                   "block_cos_xdas_dim": g.block_cosines(qd["xdas"], qd["dim"]), "block_cos_xdim_dim": g.block_cosines(qd["xdim"], qd["dim"]), "block_cos_xdim_xdas": g.block_cosines(qd["xdim"], qd["xdas"])}
        print(name, "repro", round(repro, 4), "v80", round(R[name]["v80_c_odd"], 4), "xdim removal", {t: round(rem["xdim"][t]["ce_damage"], 3) for t in ("A1", "A2", "C")}, "cross", round(max(abs(v) for v in rem["xdim"]["cross"].values()), 3),
              {fam: {k: round(v["point"], 3) for k, v in e.items()} for fam, e in ext.items()}, round(time.perf_counter() - t0), "s", flush=True)

    e = lambda n, fam, k: R[n]["extraction"][fam][k]
    rm = lambda n, arm, t: R[n]["removal"][arm][t]
    mean_abs_cos = {n: sum(abs(v) for v in R[n]["block_cos_xdim_xdas"].values()) / len(R[n]["block_cos_xdim_xdas"]) for n in R}
    predictions = {
        'pred_a_reproduce': all(abs(R[n]["repro_c_odd"] - R[n]["v80_c_odd"]) <= REPRO for n in R),
        'pred_b_keeps_better': all(e(n, "A1", "xdim")["point"] >= max(e(n, "A1", "xdas")["point"], e(n, "A1", "dim")["point"]) - BETTER_SLACK for n in R),
        'pred_c_specific': all(rm(n, "xdim", "C")["ce_ub975"] <= rm(n, "xdas", "C")["ce_ub975"] + C_UB_SLACK and max(abs(v) for v in rm(n, "xdim", "cross").values()) <= CROSS_MAX for n in R),
        'pred_d_rows35': all(rm(n, "xdim", "A1")["ce_damage"] >= KEEP * rm(n, "xdas", "A1")["ce_damage"] and rm(n, "xdim", "A2")["ce_damage"] >= TRANSFER * rm(n, "xdim", "A1")["ce_damage"]
                             and rm(n, "xdim", "A2")["ce_lb975"] > 0 for n in R),
        'pred_e_init_matters': all(e(n, "A1", "random")["point"] <= RAND_FRAC * e(n, "A1", "xdim")["point"] for n in R) and any(c < COS_DIFF for c in mean_abs_cos.values()),
    }
    row2 = [n for n in R if e(n, "A1", "xdim")["point"] >= 0.8 and e(n, "A1", "xdim")["lb95"] >= 0.6]
    summary = {n: {fam: {k: (round(v["point"], 3), round(v["lb95"], 3)) for k, v in R[n]["extraction"][fam].items()} for fam in ("A1", "A2")} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_dim_init_constrained_result_v1", "candidate_id": "corpus.unit_dim_init_constrained_v82",
              "row2_met_a1_xdim": row2, "mean_abs_cos_xdim_xdas": mean_abs_cos, "bars": {"repro": REPRO, "better_slack": BETTER_SLACK, "c_ub_slack": C_UB_SLACK, "cross_max": CROSS_MAX, "keep": KEEP, "transfer": TRANSFER, "rand_frac": RAND_FRAC, "cos_diff": COS_DIFF,
                                             "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW, "lambda": LAM}},
              "summary": summary, "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
