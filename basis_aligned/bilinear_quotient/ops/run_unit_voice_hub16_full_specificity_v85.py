#!/usr/bin/env python3
# BQGATE: five frozen predictions; set (v83 hub16), recipe, split and bars fixed before the run.
"""v85: the full-specificity direction on the voice hub16 set -- can one direction meet row 2 and row 4 together?

Voice on the hub+8 set sat at 3/5: extraction 0.752 (set-limited: exact 0.809) and own-C UB 0.0102 (row 4 missed by 0.0002).
v83 added eight heads by removal-greedy on EVEN rows; the sixteen-head set's exact extraction is 0.857 (LB 0.809), so row 2
is now attainable by the set, but its dim-removal own-C rose to 0.037 (UB 0.066). Here the standard recipe is fitted on the
hub16 set: rank 1 per block, pooled EVEN A1 (voice has no verb-variant maps), complement 1.0, inertness controls = own C
EVEN + the other five behaviours' A1 EVEN (30 per control), 120 steps, lr 0.05, seed 0. Arms: xdas (random start, primary),
xdim (dim start, reported alongside because v82 showed init-dependence), dim, exact, random. Evaluation on ODD rows only.

REGISTERED BEFORE THE RUN (extraction = fraction of full-ablation margin loss restored; removal = CE damage in nat; ODD rows)
    pred_a_reproduce      exact-set extraction A1 odd within 0.02 of v83's hub16 value 0.857 (same set, new runner path).
    pred_b_removal_grows  xdas A1 removal >= 0.458 (hub8 xdas 0.358 + 0.10). Worked: 0.48 True; 0.40 False.
    pred_c_row4           xdas own-C odd ub975 <= 0.01. Worked: 0.008 True; 0.012 False.
    pred_d_row2           xdas extraction A1 odd >= 0.80 with lb95 >= 0.60. Worked: 0.82 / 0.70 True; 0.78 / 0.70 False.
    pred_e_rows_clean     xdas A2 removal >= 0.50 x A1 with lb975 > 0 AND max |cross| <= 0.05 AND random extraction <= 0.10 x xdas extraction.
    Prior: a True; b True (the eight new heads carry 0.20 nat on odd and the direction usually keeps 0.7-1.0x of dim removal);
    c ~40% (the set now has more C-active heads; the penalty must cancel 0.037 rather than 0.000); d ~50% (v81/v82 found the
    objective leaves 0.05-0.07 below exact; 0.857 - 0.06 = 0.80 is on the line); e True.
    c and d are registered on xdas ONLY. xdim's numbers are reported but decide nothing; if xdim meets a row xdas misses, that
    goes on the board as an observation to be confirmed by a split swap, as v84 does for v82's.
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
OUT = ROOT / "circuits/followups/unit_voice_hub16_full_specificity_v85_result.json"
V83 = ROOT / "circuits/followups/unit_voice_greedy_continuation_v83_result.json"
NAME = "voice_frame"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
REPRO, REMOVAL_MIN, C_UB, ROW2, ROW2_LB, TRANSFER, CROSS_MAX, RAND_FRAC = 0.02, 0.458, 0.01, 0.80, 0.60, 0.50, 0.05, 0.10
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 700, 45000


def _plan():
    return {"candidate_id": "corpus.unit_voice_hub16_full_specificity_v85", "lambda": LAM,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 4 * 2 * STEPS, "model_updates": 0, "fit_parameters": 2 * 16 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    all_heads = g.all_head_units()
    v83 = json.loads(V83.read_text())
    sets = {NAME: v83["hub16"]}
    v83_exact = v83["sets"]["hub16"]["extraction"]["A1"]["point"]
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
        R[name] = {"units": units, "extraction": ext, "v83_exact_a1_odd": v83_exact,
                   "removal": rem, "history_xdim": hist_xdim,
                   "block_cos_xdas_dim": g.block_cosines(qd["xdas"], qd["dim"]), "block_cos_xdim_dim": g.block_cosines(qd["xdim"], qd["dim"]), "block_cos_xdim_xdas": g.block_cosines(qd["xdim"], qd["xdas"])}
        print(name, "xdim removal", {t: round(rem["xdim"][t]["ce_damage"], 3) for t in ("A1", "A2", "C")}, "cross", round(max(abs(v) for v in rem["xdim"]["cross"].values()), 3),
              {fam: {k: round(v["point"], 3) for k, v in e.items()} for fam, e in ext.items()}, round(time.perf_counter() - t0), "s", flush=True)

    e = lambda n, fam, k: R[n]["extraction"][fam][k]
    rm = lambda n, arm, t: R[n]["removal"][arm][t]
    mean_abs_cos = {n: sum(abs(v) for v in R[n]["block_cos_xdim_xdas"].values()) / len(R[n]["block_cos_xdim_xdas"]) for n in R}
    n = NAME
    xd = lambda t: rm(n, "xdas", t)
    predictions = {
        'pred_a_reproduce': abs(e(n, "A1", "exact")["point"] - R[n]["v83_exact_a1_odd"]) <= REPRO,
        'pred_b_removal_grows': xd("A1")["ce_damage"] >= REMOVAL_MIN,
        'pred_c_row4': xd("C")["ce_ub975"] <= C_UB,
        'pred_d_row2': e(n, "A1", "xdas")["point"] >= ROW2 and e(n, "A1", "xdas")["lb95"] >= ROW2_LB,
        'pred_e_rows_clean': xd("A2")["ce_damage"] >= TRANSFER * xd("A1")["ce_damage"] and xd("A2")["ce_lb975"] > 0
                             and max(abs(v) for v in xd("cross").values()) <= CROSS_MAX
                             and e(n, "A1", "random")["point"] <= RAND_FRAC * e(n, "A1", "xdas")["point"],
    }
    removal_summary = {arm: {t: (round(rm(n, arm, t)["ce_damage"], 3), round(rm(n, arm, t)["ce_ub975"], 3)) for t in ("A1", "A2", "C")} | {"cross_abs_max": round(max(abs(v) for v in rm(n, arm, "cross").values()), 3)} for arm in ("xdas", "xdim")}
    row2 = [arm for arm in ("xdas", "xdim") if e(n, "A1", arm)["point"] >= ROW2 and e(n, "A1", arm)["lb95"] >= ROW2_LB]
    row4 = [arm for arm in ("xdas", "xdim") if rm(n, arm, "C")["ce_ub975"] <= C_UB]
    summary = {n: {fam: {k: (round(v["point"], 3), round(v["lb95"], 3)) for k, v in R[n]["extraction"][fam].items()} for fam in ("A1", "A2")} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_voice_hub16_full_specificity_result_v1", "candidate_id": "corpus.unit_voice_hub16_full_specificity_v85",
              "row2_met_arms": row2, "row4_met_arms": row4, "removal_summary": removal_summary, "mean_abs_cos_xdim_xdas": mean_abs_cos,
              "bars": {"repro": REPRO, "removal_min": REMOVAL_MIN, "c_ub": C_UB, "row2": ROW2, "row2_lb": ROW2_LB, "transfer": TRANSFER, "cross_max": CROSS_MAX, "rand_frac": RAND_FRAC,
                       "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW, "lambda": LAM}},
              "summary": summary, "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "removal": removal_summary, "row2": row2, "row4": row4, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
