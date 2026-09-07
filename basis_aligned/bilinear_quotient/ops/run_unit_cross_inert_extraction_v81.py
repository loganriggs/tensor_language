#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, recipe (v80 verbatim) and bars fixed before the run.
"""v81: row 2 for the full-specificity directions (v80 xctl) -- does cross-behaviour inertness cost extraction?

v80 fitted one rank-1 direction per block per hub+8 set with inertness controls = own C even + the other five behaviours' A1
even rows (30 per control) and measured removal rows 3/4/5 on odd rows. v76 measured extraction (row 2) only for the
own-C-only direction (cdas): A1 odd 0.753-1.067, A2 odd 0.463-0.95. Here both fits are repeated verbatim (seed 0; the
reproduction control is v80's xctl own-C odd removal number) and v52/v68 extraction is run on A1 ODD and A2 ODD rows for
arms exact / cdas / xdas / dim / random (same procedure as v76: all other heads mean-ablated, set heads keep only the
subspace component, margin recovery relative to the full-ablation null).

REGISTERED BEFORE THE RUN (extraction = (margin(keep) - margin(null)) / (natural - null))
    pred_a_reproduce     refit xctl own-C odd removal damage within 0.005 nat of v80 for all six. Worked: 0.013 vs 0.013 True.
    pred_b_row2_a1       xdas extraction on A1 odd >= 0.80 with lb95 >= 0.60 for at least four of six sets (v76 cdas: four).
                         Worked: 0.85 (0.80) x4 True; x3 False.
    pred_c_within_cdas   |xdas - cdas| extraction on A1 odd <= 0.05 for all six. Worked: 0.87 vs 0.88 True; 0.80 vs 0.88 False.
    pred_d_a2_kept       xdas extraction on A2 odd >= 0.85 x cdas extraction on A2 odd for all six. Worked: 0.70 vs 0.75 True; 0.55 vs 0.75 False.
    pred_e_random        random rank-1 (same blocks) extraction on A1 odd <= 0.10 x xdas for all six. Worked: 0.02 vs 0.85 True; 0.15 False.
    Prior: a True; b, c, d True is the hypothesis (v80 kept A1 removal at 0.96-1.01x and the block cosines were 0.69-1.00,
    so the direction barely moved); c False on a set says the cross-inert component carried sufficiency there; e True.
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
OUT = ROOT / "circuits/followups/unit_cross_inert_extraction_v81_result.json"
SRC = [ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json", ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"]
PREV = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
REPRO, ROW2, ROW2_LB, N_ROW2, WITHIN, A2_KEEP, RAND_FRAC = 0.005, 0.80, 0.60, 4, 0.05, 0.85, 0.10
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 700, 45000


def _plan():
    return {"candidate_id": "corpus.unit_cross_inert_extraction_v81", "lambda": LAM,
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
        qd["cdas"], _ = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW, controls=(even_c,), control_weight=LAM, mu=mu)
        xc = (even_c,) + tuple(cross_even[n] for n in modules if n != name)
        qd["xdas"], _ = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW, controls=xc, control_weight=LAM * len(xc), mu=mu)
        qd["dim"] = g.block_diff_in_means(backend, even_a1, units)
        qd["random"] = g.block_random_subspace(backend, units, rank=1, seed=1)
        repro = v51.summary(torch, v51.removal(backend, odd_c, units, qd["xdas"], mu))["ce_damage"]

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
            for k, qs in (("exact", g.block_identity(backend, units)), ("cdas", qd["cdas"]), ("xdas", qd["xdas"]), ("dim", qd["dim"]), ("random", qd["random"])):
                _, m_arm = margins(qs)
                ext[fam][k] = v52._boot_ratio(torch, [x - y for x, y in zip(m_arm, m_null)], den)
        R[name] = {"units": units, "extraction": ext, "repro_c_odd": repro, "v80_c_odd": prev[name]["damage"]["xctl"]["C"]["ce_damage"],
                   "block_cos_xdas_dim": g.block_cosines(qd["xdas"], qd["dim"]), "block_cos_xdas_cdas": g.block_cosines(qd["xdas"], qd["cdas"])}
        print(name, "repro", round(repro, 4), "v80", round(R[name]["v80_c_odd"], 4),
              {fam: {k: round(v["point"], 3) for k, v in e.items()} for fam, e in ext.items()}, round(time.perf_counter() - t0), "s", flush=True)

    e = lambda n, fam, k: R[n]["extraction"][fam][k]
    row2 = [n for n in R if e(n, "A1", "xdas")["point"] >= ROW2 and e(n, "A1", "xdas")["lb95"] >= ROW2_LB]
    predictions = {
        'pred_a_reproduce': all(abs(R[n]["repro_c_odd"] - R[n]["v80_c_odd"]) <= REPRO for n in R),
        'pred_b_row2_a1': len(row2) >= N_ROW2,
        'pred_c_within_cdas': all(abs(e(n, "A1", "xdas")["point"] - e(n, "A1", "cdas")["point"]) <= WITHIN for n in R),
        'pred_d_a2_kept': all(e(n, "A2", "xdas")["point"] >= A2_KEEP * e(n, "A2", "cdas")["point"] for n in R),
        'pred_e_random': all(e(n, "A1", "random")["point"] <= RAND_FRAC * e(n, "A1", "xdas")["point"] for n in R),
    }
    summary = {n: {fam: {k: (round(v["point"], 3), round(v["lb95"], 3)) for k, v in R[n]["extraction"][fam].items()} for fam in ("A1", "A2")} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_cross_inert_extraction_result_v1", "candidate_id": "corpus.unit_cross_inert_extraction_v81",
              "row2_met_a1": row2, "bars": {"repro": REPRO, "row2": ROW2, "row2_lb": ROW2_LB, "n_row2": N_ROW2, "within": WITHIN, "a2_keep": A2_KEEP, "rand_frac": RAND_FRAC,
                                             "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW, "lambda": LAM}},
              "summary": summary, "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
