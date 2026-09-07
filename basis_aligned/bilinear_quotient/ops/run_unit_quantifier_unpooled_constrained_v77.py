#!/usr/bin/env python3
# BQGATE: five frozen predictions; set, recipe and bars fixed before the run.
"""v77: quantifier_number -- one licensed repair: the constrained direction fitted on A1 alone (no verb-variant pooling).

v76: the pooled constrained direction for quantifier hub+8 extracts 0.753 on A1 odd (bar 0.80) while the exact set
(0.816) and the A1-only diff-in-means direction (0.824) pass. The unconstrained POOLED fit is also 0.746, so the loss is
from pooling {Each/All} with {Neither/Several} and {One/Many}, not from the C-inertness term. One repair: fit the same
constrained recipe (rank 1 per block, complement term, lambda 30 on C even rows) on the A1 EVEN rows only, then re-measure
on ODD rows: extraction A1/A2, removal A1/A2/C, cross-collateral, and the unseen pair {Either/Some} -- where the
un-pooled direction is expected to LOSE what pooling bought (v75 pooled: unseen 0.687). Both directions are reported.

REGISTERED BEFORE THE RUN (odd rows; "unpooled" = A1-only constrained fit; "pooled" = the v75/v76 recipe, refit here)
    pred_a_row2_repaired   unpooled extraction on A1 odd >= 0.80 with lb95 >= 0.60. Worked: 0.83 (0.78) True; 0.76 False.
    pred_b_c_inert         unpooled own-C odd removal point within [-0.02, +0.02]. Worked: 0.01 True; 0.04 False.
    pred_c_row5            unpooled A2 removal >= 0.50 x unpooled A1 removal AND lb975 > 0; AND unpooled A2 extraction >= 0.60.
                           Worked: 0.60 vs 0.74, ext 0.80 True; 0.30 vs 0.74 False.
    pred_d_transfer_cost   unpooled removal on the unseen pair <= 0.80 x pooled removal on it (the price of the repair).
                           Worked: 0.45 vs 0.69 True; 0.65 vs 0.69 False (then pooling bought nothing for quantifier).
    pred_e_controls        random rank-1 extraction on A1 odd <= 0.10 x unpooled; unpooled cross-collateral max <= 0.05.
                           Worked: 0.00, 0.02 True; 0.08 cross False.
    Prior: a, b, c True (the A1-only diff-in-means already extracts 0.82 and the constraint cost nothing elsewhere);
    d True; e True. a False would mean the constraint interacts with the fit on this set -- report, no further repair.
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
OUT = ROOT / "circuits/followups/unit_quantifier_unpooled_constrained_v77_result.json"
SRC = ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"
NAME = "quantifier_number"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
ROW2, ROW2_LB, C_BAND, TRANSFER, A2_EXT, COST, RAND_FRAC, CROSS_MAX = 0.80, 0.60, 0.02, 0.50, 0.60, 0.80, 0.10, 0.05
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 300, 20000


def _plan():
    return {"candidate_id": "corpus.unit_quantifier_unpooled_constrained_v77", "lambda": LAM,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * 4 * STEPS, "model_updates": 0, "fit_parameters": 2 * 10 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    all_heads = g.all_head_units()
    units = json.loads(SRC.read_text())["sets"][NAME]["final"]
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    m = modules[NAME]
    _, _, maps, fourth = v15.SETS[NAME]
    a1 = g.rows_of(m, "A1")
    even_a1 = g.prepare(backend, a1[0::2])
    pool = g.prepare(backend, a1[0::2] + [r for mp in maps for r in g.lexical_variant(a1, mp)[0::2]])
    even_c = g.prepare(backend, g.rows_of(m, "C")[0::2])
    odd = {"A1": g.prepare(backend, a1[1::2]), "A2": g.prepare(backend, g.rows_of(m, "A2")[1::2]), "C": g.prepare(backend, g.rows_of(m, "C")[1::2]),
           "unseen": g.prepare(backend, g.lexical_variant(a1, fourth)[1::2])}
    cross_odd = {n: g.prepare(backend, g.rows_of(mm, "A1")[1::2]) for n, mm in modules.items() if n != NAME}

    def mu_of(prep):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    fits = {"unpooled": (even_a1, mu_of(even_a1)), "pooled": (pool, mu_of(pool))}
    q, mu = {}, {}
    for k, (prep, mk) in fits.items():
        q[k], _ = g.fit_block_subspace_constrained(backend, prep, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW, controls=(even_c,), control_weight=LAM, mu=mk)
        mu[k] = mk
    q["random"], mu["random"] = g.block_random_subspace(backend, units, rank=1, seed=1), mu["unpooled"]

    removal = {k: {t: v51.summary(torch, v51.removal(backend, odd[t], units, q[k], mu[k])) for t in ("A1", "A2", "C", "unseen")} for k in ("unpooled", "pooled")}
    cross = {k: {n: v51.summary(torch, v51.removal(backend, p, units, q[k], mu[k]))["ce_damage"] for n, p in cross_odd.items()} for k in ("unpooled", "pooled")}

    ext = {}
    for fam in ("A1", "A2"):
        prep = odd[fam]
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
        for k, qs in (("exact", g.block_identity(backend, units)), ("unpooled", q["unpooled"]), ("pooled", q["pooled"]), ("random", q["random"])):
            _, m_arm = margins(qs)
            ext[fam][k] = v52._boot_ratio(torch, [x - y for x, y in zip(m_arm, m_null)], den)

    U, Pd = removal["unpooled"], removal["pooled"]
    predictions = {
        'pred_a_row2_repaired': ext["A1"]["unpooled"]["point"] >= ROW2 and ext["A1"]["unpooled"]["lb95"] >= ROW2_LB,
        'pred_b_c_inert': abs(U["C"]["ce_damage"]) <= C_BAND,
        'pred_c_row5': U["A2"]["ce_damage"] >= TRANSFER * U["A1"]["ce_damage"] and U["A2"]["ce_lb975"] > 0 and ext["A2"]["unpooled"]["point"] >= A2_EXT,
        'pred_d_transfer_cost': U["unseen"]["ce_damage"] <= COST * Pd["unseen"]["ce_damage"],
        'pred_e_controls': ext["A1"]["random"]["point"] <= RAND_FRAC * ext["A1"]["unpooled"]["point"] and max(cross["unpooled"].values()) <= CROSS_MAX,
    }
    summary = {"removal": {k: {t: round(v["ce_damage"], 3) for t, v in r.items()} for k, r in removal.items()},
               "extraction": {fam: {k: (round(v["point"], 3), round(v["lb95"], 3)) for k, v in e.items()} for fam, e in ext.items()},
               "cross_max": {k: round(max(v.values()), 3) for k, v in cross.items()}}
    result = {"predictions": predictions, "schema": "circuit_unit_quantifier_unpooled_constrained_result_v1", "candidate_id": "corpus.unit_quantifier_unpooled_constrained_v77",
              "units": units, "maps": list(maps), "fourth": fourth, "bars": {"row2": ROW2, "row2_lb": ROW2_LB, "c_band": C_BAND, "transfer": TRANSFER, "a2_ext": A2_EXT, "cost": COST, "rand_frac": RAND_FRAC, "cross_max": CROSS_MAX,
              "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW, "lambda": LAM}},
              "summary": summary, "removal": removal, "cross": cross, "extraction": ext, "block_cos_unpooled_pooled": g.block_cosines(q["unpooled"], q["pooled"]),
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
