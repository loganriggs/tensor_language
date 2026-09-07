#!/usr/bin/env python3
# BQGATE: five frozen predictions; set (v67 dative hub+8), recipe, split and bars fixed before the run.
"""v89: does the dative direction miss the A2 frame because the A2 frame is never in the fit pool?

dative is the one set where the full-specificity direction is weak on row 5: A2 removal 0.323 = 0.57x A1 (v80) and A2-frame
extraction 0.446 against 0.854 for the exact set (v81), while A1 extraction is 0.826. The fit pool is A1 EVEN + verb-variant
rows, all in the A1 frame. Hypothesis: the direction is frame-specific because it never saw the second frame. Test: arm `xa2`
= the standard recipe with A2 EVEN rows added to the pool (controls identical: own C EVEN + other five A1 EVEN, 30 each);
arm `xdas` = the standard recipe. Both evaluated on ODD rows only: extraction (A1, A2; exact, random alongside), removal
(A1, A2, C, other five, unseen fourth pair). v87 found dative near plateau at hub+8 (gain 0.084), so the set is hub+8.

REGISTERED BEFORE THE RUN (extraction = fraction of full-ablation margin restored; removal = CE damage in nat; ODD rows)
    pred_a_reproduce   xdas A1 odd removal within 0.005 of v80 (0.569) and xdas A2 odd extraction within 0.02 of v81 (0.446).
    pred_b_a2_gain     xa2 A2 odd extraction >= 0.65 with lb95 >= 0.50. Worked: 0.70 / 0.62 True; 0.60 / 0.55 False.
    pred_c_a1_kept     xa2 A1 odd removal >= 0.85 x xdas A1 AND xa2 A1 odd extraction >= xdas A1 extraction - 0.03. Worked: 0.50 vs 0.57, 0.81 vs 0.83 True; 0.45 False.
    pred_d_specific    xa2 own-C odd ub975 <= xdas ub975 + 0.01 AND max |cross| <= 0.05. Worked: 0.030 vs 0.026 True; 0.040 False.
    pred_e_transfer    xa2 unseen-fourth-pair removal >= 0.80 x xdas unseen AND xa2 A2 removal >= 0.70 x xa2 A1 with lb975 > 0 AND random extraction <= 0.10 x xa2 A1 extraction.
    Prior: a True; b ~55% -- if the A2 gap is frame-specificity of a rank-1 direction, pooling fixes it; if the two frames need
    two directions per block (rank 2), one direction fitted on both will split the difference and b fails while c also slips;
    c ~70%; d ~75%; e ~60%.
    b False with c True: the frames are not co-linear at rank 1 in these blocks -- report as such; rank stays 1 (no rank raise on a null).
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
OUT = ROOT / "circuits/followups/unit_dative_a2_pool_v89_result.json"
SRC = ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"
V80 = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
V81 = ROOT / "circuits/followups/unit_cross_inert_extraction_v81_result.json"
NAME = "dative"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
REPRO_REM, REPRO_EXT, A2_EXT, A2_EXT_LB, KEEP, A1_EXT_SLACK, C_UB_SLACK, CROSS_MAX, UNSEEN_KEEP, A2_TRANSFER, RAND_FRAC = 0.005, 0.02, 0.65, 0.50, 0.85, 0.03, 0.01, 0.05, 0.80, 0.70, 0.10
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 700, 45000


def _plan():
    return {"candidate_id": "corpus.unit_dative_a2_pool_v89", "lambda": LAM,
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
    sets = {NAME: json.loads(SRC.read_text())["sets"][NAME]["final"]}
    v80_a1 = json.loads(V80.read_text())["sets"][NAME]["damage"]["xctl"]["A1"]["ce_damage"]
    v81_a2 = json.loads(V81.read_text())["sets"][NAME]["extraction"]["A2"]["xdas"]["point"]
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
        odd = {"A1": g.prepare(backend, a1[1::2]), "A2": g.prepare(backend, g.rows_of(m, "A2")[1::2])}
        fourth = v15.SETS[name][3] if name in v15.SETS else None
        odd_unseen = g.prepare(backend, g.lexical_variant(a1, fourth)[1::2])
        mu = mu_of(pool, units)
        qd = {}
        xc = (even_c,) + tuple(cross_even[n] for n in modules if n != name)
        qd["xdas"], _ = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW, controls=xc, control_weight=LAM * len(xc), mu=mu)
        pool_a2 = g.prepare(backend, a1[0::2] + [r for mp in maps for r in g.lexical_variant(a1, mp)[0::2]] + g.rows_of(m, "A2")[0::2])
        mu_a2 = mu_of(pool_a2, units)
        qd["xa2"], hist_xa2 = g.fit_block_subspace_constrained(backend, pool_a2, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW, controls=xc, control_weight=LAM * len(xc), mu=mu_a2)
        qd["random"] = g.block_random_subspace(backend, units, rank=1, seed=1)
        rem = {}
        mus = {"xdas": mu, "xa2": mu_a2}
        for arm in ("xdas", "xa2"):
            rem[arm] = {t: v51.summary(torch, v51.removal(backend, p, units, qd[arm], mus[arm])) for t, p in (("A1", odd["A1"]), ("A2", odd["A2"]), ("C", odd_c), ("unseen", odd_unseen))}
            rem[arm]["cross"] = {n: v51.summary(torch, v51.removal(backend, p, units, qd[arm], mus[arm]))["ce_damage"] for n, p in cross_odd.items() if n != name}

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
            for k, qs in (("exact", g.block_identity(backend, units)), ("xdas", qd["xdas"]), ("xa2", qd["xa2"]), ("random", qd["random"])):
                _, m_arm = margins(qs)
                ext[fam][k] = v52._boot_ratio(torch, [x - y for x, y in zip(m_arm, m_null)], den)
        R[name] = {"units": units, "extraction": ext, "removal": rem, "history_xa2": hist_xa2, "v80_a1_odd": v80_a1, "v81_a2_ext_odd": v81_a2,
                   "block_cos_xa2_xdas": g.block_cosines(qd["xa2"], qd["xdas"])}
        print(name, {arm: {t: round(rem[arm][t]["ce_damage"], 3) for t in ("A1", "A2", "C", "unseen")} for arm in rem},
              {fam: {k: round(v["point"], 3) for k, v in e.items()} for fam, e in ext.items()}, round(time.perf_counter() - t0), "s", flush=True)

    e = lambda n, fam, k: R[n]["extraction"][fam][k]
    rm = lambda n, arm, t: R[n]["removal"][arm][t]
    n = NAME
    xa, xd = (lambda t: rm(n, "xa2", t)), (lambda t: rm(n, "xdas", t))
    predictions = {
        'pred_a_reproduce': abs(xd("A1")["ce_damage"] - R[n]["v80_a1_odd"]) <= REPRO_REM and abs(e(n, "A2", "xdas")["point"] - R[n]["v81_a2_ext_odd"]) <= REPRO_EXT,
        'pred_b_a2_gain': e(n, "A2", "xa2")["point"] >= A2_EXT and e(n, "A2", "xa2")["lb95"] >= A2_EXT_LB,
        'pred_c_a1_kept': xa("A1")["ce_damage"] >= KEEP * xd("A1")["ce_damage"] and e(n, "A1", "xa2")["point"] >= e(n, "A1", "xdas")["point"] - A1_EXT_SLACK,
        'pred_d_specific': xa("C")["ce_ub975"] <= xd("C")["ce_ub975"] + C_UB_SLACK and max(abs(v) for v in xa("cross").values()) <= CROSS_MAX,
        'pred_e_transfer': xa("unseen")["ce_damage"] >= UNSEEN_KEEP * xd("unseen")["ce_damage"] and xa("A2")["ce_damage"] >= A2_TRANSFER * xa("A1")["ce_damage"] and xa("A2")["ce_lb975"] > 0
                           and e(n, "A1", "random")["point"] <= RAND_FRAC * e(n, "A1", "xa2")["point"],
    }
    summary = {fam: {k: (round(v["point"], 3), round(v["lb95"], 3)) for k, v in R[n]["extraction"][fam].items()} for fam in ("A1", "A2")}
    removal_summary = {arm: {t: (round(rm(n, arm, t)["ce_damage"], 3), round(rm(n, arm, t)["ce_ub975"], 3)) for t in ("A1", "A2", "C", "unseen")} | {"cross_abs_max": round(max(abs(v) for v in rm(n, arm, "cross").values()), 3)} for arm in ("xdas", "xa2")}
    mean_abs_cos = sum(abs(v) for v in R[n]["block_cos_xa2_xdas"].values()) / len(R[n]["block_cos_xa2_xdas"])
    result = {"predictions": predictions, "schema": "circuit_unit_dative_a2_pool_result_v1", "candidate_id": "corpus.unit_dative_a2_pool_v89",
              "summary": summary, "removal_summary": removal_summary, "mean_abs_cos_xa2_xdas": mean_abs_cos,
              "bars": {"repro_rem": REPRO_REM, "repro_ext": REPRO_EXT, "a2_ext": A2_EXT, "a2_ext_lb": A2_EXT_LB, "keep": KEEP, "a1_ext_slack": A1_EXT_SLACK, "c_ub_slack": C_UB_SLACK,
                       "cross_max": CROSS_MAX, "unseen_keep": UNSEEN_KEEP, "a2_transfer": A2_TRANSFER, "rand_frac": RAND_FRAC,
                       "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW, "lambda": LAM, "xa2_pool": "A1 even + verb variants even + A2 even"}},
              "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "removal": removal_summary, "cos": round(mean_abs_cos, 3), "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
