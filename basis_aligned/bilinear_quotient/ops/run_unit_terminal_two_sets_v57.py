"""v57: terminal-evidence battery for the two greedy head sets not yet on the tier list -- verb_preposition
(attn 06:03, 13:08, 08:08) and verb_complementizer (06:03, 11:03, 07:08), v15.SETS.

Same instruments as v51 (selective removal: block diff-in-means direction set to its background coordinate at the
prediction position; CE damage with 97.5% document bootstrap; negative C; random rank-1; OOD A2; cross-collateral on
the other sets' A1 sentences at their answer position) and v52 (extraction: all 162 heads mean-ablated at the prediction
position except the exact set; ratio-of-means bootstrap; keep_dim; random-set control; interchange recovery).
Cross-collateral runs against ALL five other sets (the four of v23 and the sibling of the pair). Both new sets share
06:03 with dative and 11:03 / 07:08 hubs with the number family, so collateral on dative and quantifier is the
sceptical arm.

REGISTERED BEFORE THE RUN (priors from the four v51/v52 sets: removal 0.10-0.36 nat, C ~0, random <=0.003, A2 0.14-0.40,
extraction 0.51-0.60 tracking interchange 0.57-0.65)
    pred_a_removal         own A1 CE damage LB > 0.05 nat and specificity (target - C) LB > 0 for both sets. Worked: 0.17 LB 0.10, C -0.01 True; LB 0.02 False.
    pred_b_ood             A2 CE damage LB > 0 and point >= 0.50 x A1 point for both sets. Worked: 0.29 vs 0.17 True; 0.05 vs 0.17 False.
    pred_c_random          random rank-1 direction A1 damage <= 0.25 x own damage for both sets. Worked: 0.003 vs 0.17 True.
    pred_d_cross           cross-collateral CE (one-sided) <= 0.25 x own damage on all 10 pairs. Worked: 0.02 vs 0.17 True; 0.06 vs 0.17 False.
    pred_e_extraction      keep_exact within 0.10 of the interchange recovery for both sets AND below the 0.80 rubric bar
                           (stated prior: extraction tracks interchange; the bar is not met by heads-only sets). Worked: 0.55 vs 0.60 True; 0.85 False.
    Reading rule. a/b True: rows 3 and 5 met -> the set enters the tier list at the same tier as dative/quantifier. d False on
    dative or quantifier: the shared hub heads carry both behaviours' directions -- report the pair, do not shrink the set.
    e False by exceeding 0.80: row 2 met for a heads-only set for the first time -- report, and re-check with the v54 held-out split.
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
OUT = ROOT / "circuits/followups/unit_terminal_two_sets_v57_result.json"
NEW = {k: (v15.SETS[k][0], list(v15.SETS[k][1])) for k in ("verb_preposition", "verb_complementizer")}
OTHERS = {k: (v[0], list(v[1])) for k, v in v23.SETS.items()}
DAMAGE_MIN, OOD_FRAC, RAND_FRAC, CROSS_FRAC, TRACK_TOL, ROW2 = 0.05, 0.50, 0.25, 0.25, 0.10, 0.80
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 120, 4000


def _plan():
    return {"candidate_id": "corpus.unit_terminal_two_sets_v57", "sets": {k: v[1] for k, v in NEW.items()},
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    all_heads = g.all_head_units()
    a1 = {n: g.prepare(backend, g.rows_of(m, "A1")) for n, (m, _) in {**NEW, **OTHERS}.items()}
    c_prep = g.prepare(backend, g.rows_of(OTHERS["polarity_licensing"][0], "C"))
    report = {}
    for n, (m, units) in NEW.items():
        prep = a1[n]
        a2 = g.prepare(backend, g.rows_of(m, "A2"))
        q = g.block_diff_in_means(backend, prep, units)
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}
        tgt, neg = v51.removal(backend, prep, units, q, mu), v51.removal(backend, c_prep, units, q, mu)
        spec = [x - y for x, y in zip(tgt["ce"], neg["ce"] + neg["ce"])] if len(neg["ce"]) * 2 == len(tgt["ce"]) else None
        r = {"units": units, "target_A1": v51.summary(torch, tgt), "negative_C": v51.summary(torch, neg),
             "random_subspace_A1": v51.summary(torch, v51.removal(backend, prep, units, q_rand, mu)),
             "ood_A2": v51.summary(torch, v51.removal(backend, a2, units, q, mu)), "cross": {}}
        r["specificity_target_minus_C"] = {"point": r["target_A1"]["ce_damage"] - r["negative_C"]["ce_damage"],
                                           "lb975": r["target_A1"]["ce_lb975"] - r["negative_C"]["ce_ub975"]}
        for t in {**NEW, **OTHERS}:
            if t != n:
                r["cross"][t] = v51.summary(torch, v51.removal(backend, a1[t], units, q, mu))
        # extraction (v52): all heads mean-ablated except the exact set / its direction; random set of equal size
        mu_all = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in all_heads}
        gen = torch.Generator().manual_seed(1)
        pool = [u for u in all_heads if u not in units]
        rand_set = [pool[i] for i in torch.randperm(len(pool), generator=gen)[:len(units)].tolist()]

        def margins(set_units, qs):
            order = v52.ordered_units(set_units)
            bq = v52.block_q(torch, backend.device, set_units, qs)
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
        nat, m_null = margins(units, None)
        den = [a - b for a, b in zip(nat, m_null)]
        ext = {}
        for name, (su, qs) in {"keep_exact": (units, g.block_identity(backend, units)), "keep_dim": (units, q), "keep_rand": (units, q_rand),
                               "keep_set_r": (rand_set, g.block_identity(backend, rand_set))}.items():
            _, m_arm = margins(su, qs)
            ext[name] = v52._boot_ratio(torch, [x - y for x, y in zip(m_arm, m_null)], den)
        r["extraction_A1"] = ext
        r["interchange_recovery_A1"] = g.recovery(prep, g.patched_axis(backend, prep, units))
        report[n] = r
        print(n, json.dumps({"target": round(r["target_A1"]["ce_damage"], 3), "lb": round(r["target_A1"]["ce_lb975"], 3), "C": round(r["negative_C"]["ce_damage"], 3),
                             "rand": round(r["random_subspace_A1"]["ce_damage"], 3), "A2": round(r["ood_A2"]["ce_damage"], 3),
                             "cross": {t: round(c["ce_damage"], 3) for t, c in r["cross"].items()},
                             "extract": round(ext["keep_exact"]["point"], 3), "dim": round(ext["keep_dim"]["point"], 3), "set_r": round(ext["keep_set_r"]["point"], 3),
                             "interchange": round(r["interchange_recovery_A1"], 3)}), flush=True)
    R = report.values()
    predictions = {
        'pred_a_removal': all(r["target_A1"]["ce_lb975"] > DAMAGE_MIN and r["specificity_target_minus_C"]["lb975"] > 0 for r in R),
        'pred_b_ood': all(r["ood_A2"]["ce_lb975"] > 0 and r["ood_A2"]["ce_damage"] >= OOD_FRAC * r["target_A1"]["ce_damage"] for r in R),
        'pred_c_random': all(r["random_subspace_A1"]["ce_damage"] <= RAND_FRAC * r["target_A1"]["ce_damage"] for r in R),
        'pred_d_cross': all(c["ce_damage"] <= CROSS_FRAC * r["target_A1"]["ce_damage"] for r in R for c in r["cross"].values()),
        'pred_e_extraction': all(abs(r["extraction_A1"]["keep_exact"]["point"] - r["interchange_recovery_A1"]) <= TRACK_TOL and r["extraction_A1"]["keep_exact"]["point"] < ROW2 for r in R),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_terminal_two_sets_result_v1", "candidate_id": "corpus.unit_terminal_two_sets_v57",
              "bars": {"damage_min": DAMAGE_MIN, "ood_frac": OOD_FRAC, "rand_frac": RAND_FRAC, "cross_frac": CROSS_FRAC, "track_tol": TRACK_TOL, "row2": ROW2},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
