#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets read from the v66/v67 receipts, rubric bars fixed before the run.
"""v69: rows 3-5 of the tier rubric for the six hub+8 removal-greedy sets (v66/v67), on rows the greedy never saw.

v68 showed every hub+8 set meets row 2 (extraction 0.81-1.00). Rows 3-5 need: selective removal with specificity
against the behaviour's OWN C family (row 3), off-target CE with the rubric's UB <= 0.01 nat (row 4), OOD transfer of
the A1-fit direction to the A2 cue pair at >= 50% with LB > 0 (row 5). All directions are diff-in-means fit on A1 EVEN
rows; every evaluation is on ODD rows (A1, A2, C: 16 documents each, x base+donor). Random rank-1 direction (seed 1),
cross-collateral on the other five A1 odd families, and -- the v61 question at hub+8 -- an A2-even-fit direction on
A2 odd, to see whether enlarging the set removes dative's cue-pair keying (v61: A2-fit 3x A1-fit on A2 for the hub).

REGISTERED BEFORE THE RUN (CE damage in nat, 97.5% document bootstrap)
    pred_a_row3      for all six: A1 odd removal LB > 0 and specificity (A1 damage - C damage) LB > 0.
                     Worked: 0.72 (LB 0.57), C 0.02 (UB 0.05) True; C 0.30 (UB 0.60) with A1 0.72 (LB 0.57) False.
    pred_b_row4_C    for all six: own-C CE damage UB <= 0.01 (the rubric bar as written). Worked: UB 0.008 True; UB 0.03 False.
    pred_c_row5_A2   for all six: A1-fit direction on A2 odd has LB > 0 and >= 0.50 x A1 odd damage.
                     Worked: A2 0.40 (LB 0.25) vs A1 0.72 True; A2 0.30 False.
    pred_d_random    for all six: random rank-1 removes <= 0.05 x the A1-fit direction on A1 odd. Worked: 0.006 vs 0.72 True; 0.05 False.
    pred_e_no_keying for all six: A2-even-fit direction on A2 odd <= 1.5 x the A1-fit direction on A2 odd.
                     Worked: 0.45 vs 0.40 True; 0.60 vs 0.30 False.
    Prior: a True, d True, c True for five (dative unsure), e unsure (the question), b unsure -- v59's enlarged voice
    had C UB 0.064, and 16-document halves give wide bounds; a False here is the honest reading of the rubric bar.
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
OUT = ROOT / "circuits/followups/unit_greedy_sets_terminal_rows_v69_result.json"
SRC = [ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json",
       ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"]
C_UB, OOD_FRAC, RAND_FRAC, KEY_MAX, CROSS_FRAC = 0.01, 0.50, 0.05, 1.5, 0.10
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 8000


def _plan():
    return {"candidate_id": "corpus.unit_greedy_sets_terminal_rows_v69", "sources": [str(p) for p in SRC],
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    sets = {n: r["final"] for p in SRC for n, r in json.loads(p.read_text())["sets"].items()}
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}

    def mu_of(prep, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    preps = {}
    for n in sets:
        for fam in ("A1", "A2", "C"):
            rows = g.rows_of(modules[n], fam)
            preps[(n, fam, "even")] = g.prepare(backend, rows[0::2])
            preps[(n, fam, "odd")] = g.prepare(backend, rows[1::2])
    R = {}
    for n, units in sets.items():
        fit = preps[(n, "A1", "even")]
        q, mu = g.block_diff_in_means(backend, fit, units), mu_of(fit, units)
        fit2 = preps[(n, "A2", "even")]
        q2, mu2 = g.block_diff_in_means(backend, fit2, units), mu_of(fit2, units)
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        rem = lambda key, qq, mm: v51.removal(backend, preps[key], units, qq, mm)
        a1, c = rem((n, "A1", "odd"), q, mu), rem((n, "C", "odd"), q, mu)
        r = {"units": units, "A1": v51.summary(torch, a1), "C": v51.summary(torch, c),
             "A2_a1fit": v51.summary(torch, rem((n, "A2", "odd"), q, mu)),
             "A2_a2fit": v51.summary(torch, rem((n, "A2", "odd"), q2, mu2)),
             "random_A1": v51.summary(torch, rem((n, "A1", "odd"), q_rand, mu)),
             "cross": {t: v51.summary(torch, v51.removal(backend, preps[(t, "A1", "odd")], units, q, mu))["ce_damage"] for t in sets if t != n}}
        # A1 and C are different documents: conservative unpaired bound, LB(A1) - UB(C)
        r["specificity"] = {"point": r["A1"]["ce_damage"] - r["C"]["ce_damage"], "lb975": r["A1"]["ce_lb975"] - r["C"]["ce_ub975"]}
        R[n] = r
        print(n, json.dumps({"A1": round(r["A1"]["ce_damage"], 3), "lb": round(r["A1"]["ce_lb975"], 3), "C": round(r["C"]["ce_damage"], 3), "C_ub": round(r["C"]["ce_ub975"], 3),
                             "A2": round(r["A2_a1fit"]["ce_damage"], 3), "A2fit": round(r["A2_a2fit"]["ce_damage"], 3), "rand": round(r["random_A1"]["ce_damage"], 3),
                             "spec_lb": round(r["specificity"]["lb975"], 3), "cross_max": round(max(r["cross"].values()), 3)}))

    d = lambda n, k: R[n][k]["ce_damage"]
    predictions = {
        'pred_a_row3': all(R[n]["A1"]["ce_lb975"] > 0 and R[n]["specificity"]["lb975"] > 0 for n in R),
        'pred_b_row4_C': all(R[n]["C"]["ce_ub975"] <= C_UB for n in R),
        'pred_c_row5_A2': all(R[n]["A2_a1fit"]["ce_lb975"] > 0 and d(n, "A2_a1fit") >= OOD_FRAC * d(n, "A1") for n in R),
        'pred_d_random': all(d(n, "random_A1") <= RAND_FRAC * d(n, "A1") for n in R),
        'pred_e_no_keying': all(d(n, "A2_a2fit") <= KEY_MAX * d(n, "A2_a1fit") for n in R),
    }
    rows_met = {n: {"row3": R[n]["A1"]["ce_lb975"] > 0 and R[n]["specificity"]["lb975"] > 0, "row4": R[n]["C"]["ce_ub975"] <= C_UB,
                    "row5": R[n]["A2_a1fit"]["ce_lb975"] > 0 and d(n, "A2_a1fit") >= OOD_FRAC * d(n, "A1"),
                    "cross_clean": all(x <= CROSS_FRAC * d(n, "A1") for x in R[n]["cross"].values())} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_greedy_sets_terminal_rows_result_v1", "candidate_id": "corpus.unit_greedy_sets_terminal_rows_v69",
              "bars": {"c_ub": C_UB, "ood_frac": OOD_FRAC, "rand_frac": RAND_FRAC, "key_max": KEY_MAX, "cross_frac": CROSS_FRAC},
              "rows_met": rows_met, "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "rows_met": rows_met, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
