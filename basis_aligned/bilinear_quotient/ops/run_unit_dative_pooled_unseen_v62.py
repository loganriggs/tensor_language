"""v62: does a pooled removal direction for dative transfer to a cue pair it has never seen?

v61: dative's rank-1 removal direction is cue-pair keyed (A1-fit removes A2 at 0.45x of A2's own; block cosines 0.33-0.62),
and a direction pooled over A1+A2 exceeds each single-pair fit on held-out halves. The honest test is an UNSEEN pair:
v15's fourth map {sent -> passed, reserved -> cooked} applied to the A1 rows (g.lexical_variant; never used for fitting).
FIT (frozen): q_pool from A1 + A2 + the two v15 variants {handed/bought}, {gave/kept}; q_A1 from A1 alone; q_ceiling
from the EVEN rows of the fourth-map variant. EVALUATE on the ODD rows of the fourth-map variant (v51.removal,
background from the fitting rows). Set = original dative {14:08, 07:08, 06:03, 13:08, 11:03}.

REGISTERED BEFORE THE RUN
    pred_a_pooled_transfers   pooled on odd-fourth >= 0.80 x ceiling (fourth-even-fit on odd-fourth). Worked: 0.30 vs 0.35 True; 0.20 False.
    pred_b_single_pair_weak   A1-fit on odd-fourth <= 0.70 x ceiling. Worked: 0.15 vs 0.35 True; 0.30 False.
    pred_c_ceiling_real       ceiling LB > 0.05 nat (the set carries the unseen pair). Worked: LB 0.25 True.
    pred_d_pooled_beats_A1    pooled >= 1.3 x A1-fit on odd-fourth. Worked: 0.30 vs 0.15 True; 0.17 vs 0.15 False.
    pred_e_controls           pooled direction: C answer-CE UB <= 0.01 nat AND random rank-1 <= 0.25 x ceiling. Worked: -0.02, 0.003 True.
    Reading rule. a,b,d True: a single rank-1 direction per block serves the dative behaviour across four cue pairs once fit
    on more than one -- the pooled direction replaces the A1-fit direction in dative's row-5 entry (re-registered here on
    unseen rows). a False with c True: pooling two pairs is not enough; the direction is per-pair (report the cosine of
    q_pool to q_ceiling; do not fit on the fourth pair).
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
OUT = ROOT / "circuits/followups/unit_dative_pooled_unseen_v62_result.json"
NAME = "dative"
POOL_FRAC, SINGLE_MAX, CEIL_LB, BEAT, C_UB, RAND_FRAC = 0.80, 0.70, 0.05, 1.3, 0.01, 0.25
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 60, 2000


def _plan():
    return {"candidate_id": "corpus.unit_dative_pooled_unseen_v62", "set": v23.SETS[NAME][1], "fourth_map": v15.SETS[NAME][3],
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    module, units = v23.SETS[NAME]
    units = list(units)
    _, _, maps, fourth_map = v15.SETS[NAME]
    a1, a2 = g.rows_of(module, "A1"), g.rows_of(module, "A2")
    variants = [g.lexical_variant(a1, mp) for mp in maps]
    fourth = g.lexical_variant(a1, fourth_map)
    fit = {"A1": g.prepare(backend, a1), "pool": g.prepare(backend, a1 + a2 + variants[0] + variants[1]),
           "ceiling": g.prepare(backend, fourth[0::2])}
    odd = g.prepare(backend, fourth[1::2])
    c_prep = g.prepare(backend, g.rows_of(v23.SETS["polarity_licensing"][0], "C"))

    def mu_of(prep):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}
    q = {k: g.block_diff_in_means(backend, p, units) for k, p in fit.items()}
    mu = {k: mu_of(p) for k, p in fit.items()}
    q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
    arms = {k: v51.summary(torch, v51.removal(backend, odd, units, q[k], mu[k])) for k in fit}
    arms["random"] = v51.summary(torch, v51.removal(backend, odd, units, q_rand, mu["ceiling"]))
    arms["pool_on_C"] = v51.summary(torch, v51.removal(backend, c_prep, units, q["pool"], mu["pool"]))
    cos = {f"{k[0]}:{k[1]}": {"pool_vs_ceiling": abs(float((q["pool"][k][:, 0] * q["ceiling"][k][:, 0]).sum())),
                              "A1_vs_ceiling": abs(float((q["A1"][k][:, 0] * q["ceiling"][k][:, 0]).sum()))} for k in q["A1"]}
    d = lambda k: arms[k]["ce_damage"]
    predictions = {
        'pred_a_pooled_transfers': d("pool") >= POOL_FRAC * d("ceiling"),
        'pred_b_single_pair_weak': d("A1") <= SINGLE_MAX * d("ceiling"),
        'pred_c_ceiling_real': arms["ceiling"]["ce_lb975"] > CEIL_LB,
        'pred_d_pooled_beats_A1': d("pool") >= BEAT * d("A1"),
        'pred_e_controls': arms["pool_on_C"]["ce_ub975"] <= C_UB and d("random") <= RAND_FRAC * d("ceiling"),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_dative_pooled_unseen_result_v1", "candidate_id": "corpus.unit_dative_pooled_unseen_v62",
              "set": units, "fourth_map": fourth_map, "fit_rows": {k: len(p.rows) for k, p in fit.items()}, "eval_rows": len(odd.rows),
              "bars": {"pool_frac": POOL_FRAC, "single_max": SINGLE_MAX, "ceil_lb": CEIL_LB, "beat": BEAT, "c_ub": C_UB, "rand_frac": RAND_FRAC},
              "arms": arms, "block_abs_cos": cos, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "damage": {k: round(v_["ce_damage"], 3) for k, v_ in arms.items()},
                      "lb": {k: round(v_["ce_lb975"], 3) for k, v_ in arms.items()}, "cos": {k: {a: round(b, 2) for a, b in v_.items()} for k, v_ in cos.items()},
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
