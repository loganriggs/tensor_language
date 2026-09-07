"""v63: v61 replicated on verb_complementizer -- is cue-pair keying of the removal direction general?

v57: verb_complementizer {06:03, 11:03, 07:08} removes 0.583 nat on A1 but 0.311 on A2 (0.53x), the same weakness dative showed
(v61: the direction, not the set). Same design, same registered bars; a second behaviour decides whether the keying is general.

Terminal table: dative's rank-1 block diff-in-means direction (fit on A1) removes 0.259 nat on A1 but only 0.144 on A2
(0.56x; enlarged set 0.45x), the weakest row 5 of the six sets; the other sets transfer at 0.76-1.6x. Two hypotheses.
  H_direction: the A1 direction is cue-pair keyed (v13/v14 for interchange axes): a direction fit on A2 removes A2 as
               strongly as A1's removes A1, and the two directions are not parallel.
  H_set:       the five heads carry the A2 pair weakly however the direction is fit: A2-fit on A2 is itself weak.
Design (original set {14:08, 07:08, 06:03, 13:08, 11:03}): FIT on even rows of A1 and of A2 (and pooled), EVALUATE on the
odd rows (v51.removal, background from the fit rows). Arms: A1->A1, A2->A2, A1->A2, A2->A1, pool->A1, pool->A2, random.

REGISTERED BEFORE THE RUN
    pred_a_A2_self_strong   A2-fit on odd A2 >= 0.70 x A1-fit on odd A1 (H_direction). Worked: 0.20 vs 0.26 True; 0.12 vs 0.26 False (H_set).
    pred_b_cross_weak       A1-fit on odd A2 <= 0.70 x A2-fit on odd A2 (the direction does not transfer). Worked: 0.12 vs 0.20 True.
    pred_c_not_parallel     mean per-block |cos(q_A1, q_A2)| <= 0.80. Worked: 0.55 True; 0.92 False.
    pred_d_pooled_covers    pooled direction on odd A1 >= 0.80 x A1->A1 AND on odd A2 >= 0.80 x A2->A2 (one direction can serve both
                            if fit on both). Worked: 0.23/0.26, 0.18/0.20 True.
    pred_e_random           random rank-1 <= 0.25 x own on both odd halves. Worked: 0.003 True.
    Reading rule. a,b,c True: the removal direction is cue-pair keyed for dative -- row 5 as registered (A1-fit direction) stays
    unmet; d True offers a pooled direction as the honest replacement, to be re-registered on fresh rows, not swapped in here.
    a False: the set itself carries the A2 pair weakly -- a set question, not a direction question; no head is added here.
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
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_complementizer_ood_direction_v63_result.json"
NAME = "verb_complementizer"
SELF_FRAC, CROSS_FRAC, COS_MAX, POOL_FRAC, RAND_FRAC = 0.70, 0.70, 0.80, 0.80, 0.25
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 60, 1500


def _plan():
    return {"candidate_id": "corpus.unit_complementizer_ood_direction_v63", "set": v15.SETS[NAME][:2][1],
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    module, units = v15.SETS[NAME][:2]
    units = list(units)
    rows = {fam: g.rows_of(module, fam) for fam in ("A1", "A2")}
    fit = {fam: g.prepare(backend, r[0::2]) for fam, r in rows.items()}
    odd = {fam: g.prepare(backend, r[1::2]) for fam, r in rows.items()}
    fit["pool"] = g.prepare(backend, rows["A1"][0::2] + rows["A2"][0::2])

    def mu_of(prep):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}
    q = {k: g.block_diff_in_means(backend, p, units) for k, p in fit.items()}
    mu = {k: mu_of(p) for k, p in fit.items()}
    q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
    arms = {}
    for src in ("A1", "A2", "pool"):
        for tgt in ("A1", "A2"):
            arms[f"{src}->{tgt}"] = v51.summary(torch, v51.removal(backend, odd[tgt], units, q[src], mu[src]))
    for tgt in ("A1", "A2"):
        arms[f"random->{tgt}"] = v51.summary(torch, v51.removal(backend, odd[tgt], units, q_rand, mu[tgt]))
    cos = {f"{k[0]}:{k[1]}": abs(float((q["A1"][k][:, 0] * q["A2"][k][:, 0]).sum())) for k in q["A1"]}
    mean_cos = sum(cos.values()) / len(cos)
    d = lambda k: arms[k]["ce_damage"]
    predictions = {
        'pred_a_A2_self_strong': d("A2->A2") >= SELF_FRAC * d("A1->A1"),
        'pred_b_cross_weak': d("A1->A2") <= CROSS_FRAC * d("A2->A2"),
        'pred_c_not_parallel': mean_cos <= COS_MAX,
        'pred_d_pooled_covers': d("pool->A1") >= POOL_FRAC * d("A1->A1") and d("pool->A2") >= POOL_FRAC * d("A2->A2"),
        'pred_e_random': d("random->A1") <= RAND_FRAC * d("A1->A1") and d("random->A2") <= RAND_FRAC * d("A2->A2"),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_complementizer_ood_direction_result_v1", "candidate_id": "corpus.unit_complementizer_ood_direction_v63",
              "set": units, "rows_fit": {k: len(p.rows) for k, p in fit.items()}, "rows_odd": {k: len(p.rows) for k, p in odd.items()},
              "bars": {"self_frac": SELF_FRAC, "cross_frac": CROSS_FRAC, "cos_max": COS_MAX, "pool_frac": POOL_FRAC, "rand_frac": RAND_FRAC},
              "arms": arms, "block_abs_cos_A1_A2": cos, "mean_abs_cos": mean_cos,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "damage": {k: round(v_["ce_damage"], 3) for k, v_ in arms.items()},
                      "lb": {k: round(v_["ce_lb975"], 3) for k, v_ in arms.items()}, "cos": {k: round(v_, 2) for k, v_ in cos.items()},
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
