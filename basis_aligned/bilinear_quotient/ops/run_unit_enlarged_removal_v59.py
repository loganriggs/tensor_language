"""v59: rows 3-5 for the ENLARGED head sets of v54 -- does the shared attn:09:head:07 leak across behaviours?

v54 added 3-4 heads to each of the four sets under the extraction objective (held-out 0.72-0.78) and attn:09:head:07
entered ALL four; attn:08:head:01, 13:01, 05:03, 06:03, 00:03 recur in two or three. v51 measured selective removal and
12-pair cross-collateral for the ORIGINAL sets only (all pairs under a one-sided 0.25 bar). A direction that now runs
through heads shared by every behaviour is the first place a set enlargement could buy extraction with specificity.
Instrument = v51.removal (block diff-in-means, rank 1 per block, background = per-unit A1 mean at the prediction
position; CE damage with 97.5% document bootstrap); arms own A1, C, random rank-1, A2, and the other three sets' A1.
Original-set numbers (v51) are read from disk for the ratios.

REGISTERED BEFORE THE RUN
    pred_a_more_damage    enlarged-set own A1 CE damage >= 1.2 x the v51 original-set damage on all four sets (the added heads
                          carry more of the behaviour). Worked: 0.45 vs 0.355 True; 0.37 vs 0.355 False.
    pred_b_C_inert        C answer-CE shift (one-sided) UB <= 0.01 nat on all four. Worked: -0.02 True; 0.03 False.
    pred_c_no_leak        cross-collateral CE (one-sided point) <= 0.25 x own damage on all 12 pairs (09:07 does not leak).
                          Worked: 0.04 vs 0.45 True; 0.15 vs 0.45 False.
    pred_d_ood            A2 damage LB > 0 and point >= 0.50 x own on all four. Worked: 0.35 vs 0.45 True.
    pred_e_random         random rank-1 direction <= 0.25 x own on all four. Worked: 0.003 True.
    Reading rule. a True, c True: enlargement bought terminal strength without specificity loss -- the enlarged sets replace the
    originals on the tier list. c False: name the leaking pair; the shared hub head is where the two behaviours' directions
    overlap -- report, do not remove the head (a removal is a new hypothesis). a False: the added heads improve extraction
    (sufficiency in isolation) without adding necessity -- the two rubric rows measure different things; report both.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier2_characterization_v23 as v23
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_enlarged_removal_v59_result.json"
V54 = ROOT / "circuits/followups/unit_extraction_greedy_v54_result.json"
V51 = ROOT / "circuits/followups/unit_selective_removal_four_sets_v51_result.json"
GAIN_MIN, C_UB, CROSS_FRAC, OOD_FRAC, RAND_FRAC = 1.2, 0.01, 0.25, 0.50, 0.25
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 100, 3500


def _plan():
    return {"candidate_id": "corpus.unit_enlarged_removal_v59", "sets_from": str(V54.relative_to(ROOT)),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    enlarged = {n: list(r["units_final"]) for n, r in json.loads(V54.read_text())["behaviours"].items()}
    orig = json.loads(V51.read_text())["behaviours"]
    preps = {n: {fam: g.prepare(backend, g.rows_of(m, fam)) for fam in ("A1", "A2")} for n, (m, _) in v23.SETS.items()}
    c_prep = g.prepare(backend, g.rows_of(v23.SETS["polarity_licensing"][0], "C"))
    report = {}
    for n in v23.SETS:
        units = enlarged[n]
        a1 = preps[n]["A1"]
        q = g.block_diff_in_means(backend, a1, units)
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (a1.base_cache, a1.donor_cache) for rid in a1.base_batch.row_ids]).mean(0) for u in units}
        r = {"units": units, "added": [u for u in units if u not in v23.SETS[n][1]], "original_target_ce": orig[n]["target_A1"]["ce_damage"],
             "target_A1": v51.summary(torch, v51.removal(backend, a1, units, q, mu)),
             "negative_C": v51.summary(torch, v51.removal(backend, c_prep, units, q, mu)),
             "random_subspace_A1": v51.summary(torch, v51.removal(backend, a1, units, q_rand, mu)),
             "ood_A2": v51.summary(torch, v51.removal(backend, preps[n]["A2"], units, q, mu)), "cross": {}}
        for t in v23.SETS:
            if t != n:
                r["cross"][t] = v51.summary(torch, v51.removal(backend, preps[t]["A1"], units, q, mu))
        r["gain_over_original"] = r["target_A1"]["ce_damage"] / r["original_target_ce"]
        report[n] = r
        print(n, json.dumps({"target": round(r["target_A1"]["ce_damage"], 3), "lb": round(r["target_A1"]["ce_lb975"], 3), "orig": round(r["original_target_ce"], 3),
                             "gain": round(r["gain_over_original"], 2), "C_ub": round(r["negative_C"]["ce_ub975"], 3), "rand": round(r["random_subspace_A1"]["ce_damage"], 3),
                             "A2": round(r["ood_A2"]["ce_damage"], 3), "cross": {t: round(c["ce_damage"], 3) for t, c in r["cross"].items()}}), flush=True)
    R = report.values()
    predictions = {
        'pred_a_more_damage': all(r["gain_over_original"] >= GAIN_MIN for r in R),
        'pred_b_C_inert': all(r["negative_C"]["ce_ub975"] <= C_UB for r in R),
        'pred_c_no_leak': all(c["ce_damage"] <= CROSS_FRAC * r["target_A1"]["ce_damage"] for r in R for c in r["cross"].values()),
        'pred_d_ood': all(r["ood_A2"]["ce_lb975"] > 0 and r["ood_A2"]["ce_damage"] >= OOD_FRAC * r["target_A1"]["ce_damage"] for r in R),
        'pred_e_random': all(r["random_subspace_A1"]["ce_damage"] <= RAND_FRAC * r["target_A1"]["ce_damage"] for r in R),
    }
    leaks = [(n, t, c["ce_damage"]) for n, r in report.items() for t, c in r["cross"].items() if c["ce_damage"] > CROSS_FRAC * r["target_A1"]["ce_damage"]]
    result = {"predictions": predictions, "schema": "circuit_unit_enlarged_removal_result_v1", "candidate_id": "corpus.unit_enlarged_removal_v59",
              "bars": {"gain_min": GAIN_MIN, "c_ub": C_UB, "cross_frac": CROSS_FRAC, "ood_frac": OOD_FRAC, "rand_frac": RAND_FRAC},
              "leaking_pairs": leaks, "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "leaking_pairs": leaks, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
