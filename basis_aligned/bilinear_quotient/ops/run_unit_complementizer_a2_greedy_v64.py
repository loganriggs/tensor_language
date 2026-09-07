#!/usr/bin/env python3
# BQGATE: five frozen predictions; set, families, bars and greedy budget fixed before the run.
"""v64: which heads carry verb_complementizer's second cue pair? Removal-greedy extension of the hub set on A2.

v57/v63: the hub set {06:03, 11:03, 07:08} removes 0.598 nat on A1 (wondered/remarked) but only 0.353 on A2, and
v63 showed the direction is the same on both pairs (|cos| 0.97-0.99; A2-fit gains 0.03). So A2's answer routes partly
through heads outside the set. Greedy forward selection over the other 159 heads, scored by v51 removal damage on the
A2 EVEN rows (diff-in-means direction refit on A2 even rows for each candidate set), at most 3 additions; every
selected set is then evaluated on the A2 ODD rows (never used for selection), on A1 odd rows (A1-even-fit direction),
on cross-collateral (A1 rows of the four v23 sets), and against a random rank-1 direction. Selection stops when the
even-row gain of the best candidate is below GAIN_MIN.

REGISTERED BEFORE THE RUN (A2 odd-row removal damage in nat unless stated)
    pred_a_first_gain      the best single addition raises A2 odd removal by >= 0.08 over the hub set.
                           Worked: 0.353 -> 0.46 True; 0.353 -> 0.39 False.
    pred_b_reaches_A1      the final extended set's A2 odd removal >= 0.80 x the hub set's A1 odd removal (0.598 -> 0.478).
                           Worked: 0.50 True; 0.44 False.
    pred_c_cross_clean     the final set's cross-collateral CE damage (A2-fit direction) on every v23 A1 family
                           <= 0.10 x its A2 odd removal. Worked: 0.02 vs 0.45 True; 0.06 vs 0.45 False.
    pred_d_random          random rank-1 on the final set removes <= 0.25 x the A2-fit direction on A2 odd.
                           Worked: 0.01 vs 0.45 True; 0.15 vs 0.45 False.
    pred_e_selection_holds the total even-row gain of the greedy transfers to odd rows at >= 0.50x.
                           Worked: even +0.15, odd +0.10 True; even +0.15, odd +0.04 False.
    Prior: a True, e True (the pair is homogeneous; 40+ rows per half), b unsure (the A2 route may be spread over many
    weak heads), c unsure (a head that carries A2 may also carry quantifier), d True.
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
OUT = ROOT / "circuits/followups/unit_complementizer_a2_greedy_v64_result.json"
NAME = "verb_complementizer"
FIRST_GAIN, A1_FRAC, CROSS_FRAC, RAND_FRAC, HOLD_FRAC = 0.08, 0.80, 0.10, 0.25, 0.50
MAX_ADD, GAIN_MIN = 3, 0.02
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 1100, 60000


def _plan():
    return {"candidate_id": "corpus.unit_complementizer_a2_greedy_v64", "set": v15.SETS[NAME][1], "max_add": MAX_ADD,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    module, hub = v15.SETS[NAME][:2]
    hub = list(hub)
    rows = {fam: g.rows_of(module, fam) for fam in ("A1", "A2")}
    even = {fam: g.prepare(backend, r[0::2]) for fam, r in rows.items()}
    odd = {fam: g.prepare(backend, r[1::2]) for fam, r in rows.items()}
    cross = {k: g.prepare(backend, g.rows_of(m, "A1")) for k, (m, _) in v23.SETS.items()}

    def mu_of(prep, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    def damage(fit, ev, units):
        q = g.block_diff_in_means(backend, fit, units)
        return v51.summary(torch, v51.removal(backend, ev, units, q, mu_of(fit, units)))

    # greedy on A2 even rows
    chosen, path = list(hub), []
    cur_even = damage(even["A2"], even["A2"], chosen)["ce_damage"]
    base_even = cur_even
    pool = [u for u in g.all_head_units() if u not in hub]
    for step in range(MAX_ADD):
        scores = {u: damage(even["A2"], even["A2"], chosen + [u])["ce_damage"] for u in pool}
        best = max(scores, key=scores.get)
        gain = scores[best] - cur_even
        if gain < GAIN_MIN:
            path.append({"step": step, "stopped": True, "best": best, "gain_even": gain}); break
        chosen.append(best); pool.remove(best); cur_even = scores[best]
        path.append({"step": step, "added": best, "even_damage": cur_even, "gain_even": gain,
                     "runner_up": sorted(scores.items(), key=lambda kv: -kv[1])[1:4]})

    # evaluation on rows never used for selection
    ev = {"hub_A2": damage(even["A2"], odd["A2"], hub), "hub_A1": damage(even["A1"], odd["A1"], hub)}
    sets = {f"ext{i}": chosen[:len(hub) + i] for i in range(1, len(chosen) - len(hub) + 1)}
    for k, s in sets.items():
        ev[f"{k}_A2"] = damage(even["A2"], odd["A2"], s)
        ev[f"{k}_A1"] = damage(even["A1"], odd["A1"], s)
    final = chosen
    q_a2 = g.block_diff_in_means(backend, even["A2"], final)
    mu_a2 = mu_of(even["A2"], final)
    ev["final_cross"] = {k: v51.summary(torch, v51.removal(backend, p, final, q_a2, mu_a2)) for k, p in cross.items()}
    q_rand = g.block_random_subspace(backend, final, rank=1, seed=1)
    ev["final_random_A2"] = v51.summary(torch, v51.removal(backend, odd["A2"], final, q_rand, mu_of(odd["A2"], final)))

    d = lambda k: ev[k]["ce_damage"]
    n_add = len(final) - len(hub)
    final_a2 = d(f"ext{n_add}_A2") if n_add else d("hub_A2")
    first = (d("ext1_A2") - d("hub_A2")) if n_add else 0.0
    gain_even_total = cur_even - base_even
    gain_odd_total = final_a2 - d("hub_A2")
    predictions = {
        'pred_a_first_gain': first >= FIRST_GAIN,
        'pred_b_reaches_A1': final_a2 >= A1_FRAC * d("hub_A1"),
        'pred_c_cross_clean': all(c["ce_damage"] <= CROSS_FRAC * final_a2 for c in ev["final_cross"].values()),
        'pred_d_random': ev["final_random_A2"]["ce_damage"] <= RAND_FRAC * final_a2,
        'pred_e_selection_holds': n_add > 0 and gain_odd_total >= HOLD_FRAC * gain_even_total,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_complementizer_a2_greedy_result_v1",
              "candidate_id": "corpus.unit_complementizer_a2_greedy_v64", "hub": hub, "final": final, "path": path,
              "bars": {"first_gain": FIRST_GAIN, "a1_frac": A1_FRAC, "cross_frac": CROSS_FRAC, "rand_frac": RAND_FRAC,
                       "hold_frac": HOLD_FRAC, "max_add": MAX_ADD, "gain_min": GAIN_MIN},
              "even_damage": {"hub": base_even, "final": cur_even}, "eval": ev,
              "rows": {"A1": len(rows["A1"]), "A2": len(rows["A2"])},
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "final": final,
                      "path": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items() if k != "runner_up"} for p in path],
                      "damage": {k: round(v["ce_damage"], 3) for k, v in ev.items() if k != "final_cross"},
                      "lb": {k: round(v["ce_lb975"], 3) for k, v in ev.items() if k != "final_cross"},
                      "cross": {k: round(v["ce_damage"], 3) for k, v in ev["final_cross"].items()},
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
