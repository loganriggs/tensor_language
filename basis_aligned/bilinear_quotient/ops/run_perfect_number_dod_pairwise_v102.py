#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_pairwise_terms_account_for_the_gap pred_c_pairs_with_11_3_carry_the_interaction pred_d_largest_pair_is_11_3_x_7_8 pred_e_pairs_without_5_3_pass_the_pair_bar
"""Perfect have/has DoD (v102): PAIRWISE Möbius terms of the readout removal at {11.3, 7.8, 5.3, 9.7} on the v97 fresh rows (COMPOSES).

Lane: Claude circuit lane. Parent: v97 (joint 2.11 vs singles 1.38 + 0.26 + 0.14 + 0.09 = 1.86: over-additive; the strict bar
0.25 x min single = 0.02 failed). Same rows, same weight-only projection removal; all six pairs added. I(a,b) = dmg(a+b) - dmg(a) - dmg(b).
The v99 census says the effect is relayed through MLPs 9-17 and 11.3 dominates; a relay in which 11.3 reads state that the earlier
heads (5.3, 7.8, 9.7) write would show as positive pair terms with 11.3.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native            <= 1e-4
    pred_b_pairwise_terms_account_for_the_gap   |sum of the six pair terms - (joint - sum singles)| <= 0.30 x |joint - sum singles|
                                                (higher-order terms are small). Prior: unsure.
    pred_c_pairs_with_11_3_carry_the_interaction  sum of |I| over pairs containing 11.3 >= 0.70 of the sum of |I| over all pairs
    pred_d_largest_pair_is_11_3_x_7_8           the largest |I| is I(11.3, 7.8). Prior: unsure.
    pred_e_pairs_without_5_3_pass_the_pair_bar  |I(a,b)| <= 0.25 x min(dmg a, dmg b) for the three pairs not containing 5.3. Prior: unsure.

PRICE (registered maximum): 3 batches x (native + producer + 4 singles + 6 pairs + joint) = 39 forwards; 0 backwards; 0 fits. Bar <= 42.
"""
from __future__ import annotations
from datetime import datetime, timezone
import itertools, json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_perfect_number_dod_battery_v97 as line
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/perfect_number_dod_pairwise_v102_result.json"
CANDIDATE_ID = "perfect_number.have_vs_has.dod_pairwise_v102"
GAP_RATIO, WITH_11_3_MIN, PAIR_RATIO, INSTRUMENT_TOL = 0.30, 0.70, 0.25, 1e-4
FORWARDS_MAX = 42
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_pairwise_terms_account_for_the_gap": "<= 0.30 x gap", "pred_c_pairs_with_11_3_carry_the_interaction": ">= 0.70",
               "pred_d_largest_pair_is_11_3_x_7_8": "11.3 x 7.8", "pred_e_pairs_without_5_3_pass_the_pair_bar": "<= 0.25 x min single"}


def comps_for(heads, rows, pos, neg):
    return dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, pos, neg, tuple(heads)).set_components()


def main() -> None:
    rows, have, has, agents, objects = line.build()
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "heads": list(line.HEADS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"gap_ratio": GAP_RATIO, "with_11_3_min": WITH_11_3_MIN, "pair_ratio": PAIR_RATIO}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    all_c = comps_for(line.HEADS, rows, have, has)
    fw.directions = L.readout_directions(backend.model, all_c + tuple(c for h in line.HEADS for c in comps_for((h,), rows, have, has)) + tuple(c for p in itertools.combinations(line.HEADS, 2) for c in comps_for(p, rows, have, has)), have, has)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))

    def damage(heads):
        nonlocal forwards
        arm, n = v1._run_arm(fw, rows, components=comps_for(heads, rows, have, has), mode="project"); forwards += n
        return L.summarize(rows, native, arm)["target_damage_mean"]
    name = lambda h: f"{h[0]}.{h[1]}"
    singles = {name(h): damage((h,)) for h in line.HEADS}
    pairs = {f"{name(a)} x {name(b)}": damage((a, b)) - singles[name(a)] - singles[name(b)] for a, b in itertools.combinations(line.HEADS, 2)}
    joint = damage(line.HEADS)
    gap = joint - sum(singles.values()); pair_sum = sum(pairs.values()); abs_sum = sum(abs(v) for v in pairs.values())
    with_11_3 = sum(abs(v) for k, v in pairs.items() if "11.3" in k.split(" x "))
    largest = max(pairs, key=lambda k: abs(pairs[k]))
    pair_bar = {k: abs(v) <= PAIR_RATIO * min(singles[k.split(" x ")[0]], singles[k.split(" x ")[1]]) for k, v in pairs.items()}
    print("singles", {k: round(v, 3) for k, v in singles.items()}, "joint", round(joint, 3), "gap", round(gap, 3)); print("pairs", {k: round(v, 3) for k, v in pairs.items()}, "pair sum", round(pair_sum, 3))
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
                   "pred_b_pairwise_terms_account_for_the_gap": abs(pair_sum - gap) <= GAP_RATIO * abs(gap),
                   "pred_c_pairs_with_11_3_carry_the_interaction": abs_sum > 0 and with_11_3 / abs_sum >= WITH_11_3_MIN,
                   "pred_d_largest_pair_is_11_3_x_7_8": set(largest.split(" x ")) == {"11.3", "7.8"},
                   "pred_e_pairs_without_5_3_pass_the_pair_bar": all(ok for k, ok in pair_bar.items() if "5.3" not in k.split(" x "))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "perfect_number_dod_pairwise_result_v102", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "singles": singles, "pairs": pairs, "joint": joint,
                               "gap": gap, "pair_sum": pair_sum, "with_11_3_share": with_11_3 / abs_sum if abs_sum else None, "largest_pair": largest, "pair_bar": pair_bar, "predictions": predictions,
                               "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
