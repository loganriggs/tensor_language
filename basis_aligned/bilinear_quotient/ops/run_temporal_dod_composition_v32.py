#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_pairwise_mobius_terms_below_gate pred_c_four_piece_gap_below_gate pred_d_module_split_no_less_additive_than_random_splits
"""Temporal will/had DoD battery, step 5 (v32): COMPOSES with pairwise Möbius terms and a random-split null.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v28 (S = {11.3, 9.1, 15.5, 9.4}; joint
3.45 vs sum of singles 3.34 -- the 0.25 x min-single bar failed by 0.02).

WHY. better_circuits §1 COMPOSES: Möbius interaction terms below gate relative to the smallest piece, null =
interaction for random splits of the same write. Here the removed write per head is its readout projection
r_h = (w_h . v_h) v_h at the final query; the total removal is the concatenation over the four heads. Random
splits cut that same total into four disjoint coordinate pieces ignoring head boundaries (16 seeds).

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   <= 1e-4
    pred_b_pairwise_mobius_terms_below_gate   |I(a,b)| <= 0.25 x min(dmg a, dmg b) for all six pairs. Prior:
                                       unsure -- v28's four-way gap already exceeded its bar.
    pred_c_four_piece_gap_below_gate   |joint - sum singles| <= 0.25 x min single (replay of v28's finding)
    pred_d_module_split_no_less_additive_than_random_splits   normalized gap of the head split <= median of the
                                       16 random-split normalized gaps

PRICE (registered maximum): 2 batches x (native 1 + producer 1 + capture 1 + singles 4 + pairs 6 + joint 1 +
16 splits x 4 pieces) = 2 x 78 = 156 forwards; 0 backwards; 0 fits. Bar <= 170.
"""
from __future__ import annotations

from datetime import datetime, timezone
import itertools
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_temporal_dod_removal_v28 as v28

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_composition_v32_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_composition_v32"
SPLIT_SEEDS = tuple(range(1001, 1017))
PAIR_RATIO, GAP_RATIO, INSTRUMENT_TOL = 0.25, 0.25, 1e-4
FORWARDS_MAX = 170
SINGLES = v28.SINGLES   # 11.3, 9.1, 15.5, 9.4 as separate components


def main() -> None:
    rows = v28.build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
                          "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, SINGLES, v28.WILL, v28.HAD)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    store = {}
    for start in range(0, len(rows), v1.BATCH):
        store.update(fw.capture(rows[start:start + v1.BATCH], SINGLES)); forwards += 1
    # removal vectors per slice: (w . v_hat) v_hat ; stored as 2x so that random_coordinate_split's half = the removal
    removal = {}
    for (rid, name, pos, head), w in store.items():
        v = fw.directions[(name, head)].float().to(w.device); v = v / v.norm()
        removal[(rid, name, pos, head)] = 2.0 * float(w.float() @ v) * v

    def damage(components, mode="project"):
        nonlocal forwards
        arm, n = v1._run_arm(fw, rows, components=components, mode=mode); forwards += n
        return L.summarize(rows, native, arm)["target_damage_mean"]

    singles = {c.name: damage((c,)) for c in SINGLES}
    pairs = {}
    for a, b in itertools.combinations(SINGLES, 2):
        j = damage((a, b))
        inter = j - singles[a.name] - singles[b.name]; bar = PAIR_RATIO * min(singles[a.name], singles[b.name])
        pairs[f"{a.name}+{b.name}"] = {"joint": j, "interaction": inter, "bar": bar, "passes": abs(inter) <= bar}
    joint = damage(SINGLES)
    gap = abs(joint - sum(singles.values())); gap_bar = GAP_RATIO * min(singles.values()); module_g = gap / joint
    # random coordinate splits of the same total removal, applied via subtract mode over all four components
    splits = []
    for seed in SPLIT_SEEDS:
        pieces = L.random_coordinate_split(removal, seed, pieces=4)
        piece_damage = []
        for table in pieces:
            fw.subtract = table
            piece_damage.append(damage(SINGLES, mode="subtract"))
        splits.append({"seed": seed, "piece_damages": piece_damage, "normalized_gap": abs(joint - sum(piece_damage)) / joint})
    fw.use_subtract = False
    random_g = sorted(s["normalized_gap"] for s in splits); random_median = random_g[len(random_g) // 2]
    print("singles", {k: round(v, 3) for k, v in singles.items()}, "joint", round(joint, 3), "gap", round(gap, 4), "bar", round(gap_bar, 4))
    print("pairs", {k: round(v["interaction"], 4) for k, v in pairs.items()}, "module_g", round(module_g, 4), "random median", round(random_median, 4))
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_pairwise_mobius_terms_below_gate": all(p["passes"] for p in pairs.values()),
        "pred_c_four_piece_gap_below_gate": gap <= gap_bar,
        "pred_d_module_split_no_less_additive_than_random_splits": module_g <= random_median,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_composition_result_v32", "candidate_id": CANDIDATE_ID, "instrument_max_abs_error": instrument, "singles": singles, "pairs": pairs,
              "joint": joint, "gap": gap, "gap_bar": gap_bar, "module_normalized_gap": module_g, "random_splits": splits, "random_normalized_gap_median": random_median,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
