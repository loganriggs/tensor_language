#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_pairwise_mobius_terms_below_gate pred_c_five_piece_gap_replays_below_gate pred_d_module_split_is_no_less_additive_than_random_splits pred_e_random_pieces_are_not_individually_selective
"""Aspectual has/had definition-of-done battery, step 3: COMPOSES with a random-split null.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent receipt:
`circuits/followups/aspectual_anchor_dod_removal_v2_result.json` (midpoint removal; joint of five
components removes 1.79 of the 2.07-logit native margin; |joint - sum(singles)| = 0.06).

WHY. better_circuits.md §1 COMPOSES: "Möbius interaction terms among the component's pieces are
below gate relative to the smallest piece's effect; joint effect predicted from singles. Null:
interaction magnitude for random splits of the same write." v2 measured the five-way gap once
and had no null. This run adds (i) all ten pairwise Möbius terms I(a,b) = dmg(a+b) - dmg(a) -
dmg(b) under midpoint removal, and (ii) sixteen RANDOM COORDINATE SPLITS of the identical total
removal (every half-delta cut into five disjoint coordinate masks that ignore module boundaries),
each scored for the same five-piece gap. If random splits are as additive as the module split,
additivity at this scale is a property of the regime, not of the decomposition -- and that is
the reading we must record rather than claim composition.

ROWS / COMPONENTS / READERS: as v1/v2 (64 fresh rows, sha 5ec7d32f..., five components). Rows
are opened by v1/v2 for removal; no selection step used them.

REGISTERED BARS
    PAIR    |I(a,b)| <= 0.25 x min(dmg(a), dmg(b)) for all ten pairs
    FIVE    |joint - sum(singles)| <= 0.25 x min(single damage over LIVE components) (LIVE as v2)
    NULL    normalized gap g = |joint - sum(pieces)| / joint; module-split g <= MEDIAN of the 16
            random-split g
    SEL     a random piece is "selective" if it passes the v2 GATE on all three readers
            (excess over... no null of a null: use the raw v2 gate: mean|move| <= 0.25 x its damage)
            and is LIVE; prediction: fewer than half of the 80 random pieces are selective+live

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   no-edit forward matches producer.native within 1e-4
    pred_b_pairwise_mobius_terms_below_gate    PAIR holds for all ten pairs. Prior: uncertain;
                                       the attn5 piece (0.09) makes the bar 0.023 for its pairs.
    pred_c_five_piece_gap_replays_below_gate   FIVE holds (replay of v2's finding, 0.06 <= 0.13)
    pred_d_module_split_is_no_less_additive_than_random_splits   NULL holds. Stated prior: unsure.
                                       If random splits are equally additive, composition is
                                       uninformative here and the property stays "not shown".
    pred_e_random_pieces_are_not_individually_selective   SEL holds: random pieces of the same
                                       removal are mostly not individually live+selective, so the
                                       module split's per-piece selectivity (v2) is not generic.

PRICE (registered maximum): capture 2 + native 2 + producer 2 + singles 10 + pairs 20 + joint 2
+ 16 splits x 5 pieces x 2 = 160 -> 198 forwards; 0 backwards; 0 fits. Bar <= 220.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_composition_v3_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_composition_v3"
EXPECTED_ROWS_SHA256 = v1.EXPECTED_ROWS_SHA256
SPLIT_SEEDS = tuple(range(201, 217))
PAIR_RATIO, FIVE_RATIO, GATE_RATIO, LIVE_FRACTION, LIVE_POSITIVE, INSTRUMENT_TOL = 0.25, 0.25, 0.25, 0.10, 0.75, 1e-4
FORWARDS_MAX = 220


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows),
            "components": [c.name for c in L.COMPONENTS], "split_seeds": list(SPLIT_SEEDS),
            "pairs": 10, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only",
            "bars": {"pair_ratio": PAIR_RATIO, "five_ratio": FIVE_RATIO, "gate_ratio": GATE_RATIO,
                     "live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE,
                     "instrument_tol": INSTRUMENT_TOL}}


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != EXPECTED_ROWS_SHA256:
        raise SystemExit("fresh rows changed; refusing to run against an unregistered panel")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    forwards = 0
    store = {}
    for chunk in v1._batches(rows):
        store.update(fw.capture(chunk, L.COMPONENTS)); forwards += 1
    fw.deltas = L.paired_deltas(store, rows, L.COMPONENTS)

    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument_err = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    pred_a = instrument_err <= INSTRUMENT_TOL

    def damage(components, mode="midpoint"):
        nonlocal forwards
        arm, n = v1._run_arm(fw, rows, components=components, mode=mode); forwards += n
        return L.summarize(rows, native, arm)

    singles = {c.name: damage((c,)) for c in L.COMPONENTS}
    live = [k for k, s in singles.items() if s["target_damage_fraction"] >= LIVE_FRACTION
            and s["target_damage_positive_fraction"] >= LIVE_POSITIVE]
    pairs = {}
    for a, b in itertools.combinations(L.COMPONENTS, 2):
        s = damage((a, b))
        interaction = s["target_damage_mean"] - singles[a.name]["target_damage_mean"] - singles[b.name]["target_damage_mean"]
        bar = PAIR_RATIO * min(singles[a.name]["target_damage_mean"], singles[b.name]["target_damage_mean"])
        pairs[f"{a.name}+{b.name}"] = {"joint": s["target_damage_mean"], "interaction": interaction,
                                        "bar": bar, "passes": abs(interaction) <= bar}
    joint = damage(L.COMPONENTS)
    sum_singles = sum(s["target_damage_mean"] for s in singles.values())
    five_gap = abs(joint["target_damage_mean"] - sum_singles)
    five_bar = FIVE_RATIO * (min(singles[k]["target_damage_mean"] for k in live) if live else 0.0)
    module_g = five_gap / joint["target_damage_mean"]

    splits = []
    selective_live_pieces = 0
    for seed in SPLIT_SEEDS:
        pieces = L.random_coordinate_split(fw.deltas, seed, pieces=len(L.COMPONENTS))
        piece_damages = []
        for k, table in enumerate(pieces):
            fw.subtract = table
            s = damage(L.COMPONENTS, mode="subtract")
            piece_damages.append(s["target_damage_mean"])
            is_live = (s["target_damage_fraction"] >= LIVE_FRACTION and s["target_damage_positive_fraction"] >= LIVE_POSITIVE)
            is_sel = all(s[f"{name}_abs_move_mean"] <= GATE_RATIO * s["target_damage_mean"] for name in L.UNRELATED)
            selective_live_pieces += int(is_live and is_sel)
        gap = abs(joint["target_damage_mean"] - sum(piece_damages))
        splits.append({"seed": seed, "piece_damages": piece_damages, "gap": gap,
                       "normalized_gap": gap / joint["target_damage_mean"]})
        print("split", seed, [round(x, 3) for x in piece_damages], "gap", round(gap, 4))
    fw.use_subtract = False
    random_g = sorted(s["normalized_gap"] for s in splits)
    random_median = random_g[len(random_g) // 2]

    predictions = {
        "pred_a_instrument_replays_native": bool(pred_a),
        "pred_b_pairwise_mobius_terms_below_gate": all(p["passes"] for p in pairs.values()),
        "pred_c_five_piece_gap_replays_below_gate": bool(live) and five_gap <= five_bar,
        "pred_d_module_split_is_no_less_additive_than_random_splits": module_g <= random_median,
        "pred_e_random_pieces_are_not_individually_selective": selective_live_pieces < len(SPLIT_SEEDS) * len(L.COMPONENTS) / 2,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_composition_result_v3", "candidate_id": CANDIDATE_ID,
              "plan": _plan(rows), "instrument_max_abs_error": instrument_err,
              "singles": {k: s["target_damage_mean"] for k, s in singles.items()}, "live": live,
              "pairs": pairs, "joint": joint["target_damage_mean"], "sum_singles": sum_singles,
              "five_gap": five_gap, "five_bar": five_bar, "module_normalized_gap": module_g,
              "random_splits": splits, "random_normalized_gap_median": random_median,
              "random_normalized_gap_min": random_g[0], "random_normalized_gap_max": random_g[-1],
              "random_pieces_live_and_selective": selective_live_pieces,
              "random_pieces_total": len(SPLIT_SEEDS) * len(L.COMPONENTS),
              "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "pairs": {k: round(v["interaction"], 4) for k, v in pairs.items()},
                      "module_g": module_g, "random_g_median": random_median,
                      "random_selective_live": selective_live_pieces, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
