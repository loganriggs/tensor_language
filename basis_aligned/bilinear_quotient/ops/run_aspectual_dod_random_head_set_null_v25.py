#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_component_beats_every_random_triple pred_c_random_triples_are_not_live pred_d_component_damage_exceeds_random_median_tenfold
"""Aspectual has/had definition-of-done battery, step 25: matched-count RANDOM HEAD-SET null for SIMPLE.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v7 (blind 162-head sweep), v8 (the
three-head component removes 50-66% of the margin).

WHY. better_circuits §1 SIMPLE: the description length is a count and its null is "the same count for a
random component of matching effect size". Matching effect size is not possible (nothing else reaches it,
v7), so the practical null is matched COUNT: sixteen random three-head sets, each removed along its own
weight-only readout directions `O_h^T(u_has - u_had)` at the final query, scored exactly like the component.
This prices "three heads" as a description: how much has/had effect does an arbitrary three-head readout
removal carry?

ROWS: 64 discovery rows (opened). Random sets: seeded (2026_09_17), drawn from the 159 heads outside the
component, without replacement within a set; no set is retuned.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   no-edit forward matches producer.native <= 1e-4
    pred_b_component_beats_every_random_triple   component damage > max over the 16 random-triple damages
    pred_c_random_triples_are_not_live  no random triple passes LIVE (>= 0.10 fraction and positive >= 0.75)
    pred_d_component_damage_exceeds_random_median_tenfold   component damage >= 10 x median |random damage|

PRICE (registered maximum): native 2 + producer 2 + component 2 + 16 x 2 = 38 forwards; 0 backwards; 0 fits.
Bar <= 44.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import random
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_random_head_set_null_v25_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_random_head_set_null_v25"
TOKENS = {"has": 468, "had": 550}
LIVE_FRACTION, LIVE_POSITIVE, TENFOLD, INSTRUMENT_TOL = 0.10, 0.75, 10.0, 1e-4
FORWARDS_MAX = 44
COMPONENT = (L.Component("attn8_h1_final", 8, "attn", (1,), "final"), L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"))
EXCLUDED = {(8, 1), (9, 1), (9, 4)}
POOL = [(l, h) for l in range(18) for h in range(9) if (l, h) not in EXCLUDED]
SETS = [tuple(sorted(random.Random(2026_09_17 + s).sample(POOL, 3))) for s in range(16)]


def components_for(triple):
    by_layer = {}
    for l, h in triple:
        by_layer.setdefault(l, []).append(h)
    return tuple(L.Component(f"rand_attn{l}_" + "_".join(map(str, hs)), l, "attn", tuple(hs), "final") for l, hs in sorted(by_layer.items()))


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "random_sets": SETS, "forwards_max": FORWARDS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only", "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "tenfold": TENFOLD, "instrument_tol": INSTRUMENT_TOL}}


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    all_components = COMPONENT + tuple(c for s in SETS for c in components_for(s))
    fw.directions = L.readout_directions(backend.model, all_components, TOKENS["has"], TOKENS["had"])
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    comp_arm, n = v1._run_arm(fw, rows, components=COMPONENT, mode="project"); forwards += n
    comp = L.summarize(rows, native, comp_arm)
    randoms = []
    for s in SETS:
        arm, n = v1._run_arm(fw, rows, components=components_for(s), mode="project"); forwards += n
        sm = L.summarize(rows, native, arm)
        randoms.append({"set": [f"{l}.{h}" for l, h in s], "damage": sm["target_damage_mean"], "fraction": sm["target_damage_fraction"],
                        "positive": sm["target_damage_positive_fraction"], "live": sm["target_damage_fraction"] >= LIVE_FRACTION and sm["target_damage_positive_fraction"] >= LIVE_POSITIVE})
    dmg = [r["damage"] for r in randoms]
    med_abs = sorted(abs(x) for x in dmg)[len(dmg) // 2]
    print("component", round(comp["target_damage_mean"], 4), "random max", round(max(dmg), 4), "median|.|", round(med_abs, 4), [(r["set"], round(r["damage"], 3)) for r in randoms[:6]])
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_component_beats_every_random_triple": comp["target_damage_mean"] > max(dmg),
        "pred_c_random_triples_are_not_live": not any(r["live"] for r in randoms),
        "pred_d_component_damage_exceeds_random_median_tenfold": comp["target_damage_mean"] >= TENFOLD * med_abs,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_random_head_set_null_result_v25", "candidate_id": CANDIDATE_ID, "plan": _plan(rows), "instrument_max_abs_error": instrument,
              "component": comp, "random": randoms, "random_damage_max": max(dmg), "random_damage_median_abs": med_abs,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
