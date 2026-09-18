#!/usr/bin/env python3
# BQGATE: LIBRARY -- generic reuse census for auxiliary-decision lines: blind readout sweep + top-4 set battery.
"""One line spec -> the first two rungs of the definition-of-done battery on that line's authored A1/A2 contexts:
blind 162-head weight-only readout sweep, the sweep's top-4 set with 16 norm-matched nulls and three readers, and
the overlap of the top-4 with the shared auxiliary readout family {8.1, 9.1, 9.4, 11.3, 15.5}. Written after
review 4 to make each further line cost one script of ~20 lines (efficiency item).

PRICE is registered per 32-row batch: 181 forwards (native + producer replay, 162-head sweep, set, 16 nulls); the
runner refuses to write a receipt above batches x 181. (v49/v50 with a fixed 380 bar on 128-row panels withheld
their receipts; v49b/v50b are the same science with the corrected registered price.)"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1

ROOT = Path(__file__).resolve().parent.parent
FAMILY = ("attn8_h1", "attn9_h1", "attn9_h4", "attn11_h3", "attn15_h5")
NULL_SEEDS = tuple(range(1501, 1517))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, CAPABILITY_MIN, INSTRUMENT_TOL, FAMILY_MIN = 0.10, 0.75, 0.25, 0.85, 1e-4, 2
FORWARDS_PER_BATCH = 2 + 162 + 1 + 16   # native+producer, sweep, set, nulls -- per 32-row batch
HEADS = tuple(L.Component(f"attn{l}_h{h}", l, "attn", (h,), "final") for l in range(18) for h in range(9))


def rows_from_candidate(module, positive_token: str, negative_token: str, tag: str):
    src = module.build_rows() if hasattr(module, "build_rows") else module.SPEC.build()
    pos_id, neg_id = L._single(positive_token), L._single(negative_token)
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows, seen = [], set()
    for r in src:
        fam = r.get("family", r.get("transform_id"))
        if fam not in ("A1", "A2"):
            continue
        for side in ("base", "donor"):
            text, ids, aid = r[f"{side}_text"], tuple(r[f"{side}_ids"]), r[f"{side}_answer_id"]
            if text in seen or aid not in (pos_id, neg_id):
                continue
            seen.add(text)
            present = aid == pos_id
            rows.append(L.Row(hashlib.sha256(json.dumps([tag, text]).encode()).hexdigest()[:24], fam, r["group_number"], present, text, ids,
                              positive_token if present else negative_token, negative_token if present else positive_token, pos_id if present else neg_id, neg_id if present else pos_id,
                              len(ids) - 1, (), reader_ids))
    return rows, pos_id, neg_id


def run(candidate_id: str, out_name: str, rows, pos_id: int, neg_id: int) -> None:
    out = ROOT / f"circuits/followups/{out_name}"
    sha = L.rows_sha256(rows)
    batches = (len(rows) + v1.BATCH - 1) // v1.BATCH
    forwards_max = batches * FORWARDS_PER_BATCH
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": candidate_id, "rows": len(rows), "rows_sha256": sha, "forwards_max": forwards_max, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, HEADS, pos_id, neg_id)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    capability = {}
    for c in sorted({r.construction for r in rows}):
        for present in (True, False):
            cell = [1.0 if native[i]["answer"] > native[i]["foil"] else 0.0 for i, r in enumerate(rows) if r.construction == c and r.present == present]
            capability[f"{c}/{'positive' if present else 'negative'}"] = sum(cell) / len(cell)
    sweep = {}
    for comp in HEADS:
        arm, n = v1._run_arm(fw, rows, components=(comp,), mode="project"); forwards += n
        sweep[comp.name] = L.summarize(rows, native, arm)["target_damage_mean"]
    ranked = sorted(sweep, key=lambda k: -sweep[k]); top4 = ranked[:4]
    by_layer = {}
    for name in top4:
        comp = next(c for c in HEADS if c.name == name); by_layer.setdefault(comp.layer, []).append(comp.heads[0])
    SET = tuple(L.Component(f"set_attn{l}_" + "_".join(map(str, hs)), l, "attn", tuple(hs), "final") for l, hs in sorted(by_layer.items()))
    fw.directions.update(L.readout_directions(backend.model, SET, pos_id, neg_id))
    joint, n = v1._run_arm(fw, rows, components=SET, mode="project"); forwards += n
    js = L.summarize(rows, native, joint)
    nulls = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=SET, mode="project_random", seed=seed); forwards += n
        nulls.append(L.summarize(rows, native, arm))
    null_max = max(s["target_damage_mean"] for s in nulls)
    null_moves = {name: sum(s[f"{name}_abs_move_mean"] for s in nulls) / len(nulls) for name in L.UNRELATED}
    gates = {name: js[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * js["target_damage_mean"] for name in L.UNRELATED}
    family_overlap = [h for h in top4 if h in FAMILY]
    print("capability", capability); print("top10", [(k, round(sweep[k], 3)) for k in ranked[:10]])
    print("set", top4, "joint", round(js["target_damage_mean"], 4), "fraction", round(js["target_damage_fraction"], 3), "pos", round(js["target_damage_positive_fraction"], 3), "null_max", round(null_max, 4), "gates", gates, "family", family_overlap)
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_native_capability": all(v >= CAPABILITY_MIN for v in capability.values()),
        "pred_c_top4_set_live_and_beats_null": js["target_damage_fraction"] >= LIVE_FRACTION and js["target_damage_positive_fraction"] >= LIVE_POSITIVE and js["target_damage_mean"] > null_max,
        "pred_d_top4_set_selective": all(gates.values()),
        "pred_e_top4_overlaps_shared_family": len(family_overlap) >= FAMILY_MIN,
    }
    if forwards > forwards_max:
        raise SystemExit(f"price exceeded: {forwards} > {forwards_max}")
    result = {"schema": "dod_reuse_census_result_v1", "candidate_id": candidate_id, "rows_sha256": sha, "rows": len(rows), "instrument_max_abs_error": instrument, "capability": capability,
              "sweep_top20": [(k, sweep[k]) for k in ranked[:20]], "sweep": sweep, "set": top4, "joint": js, "null_damage_max": null_max, "null_unrelated_abs_move_mean": null_moves,
              "gates": gates, "family_overlap": family_overlap, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))
