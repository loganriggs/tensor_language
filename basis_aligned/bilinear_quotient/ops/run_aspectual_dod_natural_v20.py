#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_since_rows_shift_toward_had pred_c_by_rows_shift_toward_has pred_d_head_8_1_alone_shifts_the_same_way pred_e_unrelated_readers_within_gate
"""Aspectual has/had definition-of-done battery, step 20: NATURAL-TEXT (FineWeb) rows, frozen mechanism prediction.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`; rows: `aspectual_dod_natural_rows.py` (outcome-blind
miner, receipt `circuits/followups/aspectual_anchor_dod_natural_rows_v20.json`, 64 rows: 16 per (cue, next
token) cell, FineWeb sample-10BT stream order, no model score in selection).

WHY. better_circuits §1 PREDICTS OOD: "ideally new corpus". The component's mechanism (v10-v15) is that head
8.1 reads the block-0 VALUE of the since/by TOKEN -- a token-only lookup that cannot see whether `by` is a
deadline or an agent, or whether `since` is temporal or causal. So on natural rows the prediction is a
cue-conditional SHIFT of the has-had contrast, not "damage to the correct answer": removing the readout
component must lower has-had on `since`-cued rows and raise it on `by`-cued rows, whatever the true next
token is. The natural rows are exactly that: a has/had target with a `since`/`by` token in the preceding 10
tokens, whatever its sense (many `by` rows are agentive). Frozen numbers below.

ARMS: native; triple {8.1, 9.1, 9.4} readout removal (weight-only directions); head 8.1 alone.
READERS: has-had (signed, NOT oriented to the true answer), was-were, who-which, night-day.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native      no-edit forward matches producer.native <= 1e-4 on 64 rows
    pred_b_since_rows_shift_toward_had    triple removal: mean d(has-had) <= -0.15 logits over the 32 since rows
                                          and <= 0 on >= 60% of them
    pred_c_by_rows_shift_toward_has       triple removal: mean d(has-had) >= +0.15 over the 32 by rows and >= 0 on
                                          >= 60% of them. Prior: unsure -- agentive `by` may sit in contexts
                                          where the block-9 bank never forms; head 8.1's direct term should
                                          still fire.
    pred_d_head_8_1_alone_shifts_the_same_way   8.1-only removal satisfies both sign conditions with the weaker
                                          magnitude |mean| >= 0.05 per cue
    pred_e_unrelated_readers_within_gate  each unrelated reader's mean |move| <= 0.25 x |mean d(has-had)| pooled
                                          over all 64 rows, for the triple removal

PRICE (registered maximum): 64 rows in 2 batches; native 2 + producer 2 + triple 2 + 8.1 2 = 8 forwards;
0 backwards; 0 fits. Bar <= 12.
"""
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
ROWS = ROOT / "circuits/followups/aspectual_anchor_dod_natural_rows_v20.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_natural_v20_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_natural_v20"
TOKENS = {"has": 468, "had": 550}
SHIFT_MIN, DIRECTION_MIN, WEAK_MIN, GATE_RATIO, INSTRUMENT_TOL = 0.15, 0.60, 0.05, 0.25, 1e-4
FORWARDS_MAX = 12
TRIPLE = (L.Component("attn8_h1_final", 8, "attn", (1,), "final"), L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"))


def load_rows():
    doc = json.loads(ROWS.read_text())
    if hashlib.sha256(json.dumps(doc["rows"], sort_keys=True).encode()).hexdigest() != doc["rows_sha256"]:
        raise SystemExit("natural rows changed")
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for r in doc["rows"]:
        ids = tuple(r["ids"])
        present = r["label"] == "has"
        rows.append(L.Row(hashlib.sha256(json.dumps([r["doc_index"], r["position"]]).encode()).hexdigest()[:24],
                          f"natural_{r['cue']}", r["doc_index"], present, r["text"], ids, " has" if present else " had",
                          " had" if present else " has", TOKENS["has"] if present else TOKENS["had"],
                          TOKENS["had"] if present else TOKENS["has"], len(ids) - 1, (), reader_ids))
    return rows, doc["rows_sha256"]


def _plan(rows, sha):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": sha, "forwards_max": FORWARDS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only",
            "bars": {"shift_min": SHIFT_MIN, "direction_min": DIRECTION_MIN, "weak_min": WEAK_MIN, "gate_ratio": GATE_RATIO, "instrument_tol": INSTRUMENT_TOL}}


def main() -> None:
    rows, sha = load_rows()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows, sha), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, TRIPLE, TOKENS["has"], TOKENS["had"])
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    triple, n = v1._run_arm(fw, rows, components=TRIPLE, mode="project"); forwards += n
    alone, n = v1._run_arm(fw, rows, components=(TRIPLE[0],), mode="project"); forwards += n

    def shift(arm, cue):
        idx = [i for i, r in enumerate(rows) if r.construction == f"natural_{cue}"]
        d = [arm[i]["target_has_had"] - native[i]["target_has_had"] for i in idx]
        return {"mean": sum(d) / len(d), "n": len(d), "fraction_negative": sum(1 for x in d if x <= 0) / len(d),
                "fraction_positive": sum(1 for x in d if x >= 0) / len(d),
                "native_accuracy": sum(1 for i in idx if native[i]["answer"] > native[i]["foil"]) / len(idx)}
    report = {"triple": {c: shift(triple, c) for c in ("since", "by")}, "head_8_1": {c: shift(alone, c) for c in ("since", "by")}}
    pooled_abs = abs(sum(triple[i]["target_has_had"] - native[i]["target_has_had"] for i in range(len(rows))) / len(rows))
    unrelated = {name: sum(abs(triple[i][name] - native[i][name]) for i in range(len(rows))) / len(rows) for name in L.UNRELATED}
    for k, v in report.items():
        print(k, json.dumps({c: {kk: round(vv, 4) for kk, vv in s.items()} for c, s in v.items()}))
    print("pooled |shift|", round(pooled_abs, 4), "unrelated", {k: round(v, 4) for k, v in unrelated.items()})
    ts, tb, hs, hb = report["triple"]["since"], report["triple"]["by"], report["head_8_1"]["since"], report["head_8_1"]["by"]
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_since_rows_shift_toward_had": ts["mean"] <= -SHIFT_MIN and ts["fraction_negative"] >= DIRECTION_MIN,
        "pred_c_by_rows_shift_toward_has": tb["mean"] >= SHIFT_MIN and tb["fraction_positive"] >= DIRECTION_MIN,
        "pred_d_head_8_1_alone_shifts_the_same_way": hs["mean"] <= -WEAK_MIN and hs["fraction_negative"] >= DIRECTION_MIN and hb["mean"] >= WEAK_MIN and hb["fraction_positive"] >= DIRECTION_MIN,
        "pred_e_unrelated_readers_within_gate": all(v <= GATE_RATIO * pooled_abs for v in unrelated.values()),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_natural_result_v20", "candidate_id": CANDIDATE_ID, "plan": _plan(rows, sha),
              "instrument_max_abs_error": instrument, "shifts": report, "pooled_abs_shift": pooled_abs, "unrelated_abs_move_mean": unrelated,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
