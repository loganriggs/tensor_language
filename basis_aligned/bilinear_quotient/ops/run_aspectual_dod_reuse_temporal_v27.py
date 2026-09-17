#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_triple_will_had_removal_live_and_beats_null pred_c_triple_will_had_removal_selective pred_d_same_three_heads_lead_the_will_had_sweep pred_e_head_8_1_reads_the_temporal_cue_token
"""Aspectual DoD battery, step 27: REUSE — does the same three-head component serve temporal will/had?

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v4-v26 (aspectual has/had readout
component {8.1, 9.1, 9.4}; 8.1 reads the cue token's block-0 value). The temporal will/had line's own
portfolio names "block 8 head 1 cue terms" as its subject-onset writer and block-9 heads 1/4 as readers.

WHY. better_circuits §1 COMPOSES includes reuse: "shared subcomputations can serve multiple tasks". This is
not a new behaviour for discovery: the rows are the existing `temporal_auxiliary.will_vs_had` authority
(A1 "Tomorrow/Earlier the leader near the maple" -> will/had; A2 report frame), and the question is whether
the SAME heads with a different readout contrast `O_h^T(u_will - u_had)` carry it -- same recipe, same
null, same readers, plus the blind 162-head sweep and the 8.1 cue-token test, all with the aspectual bars.

ROWS: 128 contexts (A1 and A2, base and donor sides; opened rows of the temporal line, fresh for this
component). Cue = the token reading tomorrow/earlier. Readers: was-were, who-which, night-day (has-had is
now related and not used as a control).

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native      <= 1e-4
    pred_b_triple_will_had_removal_live_and_beats_null   damage fraction >= 0.10, positive >= 0.75, and
                                          > max of 16 norm-matched random-direction nulls
    pred_c_triple_will_had_removal_selective   each unrelated reader mean|move| <= null mean + 0.25 x damage
    pred_d_same_three_heads_lead_the_will_had_sweep   8.1, 9.1 and 9.4 are all within the top four of the
                                          162-head single-head will/had readout sweep. Prior: unsure.
    pred_e_head_8_1_reads_the_temporal_cue_token   replacing 8.1's final-query slice by its cue-only
                                          inherited term (native pattern x lamb x block-0 value of the
                                          tomorrow/earlier token) retains >= 0.50 of the zero-slice damage

PRICE (registered maximum): 4 batches x (native 1 + producer 1 + triple 1 + 16 nulls + 8.1 zero 1 + cue term
(capture 1 + arm 1) + sweep 162) = 4 x 184 = 736 forwards; 0 backwards; 0 fits. Bar <= 800.
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
import circuit_fast_screen_candidate_temporal_auxiliary as temporal

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_reuse_temporal_v27_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_reuse_temporal_v27"
WILL, HAD = 481, 550
NULL_SEEDS = tuple(range(701, 717))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, RETAIN_MIN, INSTRUMENT_TOL = 0.10, 0.75, 0.25, 0.50, 1e-4
FORWARDS_MAX = 800
TRIPLE = (L.Component("attn8_h1_final", 8, "attn", (1,), "final"), L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"))
HEAD81 = TRIPLE[0]
HEADS = tuple(L.Component(f"attn{l}_h{h}", l, "attn", (h,), "final") for l in range(18) for h in range(9))


def build_rows():
    src = temporal.build_rows() if hasattr(temporal, "build_rows") else temporal.SPEC.build()
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows, seen = [], set()
    for r in src:
        if r["family"] not in ("A1", "A2"):
            continue
        for side in ("base", "donor"):
            text, ids = r[f"{side}_text"], tuple(r[f"{side}_ids"])
            if text in seen:
                continue
            seen.add(text)
            answer_id = r[f"{side}_answer_id"]
            present = answer_id == WILL   # "present" here means the will side
            rows.append(L.Row(hashlib.sha256(json.dumps(["temporal", text]).encode()).hexdigest()[:24], r["family"], r["group_number"], present,
                              text, ids, " will" if present else " had", " had" if present else " will", WILL if present else HAD, HAD if present else WILL,
                              len(ids) - 1, (), reader_ids))
    return rows


def cue_position(row):
    for s, tid in enumerate(row.ids):
        if L.ENCODING.decode([tid]).strip().lower() in ("tomorrow", "earlier"):
            return s
    raise RuntimeError(row.text)


def main() -> None:
    rows = build_rows()
    sha = L.rows_sha256(rows)
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": sha, "forwards_max": FORWARDS_MAX, "model_backwards": 0,
                          "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, TRIPLE + HEADS, WILL, HAD)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    triple, n = v1._run_arm(fw, rows, components=TRIPLE, mode="project"); forwards += n
    ts = L.summarize(rows, native, triple)
    nulls = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=TRIPLE, mode="project_random", seed=seed); forwards += n
        nulls.append(L.summarize(rows, native, arm))
    null_max = max(s["target_damage_mean"] for s in nulls)
    null_moves = {name: sum(s[f"{name}_abs_move_mean"] for s in nulls) / len(nulls) for name in L.UNRELATED}
    gates = {name: ts[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * ts["target_damage_mean"] for name in L.UNRELATED}
    zero81, n = v1._run_arm(fw, rows, components=(HEAD81,), mode="zero"); forwards += n
    is_cue = lambda r, s: s == cue_position(r)
    table = {}
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        table.update(L.source_restricted_slices(fw, chunk, HEAD81, is_cue, ("inherited",))); forwards += 1
    fw.subtract = table
    cue_arm, n = v1._run_arm(fw, rows, components=(HEAD81,), mode="replace"); forwards += n
    fw.use_subtract = False
    z = L.summarize(rows, native, zero81)["target_damage_mean"]
    retention = 1.0 - L.summarize(rows, native, cue_arm)["target_damage_mean"] / z
    sweep = {}
    for comp in HEADS:
        arm, n = v1._run_arm(fw, rows, components=(comp,), mode="project"); forwards += n
        sweep[comp.name] = L.summarize(rows, native, arm)["target_damage_mean"]
    ranked = sorted(sweep, key=lambda k: -sweep[k])
    print("triple", round(ts["target_damage_mean"], 4), "fraction", round(ts["target_damage_fraction"], 3), "pos", round(ts["target_damage_positive_fraction"], 3), "null_max", round(null_max, 4), "gates", gates)
    print("8.1 zero", round(z, 4), "cue-term retention", round(retention, 4))
    print("sweep top8", [(k, round(sweep[k], 3)) for k in ranked[:8]])
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_triple_will_had_removal_live_and_beats_null": ts["target_damage_fraction"] >= LIVE_FRACTION and ts["target_damage_positive_fraction"] >= LIVE_POSITIVE and ts["target_damage_mean"] > null_max,
        "pred_c_triple_will_had_removal_selective": all(gates.values()),
        "pred_d_same_three_heads_lead_the_will_had_sweep": all(h in ranked[:4] for h in ("attn8_h1", "attn9_h1", "attn9_h4")),
        "pred_e_head_8_1_reads_the_temporal_cue_token": retention >= RETAIN_MIN,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_reuse_temporal_result_v27", "candidate_id": CANDIDATE_ID, "rows_sha256": sha, "rows": len(rows),
              "instrument_max_abs_error": instrument, "triple": ts, "null_damage_max": null_max, "null_unrelated_abs_move_mean": null_moves, "gates": gates,
              "head_8_1_zero_damage": z, "head_8_1_cue_term_retention": retention, "sweep_top20": [(k, sweep[k]) for k in ranked[:20]], "sweep": sweep,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
