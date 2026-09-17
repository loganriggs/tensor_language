#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_all_cells_capable pred_c_triple_fraction_within_015_of_frozen_constant pred_d_triple_positive_and_selective_everywhere pred_e_token_table_retains_half_everywhere
"""Aspectual has/had definition-of-done battery, step 19: FROZEN NUMERIC PREDICTION on a third lexicon and a new construction.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v8 (three-head readout component
{8.1, 9.1, 9.4}: 50-66% of the margin over four constructions, pooled 0.54), v17 (2-constant token table
for 8.1 transfers to new templates).

WHY. better_circuits §1 PREDICTS OOD: "Frozen formula predicts effect on rows with new templates, new
[lexicon], new endpoints ... Null: effect predicted by a constant, and by a fit on the training panel."
Every number below is frozen before any model access on these rows:
    F* = 0.54   the pooled v8 damage fraction of the triple readout removal (the "constant" prediction)
    +-0.15      the acceptance band, i.e. [0.39, 0.69] per construction
    T[since] = 0.220, T[by] = 0.182   the 8.1 pattern constants from v17 (discovery rows)
Rows: a THIRD lexicon (16 agents x 16 periods, disjoint from every prior panel) on the two discovery
constructions and one NEW construction, "Since the P started, the A" / "By the time the P ended, the A"
(subordinate clause + comma; the cue is `since`/`by` and the final input token is the agent).

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   no-edit forward matches producer.native <= 1e-4 on 96 rows
    pred_b_all_cells_capable           all six (construction x cue) native cells >= 0.85
    pred_c_triple_fraction_within_015_of_frozen_constant   |fraction - 0.54| <= 0.15 in every construction
    pred_d_triple_positive_and_selective_everywhere   damage positive on >= 0.75 rows and every unrelated
                                       reader's mean |move| <= 0.25 x damage (raw gate; no null arms here --
                                       the norm-matched null was established in v4/v5/v8)
    pred_e_token_table_retains_half_everywhere   8.1 final-query slice replaced by T[cue] x lamb x v1(cue)
                                       retains >= 0.50 of the zero-slice damage in every construction

PRICE (registered maximum): 96 rows in 3 batches; native 3 + producer 3 + triple 3 + zero-8.1 3 + table
(3 capture + 3 arm) = 21 forwards; 0 backwards; 0 fits. Bar <= 26.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_frozen_prediction_v19_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_frozen_prediction_v19"
EXPECTED_ROWS_SHA256 = "77777593701c015ec2bd5742767a8d57bbee88d406ca5dfa2cef297f1d84f16c"
TOKENS = {"has": 468, "had": 550}
FROZEN_FRACTION, BAND, TABLE = 0.54, 0.15, {True: 0.22014020383358002, False: 0.18158479034900665}
CAPABILITY_MIN, LIVE_POSITIVE, GATE_RATIO, RETAIN_MIN, INSTRUMENT_TOL = 0.85, 0.75, 0.25, 0.50, 1e-4
FORWARDS_MAX = 26
AGENTS = ("nun", "cook", "waiter", "broker", "tutor", "pastor", "referee", "witness", "tenant", "landlord", "dealer", "printer", "author", "hunter", "knight", "wizard")
PERIODS = ("recess", "vacation", "retreat", "rally", "parade", "banquet", "ceremony", "exam", "lecture", "seminar", "workshop", "interview", "race", "battle", "war", "epidemic")
CONSTRUCTIONS = {
    "fronted": (lambda p, a: f"Since last {p} the {a}", lambda p, a: f"By last {p} the {a}"),
    "report": (lambda p, a: f"The record shows that since last {p} the {a}", lambda p, a: f"The record shows that by last {p} the {a}"),
    "clause_comma": (lambda p, a: f"Since the {p} started, the {a}", lambda p, a: f"By the time the {p} ended, the {a}"),
}
TRIPLE = (L.Component("attn8_h1_final", 8, "attn", (1,), "final"), L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"))
HEAD81 = TRIPLE[0]


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "frozen": {"fraction": FROZEN_FRACTION, "band": BAND, "table": {"since": TABLE[True], "by": TABLE[False]}},
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"capability_min": CAPABILITY_MIN, "live_positive": LIVE_POSITIVE, "gate_ratio": GATE_RATIO, "retain_min": RETAIN_MIN, "instrument_tol": INSTRUMENT_TOL}}


def cue_position(row):
    for s, tid in enumerate(row.ids):
        if L.ENCODING.decode([tid]).strip().lower() in ("since", "by"):
            return s
    raise RuntimeError(f"no cue token in {row.text!r}")


def main() -> None:
    rows = L.build_rows_lexicon(AGENTS, PERIODS, CONSTRUCTIONS, "lexicon3")
    if L.rows_sha256(rows) != EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, TRIPLE, TOKENS["has"], TOKENS["had"])
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    triple, n = v1._run_arm(fw, rows, components=TRIPLE, mode="project"); forwards += n
    zero81, n = v1._run_arm(fw, rows, components=(HEAD81,), mode="zero"); forwards += n
    is_cue = lambda r, s: s == cue_position(r)
    table = {}
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        table.update(L.source_restricted_slices(fw, chunk, HEAD81, is_cue, ("inherited",), lambda r, s: TABLE[r.present] if is_cue(r, s) else None)); forwards += 1
    fw.subtract = table
    tabled, n = v1._run_arm(fw, rows, components=(HEAD81,), mode="replace"); forwards += n
    fw.use_subtract = False

    constructions = sorted(CONSTRUCTIONS)
    capability, report = {}, {}
    for c in constructions:
        idx = [i for i, r in enumerate(rows) if r.construction == c]
        sub_rows, sub_native = [rows[i] for i in idx], [native[i] for i in idx]
        for present in (True, False):
            cell = [1.0 if native[i]["answer"] > native[i]["foil"] else 0.0 for i in idx if rows[i].present == present]
            capability[f"{c}/{'since' if present else 'by'}"] = sum(cell) / len(cell)
        t = L.summarize(sub_rows, sub_native, [triple[i] for i in idx])
        z = L.summarize(sub_rows, sub_native, [zero81[i] for i in idx])["target_damage_mean"]
        a = L.summarize(sub_rows, sub_native, [tabled[i] for i in idx])["target_damage_mean"]
        gates = {name: t[f"{name}_abs_move_mean"] <= GATE_RATIO * t["target_damage_mean"] for name in L.UNRELATED}
        report[c] = {"triple": t, "within_band": abs(t["target_damage_fraction"] - FROZEN_FRACTION) <= BAND,
                     "positive_ok": t["target_damage_positive_fraction"] >= LIVE_POSITIVE, "gates": gates,
                     "zero_8_1_damage": z, "table_damage": a, "table_retention": 1.0 - a / z}
        print(c, "fraction", round(t["target_damage_fraction"], 3), "pos", round(t["target_damage_positive_fraction"], 3), "gates", gates, "table retention", round(report[c]["table_retention"], 3))
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_all_cells_capable": all(v >= CAPABILITY_MIN for v in capability.values()),
        "pred_c_triple_fraction_within_015_of_frozen_constant": all(r["within_band"] for r in report.values()),
        "pred_d_triple_positive_and_selective_everywhere": all(r["positive_ok"] and all(r["gates"].values()) for r in report.values()),
        "pred_e_token_table_retains_half_everywhere": all(r["table_retention"] >= RETAIN_MIN for r in report.values()),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_frozen_prediction_result_v19", "candidate_id": CANDIDATE_ID, "plan": _plan(rows),
              "instrument_max_abs_error": instrument, "capability": capability, "constructions": report,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "capability": capability, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
