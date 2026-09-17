#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_full_recompute_replays_native pred_b_bank_write_removal_reduces_block9_readout_damage pred_c_mediated_share_in_band pred_d_bank_write_removal_alone_is_live
"""Aspectual has/had definition-of-done battery, step 18: the relay 8.1@bank -> 9.1/9.4 as an EDIT.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v14 (fold: head 8.1's write at
`last`/period/`the` on the block-9 reader directions is 8.1 reading the cue; attention8 = 29% / 17% of
the bank contrast on those directions, v13).

WHY. Folds nominate, edits decide (better_circuits §3.3). v14 attributed part of the bank the block-9
heads read to head 8.1's write there. The edit: remove head 8.1's write at the three bank positions
only (zero its slice there; the final-query slice stays native) and measure how much the block-9
readout removal's damage shrinks -- the MEDIATED share. Arms: A = 9.1/9.4 readout removal alone (v4);
B = 8.1 zeroed at bank positions alone; AB = both. Mediated share m = 1 - (dmg(AB) - dmg(B)) / dmg(A):
the fraction of the block-9 readout effect that no longer exists once 8.1's bank write is gone.
Null-free by design: the comparison is between the head's own two sites.

ROWS: 64 discovery rows (opened).

PREDICTIONS (scored as written; failures preserved)
    pred_a_full_recompute_replays_native   replacing the 8.1 bank slices by their full recomputation
                                           replays native <= 1e-3 (instrument for the bank-site edit)
    pred_b_bank_write_removal_reduces_block9_readout_damage   dmg(AB) - dmg(B) < dmg(A)
    pred_c_mediated_share_in_band          0.15 <= m <= 0.60 (fold nominated 17-29% of the bank on the
                                           reader direction; the readout damage also has the MLP part)
    pred_d_bank_write_removal_alone_is_live   dmg(B) >= 0.10 fraction and positive on >= 0.75 rows

PRICE (registered maximum): native 2 + producer 2 + recompute (2 capture + 2 arm) + A 2 + B 2 + AB 2 =
14 forwards; 0 backwards; 0 fits. Bar <= 18.
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
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_relay_mediation_v18_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_relay_mediation_v18"
TOKENS = {"has": 468, "had": 550}
BAND, LIVE_FRACTION, LIVE_POSITIVE, INSTRUMENT_TOL = (0.15, 0.60), 0.10, 0.75, 1e-3
FORWARDS_MAX = 18
BANK81 = L.Component("attn8_h1_bank", 8, "attn", (1,), "source")
READ9 = L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final")


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "forwards_max": FORWARDS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only", "bars": {"band": list(BAND), "live_fraction": LIVE_FRACTION,
                                                                "live_positive": LIVE_POSITIVE, "instrument_tol": INSTRUMENT_TOL}}


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, (READ9,), TOKENS["has"], TOKENS["had"])
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    # instrument: full recomputation at the bank slices
    table = {}
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        table.update(L.source_restricted_slices(fw, chunk, BANK81, lambda r, s: True, ("current", "inherited"))); forwards += 1
    fw.subtract = table
    full, n = v1._run_arm(fw, rows, components=(BANK81,), mode="replace"); forwards += n
    fw.use_subtract = False
    instrument = max(max(abs(a["answer"] - b["answer"]), abs(a["foil"] - b["foil"])) for a, b in zip(full, native))

    A, n = v1._run_arm(fw, rows, components=(READ9,), mode="project"); forwards += n
    B, n = v1._run_arm(fw, rows, components=(BANK81,), mode="zero"); forwards += n
    # AB: zero 8.1 at bank AND project 9.1/9.4 -- the lib applies one mode per forward, so AB uses a
    # "zero" for BANK81 via a subtract table equal to the captured slices and "project" for READ9.
    fw.subtract = {k: v for k, v in table.items()}   # subtracting the full recomputation == zeroing
    # per-component mode support: monkeypatch _edit to route BANK81 to "subtract" and READ9 to "project"
    original_edit = fw._edit
    def routed_edit(value, rws, component, mode, seed, layer_kind):
        m = "subtract" if component.name == BANK81.name else "project"
        return original_edit(value, rws, component, m, seed, layer_kind)
    fw._edit = routed_edit
    AB, n = v1._run_arm(fw, rows, components=(BANK81, READ9), mode="project"); forwards += n
    fw._edit = original_edit
    fw.use_subtract = False

    dA = L.summarize(rows, native, A); dB = L.summarize(rows, native, B); dAB = L.summarize(rows, native, AB)
    a, b, ab = dA["target_damage_mean"], dB["target_damage_mean"], dAB["target_damage_mean"]
    mediated = 1.0 - (ab - b) / a
    print(json.dumps({"A_block9_readout": round(a, 4), "B_8_1_bank_zero": round(b, 4), "AB": round(ab, 4),
                      "block9_effect_after_bank_removal": round(ab - b, 4), "mediated_share": round(mediated, 4)}))
    predictions = {
        "pred_a_full_recompute_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_bank_write_removal_reduces_block9_readout_damage": (ab - b) < a,
        "pred_c_mediated_share_in_band": BAND[0] <= mediated <= BAND[1],
        "pred_d_bank_write_removal_alone_is_live": dB["target_damage_fraction"] >= LIVE_FRACTION and dB["target_damage_positive_fraction"] >= LIVE_POSITIVE,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_relay_mediation_result_v18", "candidate_id": CANDIDATE_ID, "plan": _plan(rows),
              "instrument_max_abs_error": instrument, "A": dA, "B": dB, "AB": dAB, "mediated_share": mediated,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
