#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_full_recompute_replays_native pred_b_writer_removal_reduces_11_3_readout_damage pred_c_mediated_share_in_band pred_d_writer_removal_alone_is_live
"""Temporal will/had DoD battery, step 9 (v36): the NP-state relay as an EDIT (folds nominate, edits decide).

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v35 (the subject-NP state head 11.3 reads is
written 70% by mlp9, attn9, mlp10, mlp8, attn8 at the first determiner / agent positions; embedding 0.00).

Arms: A = 11.3 weight-only readout removal at the final query (v28 single, 1.39 logits); B = the five nominated
writers' outputs ZEROED at the two NP positions only (module outputs replaced by zero there; final query and
all other positions native); AB = both. Mediated share m = 1 - (dmg(AB) - dmg(B)) / dmg(A): the fraction of
11.3's readout effect that no longer exists once the nominated NP writers are gone. Instrument: replacing the
five outputs at those positions by their own recomputed values (identity) must replay native.

PREDICTIONS (scored as written; failures preserved)
    pred_a_full_recompute_replays_native   identity replacement changes answer/foil logits <= 1e-3
    pred_b_writer_removal_reduces_11_3_readout_damage   dmg(AB) - dmg(B) < dmg(A)
    pred_c_mediated_share_in_band          0.40 <= m <= 0.90 (fold nominated 0.70)
    pred_d_writer_removal_alone_is_live    dmg(B) >= 0.10 fraction and positive on >= 0.75 rows

PRICE (registered maximum): native 2 + producer 2 + identity (capture-free: subtract zero) 2 + A 2 + B 2 + AB 2 = 12
forwards; 0 backwards; 0 fits. Bar <= 16.
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
import run_temporal_dod_removal_v28 as v28

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_np_mediation_v36_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_np_mediation_v36"
BAND, LIVE_FRACTION, LIVE_POSITIVE, INSTRUMENT_TOL = (0.40, 0.90), 0.10, 0.75, 1e-3
FORWARDS_MAX = 16
HEAD113 = v28.SINGLES[0]
NP = "np"


def np_positions(row):
    n = len(row.ids); return (n - 5, n - 4)


# writers zeroed at NP positions: whole module outputs. Component.where "np" is resolved by the patched positions_of.
WRITERS = (L.Component("mlp8_np", 8, "mlp", (), NP), L.Component("attn8_np", 8, "attn", tuple(range(9)), NP), L.Component("mlp9_np", 9, "mlp", (), NP),
           L.Component("attn9_np", 9, "attn", tuple(range(9)), NP), L.Component("mlp10_np", 10, "mlp", (), NP))


def main() -> None:
    rows = v28.build()
    original = L.positions_of
    L.positions_of = lambda row, where: np_positions(row) if where == NP else original(row, where)
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "writers": [w.name for w in WRITERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0,
                          "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, (HEAD113,), v28.WILL, v28.HAD)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    # identity instrument: subtract a zero vector at every writer slice
    zeros = {}
    for row in rows:
        for w in WRITERS:
            for pos in np_positions(row):
                if w.kind == "mlp":
                    zeros[(row.row_id, w.name, pos, None)] = backend.torch.zeros(L.N_EMBD)
                else:
                    for h in w.heads:
                        zeros[(row.row_id, w.name, pos, h)] = backend.torch.zeros(L.HEAD_DIM)
    fw.subtract = zeros
    ident, n = v1._run_arm(fw, rows, components=WRITERS, mode="subtract"); forwards += n
    fw.use_subtract = False
    instrument = max(max(abs(a["answer"] - b["answer"]), abs(a["foil"] - b["foil"])) for a, b in zip(ident, native))
    A, n = v1._run_arm(fw, rows, components=(HEAD113,), mode="project"); forwards += n
    B, n = v1._run_arm(fw, rows, components=WRITERS, mode="zero"); forwards += n
    orig_edit = fw._edit
    def routed(value, rws, component, mode, seed, layer_kind):
        return orig_edit(value, rws, component, "zero" if component.where == NP else "project", seed, layer_kind)
    fw._edit = routed
    AB, n = v1._run_arm(fw, rows, components=WRITERS + (HEAD113,), mode="project"); forwards += n
    fw._edit = orig_edit
    dA, dB, dAB = (L.summarize(rows, native, x) for x in (A, B, AB))
    a, b, ab = dA["target_damage_mean"], dB["target_damage_mean"], dAB["target_damage_mean"]
    mediated = 1.0 - (ab - b) / a
    print(json.dumps({"A_11_3_readout": round(a, 4), "B_np_writers_zero": round(b, 4), "AB": round(ab, 4), "11_3_effect_after_writers": round(ab - b, 4), "mediated_share": round(mediated, 4),
                      "B_fraction": round(dB["target_damage_fraction"], 3), "B_positive": round(dB["target_damage_positive_fraction"], 3)}))
    predictions = {"pred_a_full_recompute_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_writer_removal_reduces_11_3_readout_damage": (ab - b) < a,
                   "pred_c_mediated_share_in_band": BAND[0] <= mediated <= BAND[1],
                   "pred_d_writer_removal_alone_is_live": dB["target_damage_fraction"] >= LIVE_FRACTION and dB["target_damage_positive_fraction"] >= LIVE_POSITIVE}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_np_mediation_result_v36", "candidate_id": CANDIDATE_ID, "instrument_max_abs_error": instrument, "A": dA, "B": dB, "AB": dAB,
              "mediated_share": mediated, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
