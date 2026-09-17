#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_full_recompute_replays_native pred_b_15_5_cue_term_retains_half pred_c_block9_cue_terms_retain_half pred_d_cue_inherited_only_is_weaker_than_both_branches
"""Temporal will/had DoD battery, step 11 (v38): are the cue-reading heads token-only readers of the adverb?

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v30 (15.5: cue share 0.46, inherited 0.25;
9.4: cue 0.53, inherited 0.12; 9.1: cue 0.28, the1 0.53).

Edit as the aspectual v11b: replace each head's final-query slice by its cue-only term (native pattern on the
tomorrow/earlier token x value), with (i) the inherited (block-0, token-only) branch alone and (ii) both value
branches; retention against zeroing the slice. Heads: 15.5 and the block-9 pair.

PREDICTIONS (scored as written; failures preserved)
    pred_a_full_recompute_replays_native   full recomputation of each slice replays native <= 1e-3
    pred_b_15_5_cue_term_retains_half      15.5: cue-only, both branches, retention >= 0.50
    pred_c_block9_cue_terms_retain_half    9.1+9.4 jointly: cue-only, both branches, retention >= 0.50. Prior: unsure
                                           (9.1 reads the NP more than the cue).
    pred_d_cue_inherited_only_is_weaker_than_both_branches   for both, inherited-only retention < both-branches
                                           retention (the adverb value is not token-only here)

PRICE (registered maximum): native 2 + producer 2 + per head-set (2 sets): zero 2 + full (capture 2 + arm 2) + cue-both
(2+2) + cue-inh (2+2) = 16 -> 36 forwards; 0 backwards; 0 fits. Bar <= 40.
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
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_cue_term_v38_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_cue_term_v38"
RETAIN_MIN, INSTRUMENT_TOL = 0.50, 1e-3
FORWARDS_MAX = 40
SETS = {"15.5": (L.Component("attn15_h5_final", 15, "attn", (5,), "final"),), "9.1+9.4": (L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"),)}


def is_cue(row, s):
    return L.ENCODING.decode([row.ids[s]]).strip().lower() in ("tomorrow", "earlier")


def main() -> None:
    rows = v28.build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, tuple(c for s in SETS.values() for c in s), v28.WILL, v28.HAD)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument, report = 0.0, {}
    for label, comps in SETS.items():
        comp = comps[0]
        zero, n = v1._run_arm(fw, rows, components=comps, mode="zero"); forwards += n
        zd = L.summarize(rows, native, zero)["target_damage_mean"]
        def arm_with(keep, branches):
            nonlocal forwards
            table = {}
            for start in range(0, len(rows), v1.BATCH):
                table.update(L.source_restricted_slices(fw, rows[start:start + v1.BATCH], comp, keep, branches)); forwards += 1
            fw.subtract = table
            arm, n = v1._run_arm(fw, rows, components=comps, mode="replace"); forwards += n
            return arm
        full = arm_with(lambda r, s: True, ("current", "inherited"))
        instrument = max(instrument, max(max(abs(a["answer"] - b["answer"]), abs(a["foil"] - b["foil"])) for a, b in zip(full, native)))
        both = arm_with(is_cue, ("current", "inherited")); inh = arm_with(is_cue, ("inherited",))
        fw.use_subtract = False
        rb = 1.0 - L.summarize(rows, native, both)["target_damage_mean"] / zd; ri = 1.0 - L.summarize(rows, native, inh)["target_damage_mean"] / zd
        report[label] = {"zero_damage": zd, "cue_both_retention": rb, "cue_inherited_retention": ri}
        print(label, {k: round(v, 4) for k, v in report[label].items()})
    predictions = {"pred_a_full_recompute_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_15_5_cue_term_retains_half": report["15.5"]["cue_both_retention"] >= RETAIN_MIN,
                   "pred_c_block9_cue_terms_retain_half": report["9.1+9.4"]["cue_both_retention"] >= RETAIN_MIN,
                   "pred_d_cue_inherited_only_is_weaker_than_both_branches": all(v["cue_inherited_retention"] < v["cue_both_retention"] for v in report.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_cue_term_result_v38", "candidate_id": CANDIDATE_ID, "instrument_max_abs_error": instrument, "sets": report, "predictions": predictions,
              "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
