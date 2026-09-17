#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_full_recompute_replays_native pred_b_zeroing_8_1_at_np_is_live pred_c_cue_inherited_term_retains_most pred_d_cue_both_branches_retains_more pred_e_other_single_sources_retain_little
"""Temporal will/had DoD battery, step 12 (v39): head 8.1 at the NP positions as a token-only adverb reader.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v37 (attention8's NP write on 11.3's reader
direction is 0.996 head 8.1, sourced 0.98 from the tomorrow/earlier token).

WHY. On the aspectual line head 8.1's service was one term: native pattern on the cue x lamb x block-0 value of
the cue token (v11b/v12). v37 says the same head does the same job here, one position earlier in the chain
(writing the NP state 11.3 reads). Edit: replace 8.1's pre-c_proj slice at the first determiner and the agent by
its cue-only term (inherited branch alone; both branches), or by other single sources, and score retention
against zeroing those two slices. Full recomputation must replay native.

PREDICTIONS (scored as written; failures preserved)
    pred_a_full_recompute_replays_native   <= 1e-3
    pred_b_zeroing_8_1_at_np_is_live       zero-slice damage >= 0.10 fraction, positive >= 0.75
    pred_c_cue_inherited_term_retains_most cue-only, inherited-only retention >= 0.60
    pred_d_cue_both_branches_retains_more  cue-only, both branches retention >= 0.70
    pred_e_other_single_sources_retain_little   `the1` / agent / prefix single-source arms (both branches) each <= 0.30
                                           (prefix scored on the report rows only)

PRICE (registered maximum): native 2 + producer 2 + zero 2 + 6 arms x (2 capture + 2 arm) = 30 forwards; 0 backwards;
0 fits. Bar <= 34.
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
import run_temporal_dod_np_writer_fold_v35 as v35

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_8_1_np_token_only_v39_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_8_1_np_token_only_v39"
LIVE_FRACTION, LIVE_POSITIVE, RETAIN_C, RETAIN_D, OTHER_MAX, INSTRUMENT_TOL = 0.10, 0.75, 0.60, 0.70, 0.30, 1e-3
FORWARDS_MAX = 34
NP = "np"
HEAD = L.Component("attn8_h1_np", 8, "attn", (1,), NP)


def cat(row, s):
    n = len(row.ids)
    cue = next(i for i, t in enumerate(row.ids) if L.ENCODING.decode([t]).strip().lower() in ("tomorrow", "earlier"))
    return {n - 1: "place", n - 2: "the2", n - 3: "prep", n - 4: "agent", n - 5: "the1", cue: "cue"}.get(s, "prefix")


def main() -> None:
    rows = v28.build()
    original = L.positions_of
    L.positions_of = lambda row, where: v35.np_positions(row) if where == NP else original(row, where)
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    zero, n = v1._run_arm(fw, rows, components=(HEAD,), mode="zero"); forwards += n
    arms = {"full": (lambda r, s: True, ("current", "inherited")), "cue_inherited": (lambda r, s: cat(r, s) == "cue", ("inherited",)),
            "cue_both": (lambda r, s: cat(r, s) == "cue", ("current", "inherited")), "the1_both": (lambda r, s: cat(r, s) == "the1", ("current", "inherited")),
            "agent_both": (lambda r, s: cat(r, s) == "agent", ("current", "inherited")), "prefix_both": (lambda r, s: cat(r, s) == "prefix", ("current", "inherited"))}
    results = {}
    for name, (keep, branches) in arms.items():
        table = {}
        for start in range(0, len(rows), v1.BATCH):
            table.update(L.source_restricted_slices(fw, rows[start:start + v1.BATCH], HEAD, keep, branches)); forwards += 1
        fw.subtract = table
        arm, n = v1._run_arm(fw, rows, components=(HEAD,), mode="replace"); forwards += n
        results[name] = arm
    fw.use_subtract = False
    instrument = max(max(abs(a["answer"] - b["answer"]), abs(a["foil"] - b["foil"])) for a, b in zip(results["full"], native))
    zs = L.summarize(rows, native, zero)
    def retention(name, idx=None):
        idx = list(range(len(rows))) if idx is None else idx
        sub, subn = [rows[i] for i in idx], [native[i] for i in idx]
        z = L.summarize(sub, subn, [zero[i] for i in idx])["target_damage_mean"]; a = L.summarize(sub, subn, [results[name][i] for i in idx])["target_damage_mean"]
        return 1.0 - a / z
    report_idx = [i for i, r in enumerate(rows) if r.construction == "report_frame"]
    ret = {name: retention(name, report_idx if name == "prefix_both" else None) for name in arms}
    print("zero", round(zs["target_damage_mean"], 4), "fraction", round(zs["target_damage_fraction"], 3), "pos", round(zs["target_damage_positive_fraction"], 3), {k: round(v, 3) for k, v in ret.items()})
    predictions = {"pred_a_full_recompute_replays_native": instrument <= INSTRUMENT_TOL,
                   "pred_b_zeroing_8_1_at_np_is_live": zs["target_damage_fraction"] >= LIVE_FRACTION and zs["target_damage_positive_fraction"] >= LIVE_POSITIVE,
                   "pred_c_cue_inherited_term_retains_most": ret["cue_inherited"] >= RETAIN_C, "pred_d_cue_both_branches_retains_more": ret["cue_both"] >= RETAIN_D,
                   "pred_e_other_single_sources_retain_little": all(ret[k] <= OTHER_MAX for k in ("the1_both", "agent_both", "prefix_both"))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_8_1_np_token_only_result_v39", "candidate_id": CANDIDATE_ID, "instrument_max_abs_error": instrument, "zero": zs, "retention": ret,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
