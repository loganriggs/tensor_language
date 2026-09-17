#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_cue_term_replays_v39 pred_b_constant_patterns_transfer_to_report_rows pred_c_constant_patterns_transfer_to_bare_rows pred_d_token_only_table_retains_half_pooled
"""Temporal will/had DoD battery, step 13 (v40): close head 8.1's port on this line with a 4-constant token table.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v39 (8.1 at the NP positions = cue-only inherited
term). As the aspectual v12: replace the native pattern on the cue by the MEDIAN native pattern per (cue, NP position)
taken from the OTHER construction (4 constants per direction; 8 stored numbers), scored only cross-construction.

PREDICTIONS (scored as written; failures preserved)
    pred_a_native_cue_term_replays_v39         native-pattern cue-inherited retention within 0.02 of v39
    pred_b_constant_patterns_transfer_to_report_rows   bare-frame medians scored on report rows: retention >= 0.50
    pred_c_constant_patterns_transfer_to_bare_rows     report medians scored on bare rows: retention >= 0.50
    pred_d_token_only_table_retains_half_pooled        pooled cross-construction retention >= 0.50

PRICE (registered maximum): native 2 + producer 2 + zero 2 + pattern folds 2 positions x 2 batches = 4 + 3 arms x (2 capture
+ 2 arm) = 12 -> 22 forwards; 0 backwards; 0 fits. Bar <= 26.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import statistics
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_temporal_dod_removal_v28 as v28
import run_temporal_dod_np_writer_fold_v35 as v35
import run_temporal_dod_8_1_np_token_only_v39 as v39

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_8_1_np_constant_v40_result.json"
V39 = ROOT / "circuits/followups/temporal_auxiliary_dod_8_1_np_token_only_v39_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_8_1_np_constant_v40"
REPLAY_TOL, TRANSFER_MIN, POOLED_MIN = 0.02, 0.50, 0.50
FORWARDS_MAX = 26
NP = "np"
HEAD = v39.HEAD


def main() -> None:
    rows = v28.build()
    original = L.positions_of
    L.positions_of = lambda row, where: v35.np_positions(row) if where == NP else original(row, where)
    ref_ret = json.loads(V39.read_text())["retention"]["cue_inherited"]
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "stored_constants": 8, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, (L.Component("attn8_h1_final", 8, "attn", (1,), "final"),), v28.WILL, v28.HAD)
    unit = {1: fw.directions[("attn8_h1_final", 1)]}
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    zero, n = v1._run_arm(fw, rows, components=(HEAD,), mode="zero"); forwards += n
    is_cue = lambda r, s: v39.cat(r, s) == "cue"
    cue_pos = lambda r: next(i for i, t in enumerate(r.ids) if L.ENCODING.decode([t]).strip().lower() in ("tomorrow", "earlier"))
    pattern = {}
    for k in range(2):
        for start in range(0, len(rows), v1.BATCH):
            chunk = rows[start:start + v1.BATCH]
            out, lamb = L.head_source_terms_at(fw, chunk, 8, lambda r, k=k: v35.np_positions(r)[k], unit); forwards += 1
            for row, entry in zip(chunk, out):
                pattern[(row.row_id, k)] = entry[1]["pattern"][cue_pos(row)]
    groups = {}
    for row in rows:
        for k in range(2):
            groups.setdefault((row.construction, row.present, k), []).append(pattern[(row.row_id, k)])
    median = {key: statistics.median(v) for key, v in groups.items()}
    other = {"bare_frame": "report_frame", "report_frame": "bare_frame"}

    def arm_with(override):
        nonlocal forwards
        table = {}
        for start in range(0, len(rows), v1.BATCH):
            table.update(L.source_restricted_slices(fw, rows[start:start + v1.BATCH], HEAD, is_cue, ("inherited",), override)); forwards += 1
        fw.subtract = table
        arm, n = v1._run_arm(fw, rows, components=(HEAD,), mode="replace"); forwards += n
        return arm

    def kpos(r, t):
        return 0 if t == v35.np_positions(r)[0] else 1
    native_p = arm_with(None)
    bare_p = arm_with(lambda r, s, t: median[("bare_frame", r.present, kpos(r, t))] if is_cue(r, s) else None)
    report_p = arm_with(lambda r, s, t: median[("report_frame", r.present, kpos(r, t))] if is_cue(r, s) else None)
    fw.use_subtract = False

    def retention(arm, idx):
        sub, subn = [rows[i] for i in idx], [native[i] for i in idx]
        z = L.summarize(sub, subn, [zero[i] for i in idx])["target_damage_mean"]; a = L.summarize(sub, subn, [arm[i] for i in idx])["target_damage_mean"]
        return 1.0 - a / z
    all_idx = list(range(len(rows))); rep = [i for i, r in enumerate(rows) if r.construction == "report_frame"]; bare = [i for i, r in enumerate(rows) if r.construction == "bare_frame"]
    scored = {"native_all": retention(native_p, all_idx), "bare_medians_on_report": retention(bare_p, rep), "report_medians_on_bare": retention(report_p, bare)}
    cross = [bare_p[i] if rows[i].construction == "report_frame" else report_p[i] for i in all_idx]
    scored["cross_pooled"] = retention(cross, all_idx)
    print({k: round(v, 4) for k, v in scored.items()})
    predictions = {"pred_a_native_cue_term_replays_v39": abs(scored["native_all"] - ref_ret) <= REPLAY_TOL, "pred_b_constant_patterns_transfer_to_report_rows": scored["bare_medians_on_report"] >= TRANSFER_MIN,
                   "pred_c_constant_patterns_transfer_to_bare_rows": scored["report_medians_on_bare"] >= TRANSFER_MIN, "pred_d_token_only_table_retains_half_pooled": scored["cross_pooled"] >= POOLED_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_8_1_np_constant_result_v40", "candidate_id": CANDIDATE_ID, "scored": scored, "constants": {f"{c}/{'tomorrow' if p else 'earlier'}/{k}": v for (c, p, k), v in median.items()},
              "v39_reference": ref_ret, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
