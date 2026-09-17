#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_full_recompute_replays_native pred_b_zeroing_8_1_is_live_in_two_constructions pred_c_native_cue_term_retains_most_everywhere pred_d_discovery_constants_transfer_everywhere pred_e_token_only_table_retains_half_pooled
"""Aspectual has/had definition-of-done battery, step 17: strong-form OOD for the closed 8.1 port.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v12/v15 (8.1's final-query service
= per-cue constant x lamb x v1(cue), constants transfer across the two discovery constructions).

WHY. better_circuits §1 PREDICTS OOD, strong form: "input is token IDs, not native states", on rows with
new templates. The 8.1 sub-component's final-query write is now a table on the cue token with ONE stored
constant per cue (the median native pattern over all 64 discovery rows, both constructions pooled --
2 numbers, frozen here). This run applies that table unchanged to the three template-varying
constructions of v5 (cue moved, no `last`, agent-first with comma-final, began/ended cue) and scores
what it retains of head 8.1's has/had contribution at the final query against zeroing the slice there.
The cue position in a template row is the token reading `since`/`by` (case-insensitive) that starts the
temporal phrase. No template row contributed to the constants.

PREDICTIONS (scored as written; failures preserved)
    pred_a_full_recompute_replays_native   full recomputation of the 8.1 slice replays native <= 1e-3
    pred_b_zeroing_8_1_is_live_in_two_constructions   zero-slice damage passes LIVE (>= 0.10, positive
                                           >= 0.75) in at least two of the three constructions
    pred_c_native_cue_term_retains_most_everywhere    cue-only inherited-only term with NATIVE pattern
                                           retains >= 0.60 in every live construction
    pred_d_discovery_constants_transfer_everywhere    the 2-constant table retains >= 0.50 in every live
                                           construction. Prior: unsure -- the pattern scalar may depend on
                                           the cue's distance to the query, which the templates change.
    pred_e_token_only_table_retains_half_pooled       pooled over live constructions >= 0.50

PRICE (registered maximum): 96 template rows in 3 batches; native 3 + producer 3 + zero 3 + discovery
constants fold 2 + 3 arms x (3 capture + 3 arm) = 18 -> 29 forwards; 0 backwards; 0 fits. Bar <= 34.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_token_only_ood_v17_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_token_only_ood_v17"
TOKENS = {"has": 468, "had": 550}
LIVE_FRACTION, LIVE_POSITIVE, RETAIN_C, RETAIN_D, POOLED_MIN, INSTRUMENT_TOL = 0.10, 0.75, 0.60, 0.50, 0.50, 1e-3
FORWARDS_MAX = 34
HEAD = L.Component("attn8_h1_final", 8, "attn", (1,), "final")


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "head": "8.1",
            "stored_constants": 2, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "retain_c": RETAIN_C,
                     "retain_d": RETAIN_D, "pooled_min": POOLED_MIN, "instrument_tol": INSTRUMENT_TOL}}


def cue_position(row):
    for s, tid in enumerate(row.ids):
        if L.ENCODING.decode([tid]).strip().lower() in ("since", "by"):
            return s
    raise RuntimeError(f"no cue token in {row.text!r}")


def main() -> None:
    discovery, rows = L.build_rows(), L.build_template_rows()
    if L.rows_sha256(discovery) != v1.EXPECTED_ROWS_SHA256 or L.rows_sha256(rows) != "dca4aa137b6add050fd694d496b40414b241ea86fe58086b8007bae3375917d3":
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, (HEAD,), TOKENS["has"], TOKENS["had"])
    unit = {1: fw.directions[(HEAD.name, 1)]}
    forwards = 0
    # frozen constants from the discovery rows (both constructions pooled), per cue
    values = {True: [], False: []}
    for start in range(0, len(discovery), v1.BATCH):
        chunk = discovery[start:start + v1.BATCH]
        out, lamb = L.head_source_terms_at(fw, chunk, 8, lambda r: r.final, unit); forwards += 1
        for row, entry in zip(chunk, out):
            values[row.present].append(entry[1]["pattern"][row.source_positions[0] - 1])
    constant = {present: statistics.median(v) for present, v in values.items()}

    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    zero, n = v1._run_arm(fw, rows, components=(HEAD,), mode="zero"); forwards += n
    is_cue = lambda r, s: s == cue_position(r)

    def arm_with(keep, branches, override=None):
        nonlocal forwards
        table = {}
        for start in range(0, len(rows), v1.BATCH):
            chunk = rows[start:start + v1.BATCH]
            table.update(L.source_restricted_slices(fw, chunk, HEAD, keep, branches, override)); forwards += 1
        fw.subtract = table
        arm, n = v1._run_arm(fw, rows, components=(HEAD,), mode="replace"); forwards += n
        return arm

    full = arm_with(lambda r, s: True, ("current", "inherited"))
    native_p = arm_with(is_cue, ("inherited",))
    const_p = arm_with(is_cue, ("inherited",), lambda r, s: constant[r.present] if is_cue(r, s) else None)
    fw.use_subtract = False
    instrument = max(max(abs(a["answer"] - b["answer"]), abs(a["foil"] - b["foil"])) for a, b in zip(full, native))

    constructions = sorted({r.construction for r in rows})
    report = {}
    for c in constructions:
        idx = [i for i, r in enumerate(rows) if r.construction == c]
        sub_rows, sub_native = [rows[i] for i in idx], [native[i] for i in idx]
        z = L.summarize(sub_rows, sub_native, [zero[i] for i in idx])
        def ret(arm):
            a = L.summarize(sub_rows, sub_native, [arm[i] for i in idx])["target_damage_mean"]
            return 1.0 - a / z["target_damage_mean"]
        report[c] = {"zero": z, "live": z["target_damage_fraction"] >= LIVE_FRACTION and z["target_damage_positive_fraction"] >= LIVE_POSITIVE,
                     "native_cue_retention": ret(native_p), "constant_cue_retention": ret(const_p)}
        print(c, "zero", round(z["target_damage_mean"], 4), "frac", round(z["target_damage_fraction"], 3), "pos", round(z["target_damage_positive_fraction"], 3),
              "native", round(report[c]["native_cue_retention"], 4), "constant", round(report[c]["constant_cue_retention"], 4))
    live = [c for c in constructions if report[c]["live"]]
    live_idx = [i for i, r in enumerate(rows) if r.construction in live]
    pooled = None
    if live_idx:
        sub_rows, sub_native = [rows[i] for i in live_idx], [native[i] for i in live_idx]
        z = L.summarize(sub_rows, sub_native, [zero[i] for i in live_idx])["target_damage_mean"]
        a = L.summarize(sub_rows, sub_native, [const_p[i] for i in live_idx])["target_damage_mean"]
        pooled = 1.0 - a / z
    predictions = {
        "pred_a_full_recompute_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_zeroing_8_1_is_live_in_two_constructions": len(live) >= 2,
        "pred_c_native_cue_term_retains_most_everywhere": bool(live) and all(report[c]["native_cue_retention"] >= RETAIN_C for c in live),
        "pred_d_discovery_constants_transfer_everywhere": bool(live) and all(report[c]["constant_cue_retention"] >= RETAIN_D for c in live),
        "pred_e_token_only_table_retains_half_pooled": pooled is not None and pooled >= POOLED_MIN,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_token_only_ood_result_v17", "candidate_id": CANDIDATE_ID, "plan": _plan(rows),
              "instrument_max_abs_error": instrument, "constants": {"since": constant[True], "by": constant[False]}, "lambda": lamb,
              "constructions": report, "live_constructions": live, "pooled_constant_retention": pooled,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "pooled": pooled, "constants": result["constants"], "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
