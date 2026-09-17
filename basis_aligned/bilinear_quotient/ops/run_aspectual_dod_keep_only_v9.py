#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_zeroing_the_three_heads_is_live pred_c_keep_only_readout_retains_most pred_d_keep_only_random_retains_little pred_e_keep_only_readout_beats_every_random_keep pred_f_retention_holds_in_every_construction
"""Aspectual has/had definition-of-done battery, step 9: EXTRACTION at the component's own boundary.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v8 (three-head readout
component {8.1, 9.1, 9.4}: removal along `O_h^T (u_has - u_had)` is live, null-beating, selective,
additive, template-stable).

WHY. better_circuits §1 EXTRACTED: the component "runs standalone with declared inputs". The
declared input of this component is three 128-d head slices at the final query; its declared
computation is one scalar per head, `c_h = w_h . v_h`, written back as `c_h v_h` through the
native `c_proj`. Removal (v4-v8) showed that direction is NECESSARY. This run tests SUFFICIENCY
at the same boundary: replace each of the three head slices by ONLY its readout projection
(`keep_only`: everything the head writes off that direction is discarded) and ask how much of
the three heads' has/had contribution survives. Null: keep only a random unit direction of the
same slice (16 seeds). Reference: zero the three slices entirely (the heads' whole contribution).
Retention = 1 - damage(arm) / damage(zero); a retention near 1 means the one direction per head
carries the heads' aspect service; the random keep prices what any single direction retains.
Unrelated readers are reported but not gated here: discarding the complement is a large
off-target edit by construction and the selectivity claim was settled in v8.

ROWS: 64 discovery-shaped + 96 template rows (opened for removal). Scoring per construction and
pooled.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native
    pred_b_zeroing_the_three_heads_is_live      zero damage >= 0.10 fraction, positive >= 0.75 (pooled)
    pred_c_keep_only_readout_retains_most       pooled retention >= 0.70
    pred_d_keep_only_random_retains_little      pooled retention of the random keep <= 0.30 for
                                                every one of the 16 seeds
    pred_e_keep_only_readout_beats_every_random_keep   readout retention > max random retention
    pred_f_retention_holds_in_every_construction       readout retention >= 0.50 in each of the four
                                                constructions. Prior: unsure -- the head slices
                                                may carry aspect content off the readout direction
                                                that MLPs 9-15 read (v6 showed a 43% MLP cascade).

PRICE (registered maximum): 160 rows in 5 batches; native 5 + producer 5 + zero 5 + keep 5 +
16 random keeps x 5 = 100 forwards; 0 backwards; 0 fits. Bar <= 120.
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
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_keep_only_v9_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_keep_only_v9"
TOKENS = {"has": 468, "had": 550}
NULL_SEEDS = tuple(range(601, 617))
LIVE_FRACTION, LIVE_POSITIVE, RETAIN_MIN, RANDOM_MAX, PER_CONSTRUCTION_MIN, INSTRUMENT_TOL = 0.10, 0.75, 0.70, 0.30, 0.50, 1e-4
FORWARDS_MAX = 120
TRIPLE = (L.Component("attn8_h1_final", 8, "attn", (1,), "final"),
          L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"))
DISCOVERY = "discovery"


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows),
            "component": "attn8_h1+attn9_h1_h4", "null_seeds": list(NULL_SEEDS), "forwards_max": FORWARDS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "retain_min": RETAIN_MIN,
                     "random_max": RANDOM_MAX, "per_construction_min": PER_CONSTRUCTION_MIN, "instrument_tol": INSTRUMENT_TOL}}


def main() -> None:
    discovery, templates = L.build_rows(), L.build_template_rows()
    if L.rows_sha256(discovery) != v1.EXPECTED_ROWS_SHA256 or L.rows_sha256(templates) != "dca4aa137b6add050fd694d496b40414b241ea86fe58086b8007bae3375917d3":
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    rows = discovery + templates
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, TRIPLE, TOKENS["has"], TOKENS["had"])
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument_err = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    zero, n = v1._run_arm(fw, rows, components=TRIPLE, mode="zero"); forwards += n
    keep, n = v1._run_arm(fw, rows, components=TRIPLE, mode="keep_only"); forwards += n
    randoms = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=TRIPLE, mode="keep_only_random", seed=seed); forwards += n
        randoms.append(arm)

    def group(row):
        return DISCOVERY if row.construction in ("fronted", "report") else row.construction

    def score(idx):
        sub_rows, sub_native = [rows[i] for i in idx], [native[i] for i in idx]
        z = L.summarize(sub_rows, sub_native, [zero[i] for i in idx])
        k = L.summarize(sub_rows, sub_native, [keep[i] for i in idx])
        rs = [L.summarize(sub_rows, sub_native, [r[i] for i in idx]) for r in randoms]
        zd = z["target_damage_mean"]
        retention = 1.0 - k["target_damage_mean"] / zd if zd else None
        random_retention = [1.0 - r["target_damage_mean"] / zd if zd else None for r in rs]
        return {"zero": z, "keep_only_readout": k, "retention": retention,
                "random_retention": random_retention, "random_retention_max": max(random_retention),
                "random_retention_median": sorted(random_retention)[len(random_retention) // 2],
                "random_keep_damage_mean": sum(r["target_damage_mean"] for r in rs) / len(rs)}

    constructions = sorted({group(r) for r in rows})
    report = {c: score([i for i, r in enumerate(rows) if group(r) == c]) for c in constructions}
    pooled = score(list(range(len(rows))))
    for c, s in list(report.items()) + [("pooled", pooled)]:
        print(c, "zero", round(s["zero"]["target_damage_mean"], 4), "keep", round(s["keep_only_readout"]["target_damage_mean"], 4),
              "retention", round(s["retention"], 4), "random max/med", round(s["random_retention_max"], 4), round(s["random_retention_median"], 4))
    predictions = {
        "pred_a_instrument_replays_native": bool(instrument_err <= INSTRUMENT_TOL),
        "pred_b_zeroing_the_three_heads_is_live": pooled["zero"]["target_damage_fraction"] >= LIVE_FRACTION and pooled["zero"]["target_damage_positive_fraction"] >= LIVE_POSITIVE,
        "pred_c_keep_only_readout_retains_most": pooled["retention"] >= RETAIN_MIN,
        "pred_d_keep_only_random_retains_little": all(r <= RANDOM_MAX for r in pooled["random_retention"]),
        "pred_e_keep_only_readout_beats_every_random_keep": pooled["retention"] > pooled["random_retention_max"],
        "pred_f_retention_holds_in_every_construction": all(s["retention"] >= PER_CONSTRUCTION_MIN for s in report.values()),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_keep_only_result_v9", "candidate_id": CANDIDATE_ID, "plan": _plan(rows),
              "instrument_max_abs_error": instrument_err, "constructions": report, "pooled": pooled,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "pooled_retention": pooled["retention"],
                      "random_retention_max": pooled["random_retention_max"], "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
