#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_head_8_1_live_and_beats_null_everywhere pred_c_head_8_1_selective_everywhere pred_d_triple_live_null_selective_everywhere pred_e_triple_is_additive_with_block9_pair pred_f_triple_fraction_at_least_045_pooled
"""Aspectual has/had definition-of-done battery, step 8: promote the THREE-HEAD readout component.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v4/v5 (9.1+9.4 weight-only
readout removal: live, null-beating, selective, template-stable) and v7 (blind sweep: 8.1 is a
third live head, 0.34 logits, that the released path never named).

WHY. The component named by the v1-v12 releases at the final query was block9 H1/H4. The blind
sweep says the weight-only readout removal is carried by {8.1, 9.1, 9.4} and essentially nothing
else. Before the scorecard can call the three-head set the component, 8.1 must pass the same
battery as the pair: live, beats a norm-matched random null, selective on three unrelated
readers, on the discovery shape AND the three template-varying constructions; and the triple
must compose additively with the pair. Registered kill: 8.1 failing the was-were gate marks it a
shared subject-onset writer (as the temporal will/had line describes it) and keeps it out of the
aspect-private component.

ROWS: 64 discovery-shaped + 96 template rows (all opened for removal). COMPONENTS: {8.1} and
{8.1, 9.1, 9.4}; removal along each head's `O_h^T (u_has - u_had)` at the final input token; null
= random 128-d direction of equal removed norm per head, 16 seeds. Pair values come from the v4
(discovery) and v5 (templates) receipts, hash-read at run time.

BARS: LIVE / NULL / GATE as v2/v4/v5; ADD |triple - single(8.1) - pair| <= 0.25 x min(single, pair).

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native
    pred_b_head_8_1_live_and_beats_null_everywhere   in all four constructions
    pred_c_head_8_1_selective_everywhere             in all four constructions (prior: unsure)
    pred_d_triple_live_null_selective_everywhere     in all four constructions
    pred_e_triple_is_additive_with_block9_pair       ADD in all four constructions
    pred_f_triple_fraction_at_least_045_pooled       pooled triple damage fraction >= 0.45

PRICE (registered maximum): 160 rows in 5 batches; native 5 + producer 5 + 2 x 17 x 5 = 170 ->
180 forwards; 0 backwards; 0 fits. Bar <= 200.
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
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_triple_v8_result.json"
V4 = ROOT / "circuits/followups/aspectual_anchor_dod_readout_removal_v4_result.json"
V5 = ROOT / "circuits/followups/aspectual_anchor_dod_template_transfer_v5_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_triple_v8"
TOKENS = {"has": 468, "had": 550}
NULL_SEEDS = tuple(range(501, 517))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, ADD_RATIO, POOLED_MIN, INSTRUMENT_TOL = 0.10, 0.75, 0.25, 0.25, 0.45, 1e-4
FORWARDS_MAX = 200
SINGLE = L.Component("attn8_h1_final", 8, "attn", (1,), "final")
TRIPLE = (SINGLE, L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"))
DISCOVERY = "discovery"


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows),
            "components": ["attn8_h1", "attn8_h1+attn9_h1_h4"], "null_seeds": list(NULL_SEEDS),
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "gate_ratio_over_null": GATE_RATIO,
                     "add_ratio": ADD_RATIO, "pooled_min": POOLED_MIN, "instrument_tol": INSTRUMENT_TOL}}


def main() -> None:
    discovery, templates = L.build_rows(), L.build_template_rows()
    if L.rows_sha256(discovery) != v1.EXPECTED_ROWS_SHA256 or L.rows_sha256(templates) != "dca4aa137b6add050fd694d496b40414b241ea86fe58086b8007bae3375917d3":
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    rows = discovery + templates
    v4, v5 = json.loads(V4.read_text()), json.loads(V5.read_text())
    pair = {DISCOVERY: v4["components"]["attn9_h1_h4_final"]["project"]["target_damage_mean"]}
    for c, rep in v5["constructions"].items():
        pair[c] = rep["components"]["attn9_h1_h4_final"]["project"]["target_damage_mean"]
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

    arms = {}
    for label, comps in (("single", (SINGLE,)), ("triple", TRIPLE)):
        arm, n = v1._run_arm(fw, rows, components=comps, mode="project"); forwards += n
        nulls = []
        for seed in NULL_SEEDS:
            null_arm, n = v1._run_arm(fw, rows, components=comps, mode="project_random", seed=seed); forwards += n
            nulls.append(null_arm)
        arms[label] = (arm, nulls)

    def group(row):
        return DISCOVERY if row.construction in ("fronted", "report") else row.construction
    constructions = sorted({group(r) for r in rows})
    report = {}
    for c in constructions:
        idx = [i for i, r in enumerate(rows) if group(r) == c]
        sub_rows, sub_native = [rows[i] for i in idx], [native[i] for i in idx]
        report[c] = {}
        for label, (arm, nulls) in arms.items():
            s = L.summarize(sub_rows, sub_native, [arm[i] for i in idx])
            ns = [L.summarize(sub_rows, sub_native, [nl[i] for i in idx]) for nl in nulls]
            null_damages = [x["target_damage_mean"] for x in ns]
            null_moves = {name: sum(x[f"{name}_abs_move_mean"] for x in ns) / len(ns) for name in L.UNRELATED}
            live = s["target_damage_fraction"] >= LIVE_FRACTION and s["target_damage_positive_fraction"] >= LIVE_POSITIVE
            beats = s["target_damage_mean"] > max(null_damages)
            gates = {name: s[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * s["target_damage_mean"] for name in L.UNRELATED}
            report[c][label] = {"project": s, "live": live, "beats_null": beats, "gates": gates, "selective": all(gates.values()),
                                "null_damage_max": max(null_damages), "null_unrelated_abs_move_mean": null_moves}
            print(c, label, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}), "live", live, "beats", beats, "gates", gates)
        single_d, triple_d = report[c]["single"]["project"]["target_damage_mean"], report[c]["triple"]["project"]["target_damage_mean"]
        gap = abs(triple_d - single_d - pair[c])
        report[c]["additivity"] = {"triple": triple_d, "single_8_1": single_d, "pair_9_1_9_4": pair[c], "gap": gap,
                                   "bar": ADD_RATIO * min(single_d, pair[c]), "passes": gap <= ADD_RATIO * min(single_d, pair[c])}
    pooled_triple = L.summarize(rows, native, arms["triple"][0])
    predictions = {
        "pred_a_instrument_replays_native": bool(instrument_err <= INSTRUMENT_TOL),
        "pred_b_head_8_1_live_and_beats_null_everywhere": all(r["single"]["live"] and r["single"]["beats_null"] for r in report.values()),
        "pred_c_head_8_1_selective_everywhere": all(r["single"]["selective"] for r in report.values()),
        "pred_d_triple_live_null_selective_everywhere": all(r["triple"]["live"] and r["triple"]["beats_null"] and r["triple"]["selective"] for r in report.values()),
        "pred_e_triple_is_additive_with_block9_pair": all(r["additivity"]["passes"] for r in report.values()),
        "pred_f_triple_fraction_at_least_045_pooled": pooled_triple["target_damage_fraction"] >= POOLED_MIN,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_triple_result_v8", "candidate_id": CANDIDATE_ID, "plan": _plan(rows),
              "instrument_max_abs_error": instrument_err, "constructions": report, "pooled_triple": pooled_triple,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "pooled_triple_fraction": pooled_triple["target_damage_fraction"],
                      "additivity": {c: r["additivity"] for c, r in report.items()}, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
