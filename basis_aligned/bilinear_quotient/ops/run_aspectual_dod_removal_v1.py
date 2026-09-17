#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_at_least_one_component_is_live pred_d_live_components_beat_the_random_null pred_e_live_components_are_selective pred_f_head11_3_moves_the_number_reader pred_g_joint_removal_exceeds_best_single
"""Aspectual has/had definition-of-done battery, step 1: donor-free removal with a random null.

Lane: Claude circuit lane (board 2026-09-17T20:56Z). Library: `aspectual_dod_lib.py`.

WHAT IS MISSING. `better_circuits.md` §1 requires, for the SELECTIVE property, that removing a
component changes the target while >= 3 unrelated readers move less than a gate, on fresh rows,
against the null of removing an equal-norm random direction at the same site. Every aspectual
receipt so far is a paired base/donor interchange or a Shapley factorial of such interchanges;
none removes a component without a donor, none has a random-direction null, and the only
unrelated reader is the canonical C control. This run supplies all three at once for the five
components the released program names.

ROWS. 64 fresh rows (`aspectual_dod_lib.build_rows`): two validated constructions (fronted,
report-embedded) x 16 never-used single-token agent/period pairs x {since, by}. The lexicon is
disjoint from the discovery panel, the lexical-holdout v5 panel and the fresh-construction v2
panel. Rows are "fresh" relative to every selection step in the aspectual line; no row
selection uses any model score.

COMPONENTS (named by the v1-v12 releases; write removed at the position stated):
    mlp4_source_bank            MLP4 output at `last`, period noun, `the`
    attn5_h7_h1_h6_h8_final     block5 heads 7/1/6/8 pre-c_proj slices at the final query
    attn9_h1_h4_final           block9 heads 1/4 at the final query
    attn11_h3_final             block11 head 3 at the final query
    attn15_h5_final             block15 head 5 at the final query
    joint_all_five              all of the above in one forward

ARMS. native; each component zeroed; each component nulled by 16 random equal-norm directions
(seeds 1..16, per-row draws); the joint zero. Readers at the final position: target has-had
oriented toward the native answer; unrelated was-were, who-which, night-day.

REGISTERED BARS (before any model access)
    LIVE      target_damage_fraction >= 0.10 of the native mean margin AND damage positive on
              >= 0.75 of rows
    NULL      a live component's mean damage must exceed the MAXIMUM mean damage of its 16
              random-direction nulls
    GATE      each unrelated reader's mean |move| <= 0.25 x the component's mean target damage
    CAPABILITY native accuracy >= 0.85 in each of the four (construction x cue) cells

PREDICTIONS (scored exactly as written; failures preserved)
    pred_a_instrument_replays_native   no-removal forward matches producer.native answer/foil
                                       logits within 1e-4 on all 64 rows
    pred_b_native_capability           all four cells >= 0.85
    pred_c_at_least_one_component_is_live
                                       at least one single component passes LIVE. Stated prior:
                                       mlp4_source_bank and attn11_h3_final, because the released
                                       path explains ~27-33% of the donor effect and block11 H3 is
                                       its largest suffix amplifier.
    pred_d_live_components_beat_the_random_null
                                       every LIVE component passes NULL
    pred_e_live_components_are_selective
                                       every LIVE component passes GATE on all three readers
    pred_f_head11_3_moves_the_number_reader
                                       attn11_h3_final's was-were mean |move| > 0.25 x its target
                                       damage. Head 11.3 is the certified subject-number head
                                       (task14); I predict it FAILS the number gate, which would
                                       mark it as a shared component rather than an aspect-private
                                       one. If it passes, the head's has/had and was/were services
                                       are separable at this position.
    pred_g_joint_removal_exceeds_best_single
                                       joint_all_five mean damage > the best single mean damage

PRICE (registered maximum): 64 rows in batches of 32 -> 2 forwards per arm; arms = 1 native +
5 zero + 80 null + 1 joint = 87 arms = 174 forwards, plus 2 producer.native forwards for the
instrument check = 176 forwards; 0 backwards; 0 fits. Bar: <= 200 forwards.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_removal_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_removal_v1"
EXPECTED_ROWS_SHA256 = "5ec7d32f3f7b6bbc9ebc31de3411074b24a998430f7f929bd11d0813e40dfcd2"
BATCH = 32
NULL_SEEDS = tuple(range(1, 17))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, CAPABILITY_MIN, INSTRUMENT_TOL = 0.10, 0.75, 0.25, 0.85, 1e-4
FORWARDS_MAX = 200


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows),
            "components": [c.name for c in L.COMPONENTS], "null_seeds": list(NULL_SEEDS),
            "arms": 2 + len(L.COMPONENTS) * (1 + len(NULL_SEEDS)),
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only",
            "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE,
                     "gate_ratio": GATE_RATIO, "capability_min": CAPABILITY_MIN,
                     "instrument_tol": INSTRUMENT_TOL}}


def _batches(rows):
    for start in range(0, len(rows), BATCH):
        yield rows[start:start + BATCH]


def _run_arm(fw, rows, components=(), mode="zero", seed=0):
    out, forwards = [], 0
    for chunk in _batches(rows):
        out.extend(fw.forward(chunk, components=components, mode=mode, seed=seed))
        forwards += 1
    return out, forwards


def _producer_native(backend, rows):
    values, forwards = [], 0
    for chunk in _batches(rows):
        batch = producer.ModelBatch(
            row_ids=tuple(r.row_id for r in chunk), side="base",
            token_rows=tuple(tuple(r.ids) for r in chunk),
            semantic_positions=tuple(r.final for r in chunk),
            answer_ids=tuple(r.answer_id for r in chunk), foil_ids=tuple(r.foil_id for r in chunk))
        values.extend(backend.native(batch, capture=False).answer_foil)
        forwards += 1
    return values, forwards


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != EXPECTED_ROWS_SHA256:
        raise SystemExit("fresh rows changed; refusing to run against an unregistered panel")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    forwards = 0

    # instrument: the new forward must replay the producer's native logits
    native, n = _run_arm(fw, rows); forwards += n
    ref, n = _producer_native(backend, rows); forwards += n
    instrument_err = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    pred_a = instrument_err <= INSTRUMENT_TOL

    # capability by cell
    cells = {}
    for r, e in zip(rows, native):
        key = f"{r.construction}/{'since' if r.present else 'by'}"
        cells.setdefault(key, []).append(1.0 if e["answer"] > e["foil"] else 0.0)
    capability = {k: sum(v) / len(v) for k, v in cells.items()}
    pred_b = all(v >= CAPABILITY_MIN for v in capability.values())

    report = {}
    for comp in L.COMPONENTS:
        zero, n = _run_arm(fw, rows, components=(comp,), mode="zero"); forwards += n
        summary = L.summarize(rows, native, zero)
        nulls = []
        for seed in NULL_SEEDS:
            arm, n = _run_arm(fw, rows, components=(comp,), mode="random", seed=seed); forwards += n
            nulls.append(L.summarize(rows, native, arm))
        null_damages = [s["target_damage_mean"] for s in nulls]
        live = (summary["target_damage_fraction"] >= LIVE_FRACTION
                and summary["target_damage_positive_fraction"] >= LIVE_POSITIVE)
        beats_null = summary["target_damage_mean"] > max(null_damages)
        gates = {name: summary[f"{name}_abs_move_mean"] <= GATE_RATIO * summary["target_damage_mean"]
                 for name in L.UNRELATED}
        report[comp.name] = {"zero": summary, "live": live, "beats_null": beats_null,
                             "gates": gates, "selective": all(gates.values()),
                             "null_damage_means": null_damages,
                             "null_damage_max": max(null_damages),
                             "null_damage_median": sorted(null_damages)[len(null_damages) // 2],
                             "null_unrelated_abs_move_mean": {
                                 name: sum(s[f"{name}_abs_move_mean"] for s in nulls) / len(nulls)
                                 for name in L.UNRELATED}}
        print(comp.name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in summary.items()}),
              "live", live, "beats_null", beats_null, "gates", gates)
    joint, n = _run_arm(fw, rows, components=L.COMPONENTS, mode="zero"); forwards += n
    joint_summary = L.summarize(rows, native, joint)

    live_names = [k for k, v in report.items() if v["live"]]
    pred_c = bool(live_names)
    pred_d = bool(live_names) and all(report[k]["beats_null"] for k in live_names)
    pred_e = bool(live_names) and all(report[k]["selective"] for k in live_names)
    h3 = report["attn11_h3_final"]
    pred_f = h3["zero"]["number_was_were_abs_move_mean"] > GATE_RATIO * h3["zero"]["target_damage_mean"]
    best_single = max(v["zero"]["target_damage_mean"] for v in report.values())
    pred_g = joint_summary["target_damage_mean"] > best_single
    predictions = {
        "pred_a_instrument_replays_native": bool(pred_a),
        "pred_b_native_capability": bool(pred_b),
        "pred_c_at_least_one_component_is_live": bool(pred_c),
        "pred_d_live_components_beat_the_random_null": bool(pred_d),
        "pred_e_live_components_are_selective": bool(pred_e),
        "pred_f_head11_3_moves_the_number_reader": bool(pred_f),
        "pred_g_joint_removal_exceeds_best_single": bool(pred_g),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_removal_result_v1", "candidate_id": CANDIDATE_ID,
              "plan": _plan(rows), "instrument_max_abs_error": instrument_err,
              "capability": capability, "native_mean_margin": L.summarize(rows, native, native)["target_native_mean_margin"],
              "components": report, "joint_all_five": joint_summary, "live_components": live_names,
              "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "live": live_names, "forwards": forwards,
                      "seconds": result["serial_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
