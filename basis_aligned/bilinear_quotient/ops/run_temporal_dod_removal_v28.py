#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_set_removal_live_and_beats_null pred_d_set_removal_selective pred_e_set_is_additive pred_f_head_11_3_leads_on_fresh_rows
"""Temporal will/had definition-of-done battery, step 1 (v28): weight-only readout removal on FRESH rows.

Lane: Claude circuit lane (redirect per review 3). Library: `aspectual_dod_lib.py` (shared). Parent: v27
(blind 162-head will/had readout sweep on the temporal line's authored rows: 11.3 1.48, 9.1 1.05, 15.5 0.41,
9.4 0.33, 8.1 0.15 logits).

WHY. Same battery as the aspectual component, first rung: the sweep-led set S = {11.3, 9.1, 15.5, 9.4}
(top four, registered from v27) removed along `O_h^T(u_will - u_had)` at the final query, on rows the sweep
never saw: a new 16-agent lexicon (lexicon 3) and 16 new place nouns in the line's two constructions
("Tomorrow/Earlier the A near the P"; "The reports say that tomorrow/earlier the A beside the P"). Null =
16 random 128-d directions of the removed norm per head; readers was-were, who-which, night-day.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   <= 1e-4
    pred_b_native_capability           all four (construction x cue) cells >= 0.85
    pred_c_set_removal_live_and_beats_null   damage fraction >= 0.10, positive >= 0.75, > max null
    pred_d_set_removal_selective       each unrelated reader mean|move| <= null mean + 0.25 x damage. Prior:
                                       unsure -- 11.3 is the subject-number head; was-were may move.
    pred_e_set_is_additive             |joint - sum(singles)| <= 0.25 x min(single damages)
    pred_f_head_11_3_leads_on_fresh_rows   11.3's single damage is the largest of the four

PRICE (registered maximum): 64 rows in 2 batches; native 2 + producer 2 + set 2 + 16 nulls x 2 + 4 singles
x 2 = 46 forwards; 0 backwards; 0 fits. Bar <= 52.
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
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_removal_v28_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_removal_v28"
WILL, HAD = 481, 550
NULL_SEEDS = tuple(range(801, 817))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, ADD_RATIO, CAPABILITY_MIN, INSTRUMENT_TOL = 0.10, 0.75, 0.25, 0.25, 0.85, 1e-4
FORWARDS_MAX = 52
AGENTS = ("nun", "cook", "waiter", "broker", "tutor", "pastor", "referee", "witness", "tenant", "landlord", "dealer", "printer", "author", "hunter", "knight", "wizard")
PLACES = ("lake", "hill", "gate", "barn", "mill", "dock", "cliff", "beach", "pond", "fence", "well", "shed", "ridge", "cave", "creek", "marsh")
CONSTRUCTIONS = {
    "bare_frame": (lambda p, a: f"Tomorrow the {a} near the {p}", lambda p, a: f"Earlier the {a} near the {p}"),
    "report_frame": (lambda p, a: f"The reports say that tomorrow the {a} beside the {p}", lambda p, a: f"The reports say that earlier the {a} beside the {p}"),
}
SINGLES = (L.Component("attn11_h3_final", 11, "attn", (3,), "final"), L.Component("attn9_h1_final", 9, "attn", (1,), "final"),
           L.Component("attn15_h5_final", 15, "attn", (5,), "final"), L.Component("attn9_h4_final", 9, "attn", (4,), "final"))
SET = (SINGLES[0], L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"), SINGLES[2])


def build():
    rows = L.build_rows_lexicon(AGENTS, PLACES, CONSTRUCTIONS, "temporal_fresh")
    # will/had answers instead of has/had: rebuild answer fields
    out = []
    for r in rows:
        present = r.present
        out.append(L.Row(r.row_id, r.construction, r.group, present, r.text, r.ids, " will" if present else " had", " had" if present else " will",
                         WILL if present else HAD, HAD if present else WILL, r.final, (), r.reader_ids))
    return out


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "set": [c.name for c in SET], "null_seeds": list(NULL_SEEDS),
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only", "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "gate_ratio_over_null": GATE_RATIO,
                                                                "add_ratio": ADD_RATIO, "capability_min": CAPABILITY_MIN, "instrument_tol": INSTRUMENT_TOL}}


def main() -> None:
    rows = build()
    for r in rows:
        if L.ENCODING.encode(r.text + r.answer) != list(r.ids) + [r.answer_id] or L.ENCODING.encode(r.text + r.foil) != list(r.ids) + [r.foil_id]:
            raise SystemExit(f"joint tokenization failed for {r.text!r}")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, SET + SINGLES, WILL, HAD)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    capability = {}
    for c in CONSTRUCTIONS:
        for present in (True, False):
            cell = [1.0 if native[i]["answer"] > native[i]["foil"] else 0.0 for i, r in enumerate(rows) if r.construction == c and r.present == present]
            capability[f"{c}/{'tomorrow' if present else 'earlier'}"] = sum(cell) / len(cell)
    joint, n = v1._run_arm(fw, rows, components=SET, mode="project"); forwards += n
    js = L.summarize(rows, native, joint)
    nulls = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=SET, mode="project_random", seed=seed); forwards += n
        nulls.append(L.summarize(rows, native, arm))
    null_max = max(s["target_damage_mean"] for s in nulls)
    null_moves = {name: sum(s[f"{name}_abs_move_mean"] for s in nulls) / len(nulls) for name in L.UNRELATED}
    gates = {name: js[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * js["target_damage_mean"] for name in L.UNRELATED}
    singles = {}
    for comp in SINGLES:
        arm, n = v1._run_arm(fw, rows, components=(comp,), mode="project"); forwards += n
        singles[comp.name] = L.summarize(rows, native, arm)["target_damage_mean"]
    gap = abs(js["target_damage_mean"] - sum(singles.values())); bar = ADD_RATIO * min(singles.values())
    print("joint", round(js["target_damage_mean"], 4), "fraction", round(js["target_damage_fraction"], 3), "pos", round(js["target_damage_positive_fraction"], 3),
          "null_max", round(null_max, 4), "gates", gates, "singles", {k: round(v, 3) for k, v in singles.items()}, "gap/bar", round(gap, 4), round(bar, 4))
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_native_capability": all(v >= CAPABILITY_MIN for v in capability.values()),
        "pred_c_set_removal_live_and_beats_null": js["target_damage_fraction"] >= LIVE_FRACTION and js["target_damage_positive_fraction"] >= LIVE_POSITIVE and js["target_damage_mean"] > null_max,
        "pred_d_set_removal_selective": all(gates.values()),
        "pred_e_set_is_additive": gap <= bar,
        "pred_f_head_11_3_leads_on_fresh_rows": max(singles, key=singles.get) == "attn11_h3_final",
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_removal_result_v28", "candidate_id": CANDIDATE_ID, "plan": _plan(rows), "instrument_max_abs_error": instrument,
              "capability": capability, "joint": js, "null_damage_max": null_max, "null_unrelated_abs_move_mean": null_moves, "gates": gates,
              "singles": singles, "additivity": {"gap": gap, "bar": bar}, "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "capability": capability, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
