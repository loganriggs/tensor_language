#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_set_fraction_within_015_of_frozen pred_d_set_positive_selective_and_beats_null pred_e_keep_only_retains_most pred_f_set_is_additive
"""Narrative tense DoD battery, step 2 (v43): fresh-lexicon confirmation of the readout set with frozen numbers.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v42 (set {15.5, 11.3, 9.4, 9.1} selected by a
blind sweep; 74% of the margin on the selecting rows).

Frozen before access: F* = 0.74, band +-0.15 -> [0.59, 0.89] per construction. Rows: 16 places and 16 focus adjectives new to every corpus list and every panel of mine; 16 subjects new to this
behaviour (5 never used anywhere; 11 are lexicon-3 agents used only on the temporal line -- declared, since only five
further single-token occupational nouns exist outside the corpus lists); the line's two constructions. Arms: native; set removal + 16 norm-matched nulls + readers;
the four singles (additivity); zero the four slices; keep-only readout; 16 random keeps.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   <= 1e-4
    pred_b_native_capability           all four cells >= 0.85
    pred_c_set_fraction_within_015_of_frozen   |fraction - 0.74| <= 0.15 in each construction
    pred_d_set_positive_selective_and_beats_null   positive >= 0.75, > max null, three reader gates over the null (pooled)
    pred_e_keep_only_retains_most      keep-only retention >= 0.70; every random keep <= 0.30
    pred_f_set_is_additive             |joint - sum singles| <= 0.25 x min single

PRICE (registered maximum): 64 rows in 2 batches: native 2 + producer 2 + set 2 + nulls 32 + singles 8 + zero 2 + keep 2 +
random keeps 32 = 82 forwards; 0 backwards; 0 fits. Bar <= 90.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import circuit_fast_screen_candidates as lex
import run_aspectual_dod_removal_v1 as v1
import run_temporal_dod_removal_v28 as v28
import run_temporal_dod_frozen_prediction_v33 as v33
import run_narrative_dod_sweep_and_set_v42 as v42

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/narrative_tense_dod_confirm_v43_result.json"
CANDIDATE_ID = "narrative_tense.past_vs_present.dod_confirm_v43"
FROZEN, BAND = 0.74, 0.15
NULL_SEEDS, KEEP_SEEDS = tuple(range(1201, 1217)), tuple(range(1251, 1267))
CAPABILITY_MIN, LIVE_POSITIVE, GATE_RATIO, RETAIN_MIN, RANDOM_MAX, ADD_RATIO, INSTRUMENT_TOL = 0.85, 0.75, 0.25, 0.70, 0.30, 0.25, 1e-4
FORWARDS_MAX = 90
SUBJECTS = ('clown', 'spy', 'shepherd', 'brewer', 'fisherman', 'nun', 'cook', 'waiter', 'tutor', 'pastor', 'referee', 'witness', 'landlord', 'dealer', 'printer', 'hunter')
PLACES = ('palace', 'castle', 'temple', 'prison', 'hospital', 'library', 'museum', 'station', 'factory', 'school', 'church', 'cottage', 'quarry', 'pier', 'fountain', 'tunnel')
FOCUS = ("rugged", "sandy", "secret", "silent", "sturdy", "winding", "yellow", "golden", "tidy", "vast", "sunny", "grand", "ruined", "empty", "royal", "modern")
SINGLES = (L.Component("attn15_h5_final", 15, "attn", (5,), "final"), L.Component("attn11_h3_final", 11, "attn", (3,), "final"),
           L.Component("attn9_h4_final", 9, "attn", (4,), "final"), L.Component("attn9_h1_final", 9, "attn", (1,), "final"))
SET = (SINGLES[0], SINGLES[1], L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"))


def build():
    shared = (set(lex._OBJECTS) | {p[0] for p in lex._REPORTERS} | {p[1] for p in lex._REPORTERS} | set(lex._ADJECTIVES) | set(L.AGENTS) | set(L.PERIODS)
              | L._PRIOR_AGENTS | L._PRIOR_PERIODS | set(v28.PLACES) | set(v33.AGENTS) | set(v33.PLACES) | set(v42.FOCUS))
    words = set(SUBJECTS) | set(PLACES) | set(FOCUS)
    if words & shared:
        raise SystemExit(f"lexicon overlaps: {words & shared}")
    if len(set(SUBJECTS)) != 16 or len(set(PLACES)) != 16 or len(set(FOCUS)) != 16:
        raise SystemExit("need 16 of each")
    for w in words:
        L._single(" " + w)
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for construction in ("direct", "relative"):
        for g in range(16):
            subj, place, focus = SUBJECTS[g], PLACES[g], FOCUS[g]
            for past in (True, False):
                tail = f"The main reason for the {focus} {place}"
                if construction == "direct":
                    when, verb = ("Last winter", "stood") if past else ("Every winter", "stands")
                    text = f"{when} the {subj} {verb} nearby. {tail}"
                else:
                    verb, when, draw = ("stood", "last winter", "drew") if past else ("stands", "every winter", "draws")
                    text = f"The {subj} that {verb} nearby {when} {draw} crowds. {tail}"
                ids = L.ENCODING.encode(text); answer, foil = (" was", " is") if past else (" is", " was")
                a_id, f_id = L._single(answer), L._single(foil)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise SystemExit(f"joint tokenization failed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["narrative_confirm", construction, g, past, text]).encode()).hexdigest()[:24], construction, g, past, text, tuple(ids),
                                  answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows


def main() -> None:
    rows = build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "frozen": {"fraction": FROZEN, "band": BAND}, "forwards_max": FORWARDS_MAX,
                          "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, SET + SINGLES, v42.WAS, v42.IS)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    capability = {}
    for c in ("direct", "relative"):
        for past in (True, False):
            cell = [1.0 if native[i]["answer"] > native[i]["foil"] else 0.0 for i, r in enumerate(rows) if r.construction == c and r.present == past]
            capability[f"{c}/{'past' if past else 'present'}"] = sum(cell) / len(cell)
    joint, n = v1._run_arm(fw, rows, components=SET, mode="project"); forwards += n
    js = L.summarize(rows, native, joint)
    per_c = {c: L.summarize([r for r in rows if r.construction == c], [native[i] for i, r in enumerate(rows) if r.construction == c], [joint[i] for i, r in enumerate(rows) if r.construction == c])["target_damage_fraction"] for c in ("direct", "relative")}
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
    zero, n = v1._run_arm(fw, rows, components=SET, mode="zero"); forwards += n
    zd = L.summarize(rows, native, zero)["target_damage_mean"]
    keep, n = v1._run_arm(fw, rows, components=SET, mode="keep_only"); forwards += n
    retention = 1.0 - L.summarize(rows, native, keep)["target_damage_mean"] / zd
    random_ret = []
    for seed in KEEP_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=SET, mode="keep_only_random", seed=seed); forwards += n
        random_ret.append(1.0 - L.summarize(rows, native, arm)["target_damage_mean"] / zd)
    print("capability", capability, "fractions", {k: round(v, 3) for k, v in per_c.items()}, "joint", round(js["target_damage_mean"], 3), "pos", js["target_damage_positive_fraction"], "null_max", round(null_max, 4), "gates", gates)
    print("singles", {k: round(v, 3) for k, v in singles.items()}, "gap/bar", round(gap, 4), round(bar, 4), "keep retention", round(retention, 3), "random keep max", round(max(random_ret), 3))
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_native_capability": all(v >= CAPABILITY_MIN for v in capability.values()),
        "pred_c_set_fraction_within_015_of_frozen": all(abs(v - FROZEN) <= BAND for v in per_c.values()),
        "pred_d_set_positive_selective_and_beats_null": js["target_damage_positive_fraction"] >= LIVE_POSITIVE and js["target_damage_mean"] > null_max and all(gates.values()),
        "pred_e_keep_only_retains_most": retention >= RETAIN_MIN and all(r <= RANDOM_MAX for r in random_ret),
        "pred_f_set_is_additive": gap <= bar,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "narrative_tense_dod_confirm_result_v43", "candidate_id": CANDIDATE_ID, "rows_sha256": L.rows_sha256(rows), "instrument_max_abs_error": instrument, "capability": capability,
              "joint": js, "fractions": per_c, "null_damage_max": null_max, "gates": gates, "singles": singles, "additivity": {"gap": gap, "bar": bar},
              "keep_only": {"zero_damage": zd, "retention": retention, "random_retention": random_ret}, "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
