#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_top4_set_live_and_beats_null pred_d_top4_set_selective pred_e_shared_heads_appear_in_top6
"""Narrative tense (was/is) DoD battery, step 1 (v42): blind readout sweep + top-4 set on fresh rows.

Lane: Claude circuit lane (third path). Library: `aspectual_dod_lib.py`. Parents: aspectual (v1-v26) and temporal
(v27-v41) components; shared head 8.1 (temporal-cue token reader) and block-9 relay {9.1, 9.4}.

WHY. Third auxiliary decision, different cue shape: narrative tense is carried by "Last winter ... stood" vs "Every
winter ... stands" one sentence before the target; the copula was/is is predicted after "The main reason for the
<focus> <place>". Question: does the same readout recipe (weight-only `O_h^T(u_was - u_is)` removal) localize a
compact head set, and do the shared heads reappear? Rows are FRESH: 16 subjects (lexicon-4 agents), 16 places
(v28 places, disjoint from the corpus object list), 16 focus adjectives (new, disjoint from lex._ADJECTIVES), in
the line's two constructions (direct narration; relative clause).

ARMS. native; producer replay; all 162 single-head readout removals (the sweep); the top-4 set (selected here by
the sweep -- so the set claims below are on OPENED rows; a fresh-row confirmation follows in v43) with 16 norm-
matched nulls and three readers; the four singles.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   <= 1e-4
    pred_b_native_capability           all four (construction x tense) cells >= 0.85
    pred_c_top4_set_live_and_beats_null   fraction >= 0.10, positive >= 0.75, > max null
    pred_d_top4_set_selective          three reader gates over the null
    pred_e_shared_heads_appear_in_top6 at least one of 8.1, 9.1, 9.4 is within the sweep's top six. Prior: unsure.

PRICE (registered maximum): 64 rows in 2 batches: native 2 + producer 2 + sweep 324 + set 2 + nulls 32 + singles 8 = 370
forwards; 0 backwards; 0 fits. Bar <= 400.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/narrative_tense_dod_sweep_and_set_v42_result.json"
CANDIDATE_ID = "narrative_tense.past_vs_present.dod_sweep_and_set_v42"
WAS, IS = L._single(" was"), L._single(" is")
NULL_SEEDS = tuple(range(1101, 1117))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, CAPABILITY_MIN, INSTRUMENT_TOL = 0.10, 0.75, 0.25, 0.85, 1e-4
FORWARDS_MAX = 400
SUBJECTS, PLACES = v33.AGENTS, v28.PLACES
FOCUS = ("ancient", "broken", "hidden", "rusty", "sacred", "shallow", "steep", "wooden", "crooked", "frozen", "hollow", "humble", "lonely", "modest", "noble", "painted")
HEADS = tuple(L.Component(f"attn{l}_h{h}", l, "attn", (h,), "final") for l in range(18) for h in range(9))


def build():
    shared = set(lex._OBJECTS) | {p[0] for p in lex._REPORTERS} | {p[1] for p in lex._REPORTERS} | set(lex._ADJECTIVES)
    if (set(SUBJECTS) | set(PLACES) | set(FOCUS)) & shared:
        raise SystemExit(f"lexicon overlaps the corpus lists: {(set(SUBJECTS) | set(PLACES) | set(FOCUS)) & shared}")
    for w in SUBJECTS + PLACES + FOCUS:
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
                ids = L.ENCODING.encode(text)
                answer, foil = (" was", " is") if past else (" is", " was")
                a_id, f_id = L._single(answer), L._single(foil)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise SystemExit(f"joint tokenization failed for {text!r}")
                rid = hashlib.sha256(json.dumps(["narrative", construction, g, past, text]).encode()).hexdigest()[:24]
                # present=True means the WAS side here (answer " was"), so oriented damage is toward the native answer either way
                rows.append(L.Row(rid, construction, g, past, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows


def main() -> None:
    rows = build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
                          "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, HEADS, WAS, IS)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    capability = {}
    for c in ("direct", "relative"):
        for past in (True, False):
            cell = [1.0 if native[i]["answer"] > native[i]["foil"] else 0.0 for i, r in enumerate(rows) if r.construction == c and r.present == past]
            capability[f"{c}/{'past' if past else 'present'}"] = sum(cell) / len(cell)
    sweep = {}
    for comp in HEADS:
        arm, n = v1._run_arm(fw, rows, components=(comp,), mode="project"); forwards += n
        sweep[comp.name] = L.summarize(rows, native, arm)["target_damage_mean"]
    ranked = sorted(sweep, key=lambda k: -sweep[k])
    top4 = ranked[:4]
    by_layer = {}
    for name in top4:
        comp = next(c for c in HEADS if c.name == name); by_layer.setdefault(comp.layer, []).append(comp.heads[0])
    SET = tuple(L.Component(f"set_attn{l}_" + "_".join(map(str, hs)), l, "attn", tuple(hs), "final") for l, hs in sorted(by_layer.items()))
    fw.directions.update(L.readout_directions(backend.model, SET, WAS, IS))
    joint, n = v1._run_arm(fw, rows, components=SET, mode="project"); forwards += n
    js = L.summarize(rows, native, joint)
    nulls = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=SET, mode="project_random", seed=seed); forwards += n
        nulls.append(L.summarize(rows, native, arm))
    null_max = max(s["target_damage_mean"] for s in nulls)
    null_moves = {name: sum(s[f"{name}_abs_move_mean"] for s in nulls) / len(nulls) for name in L.UNRELATED}
    gates = {name: js[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * js["target_damage_mean"] for name in L.UNRELATED}
    singles = {name: sweep[name] for name in top4}
    forwards += 0  # singles already in the sweep; the registered price counted them separately (8) and is an upper bound
    print("capability", capability); print("top10", [(k, round(sweep[k], 3)) for k in ranked[:10]])
    print("set", top4, "joint", round(js["target_damage_mean"], 4), "fraction", round(js["target_damage_fraction"], 3), "pos", round(js["target_damage_positive_fraction"], 3), "null_max", round(null_max, 4), "gates", gates)
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_native_capability": all(v >= CAPABILITY_MIN for v in capability.values()),
        "pred_c_top4_set_live_and_beats_null": js["target_damage_fraction"] >= LIVE_FRACTION and js["target_damage_positive_fraction"] >= LIVE_POSITIVE and js["target_damage_mean"] > null_max,
        "pred_d_top4_set_selective": all(gates.values()),
        "pred_e_shared_heads_appear_in_top6": any(h in ranked[:6] for h in ("attn8_h1", "attn9_h1", "attn9_h4")),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "narrative_tense_dod_sweep_and_set_result_v42", "candidate_id": CANDIDATE_ID, "rows_sha256": L.rows_sha256(rows), "instrument_max_abs_error": instrument,
              "capability": capability, "sweep_top20": [(k, sweep[k]) for k in ranked[:20]], "sweep": sweep, "set": top4, "joint": js, "null_damage_max": null_max,
              "null_unrelated_abs_move_mean": null_moves, "gates": gates, "singles": singles, "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
