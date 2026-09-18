#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_829_alone_carries_070 pred_c_953_alone_carries_at_most_030 pred_d_829_is_plural_specific pred_e_953_is_singular_specific
"""Pronoun number they/he DoD (v198): which of the MLP-8 number units does the natural-text work? v197: the trio {829, 953, 1030} zeroed at the cue of the
128 natural rows removes 3.3% of the congruent margin, but per cell plural-they lost 5.1% and singular-he only 0.7%. Arms: 829 alone (the plural detector,
v170) and 953 alone (the singular detector) at the cue; same rows, same readers, native and producer replay as in v197. No new null (v197's random
3-unit sets: <= 0.0005); the trio's congruent damage 0.1181 is the reference.
PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   hooked forward with no edit = producer native <= 1e-4
    pred_b_829_alone_carries_070       829 alone: congruent damage >= 0.70 x the trio's 0.1181
    pred_c_953_alone_carries_at_most_030  953 alone: congruent damage <= 0.30 x the trio's
    pred_d_829_is_plural_specific      829 alone: damage on plural-they >= 3 x damage on singular-he, and plural-he shifts toward the label (< 0)
    pred_e_953_is_singular_specific    953 alone: damage on singular-he > 0 and on plural-they <= 0.5 x its singular-he damage. Prior: unsure (its share is small).
PRICE (registered maximum): 4 batches x (native + 2 arms) + producer replay 4 = 16 forwards; 0 backwards; 0 fits. Bar <= 20.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import dod_battery, dod_natural_line as N, dod_units

L.READERS = {"tense_will_would": (" will", " would"), "animacy_who_which": (" who", " which"), "canonical_night_day": (" night", " day")}
L.UNRELATED = ("tense_will_would", "animacy_who_which", "canonical_night_day")

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp8_single_units_natural_v198_result.json"
SOURCES = {"fineweb": ROOT / "circuits/followups/pronoun_number_dod_natural_rows_v77.json", "pile": ROOT / "circuits/followups/pronoun_number_dod_pile_rows_v78.json"}
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp8_single_units_natural_v198"
LAYER, UNITS, CONGRUENT = 8, (829, 953, 1030), ("natural_plural_they", "natural_singular_he")
NULL_SEEDS = tuple(range(7901, 7917))
TRIO_DAMAGE, SHARE_829_MIN, SHARE_953_MAX, PLURAL_RATIO, INSTRUMENT_TOL = 0.1181, 0.70, 0.30, 3.0, 1e-4
FORWARDS_MAX = 20
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_829_alone_carries_070": ">= 0.70 x trio", "pred_c_953_alone_carries_at_most_030": "<= 0.30 x trio", "pred_d_829_is_plural_specific": "plural >= 3x singular, plural-he < 0", "pred_e_953_is_singular_specific": "singular > 0, plural <= 0.5x"}


def main() -> None:
    rows, cue, shas = [], {}, {}
    for name, path in SOURCES.items():
        rs, sha, they, he = N.rows_from_receipt(path, "they", " they", " he", lambda r: f"natural_{r['number']}_{r['label']}")
        recs = json.loads(path.read_text())["rows"]
        for row, rec in zip(rs, recs):
            if tuple(rec["ids"]) != row.ids: raise SystemExit("row order mismatch")
            cue[row.row_id] = rec["cue_offset"]
        rows.extend(rs); shas[name] = sha
    cue_of = lambda row: cue[row.row_id]
    readers = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": shas, "units": list(UNITS), "congruent": list(CONGRUENT), "null_seeds": list(NULL_SEEDS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"trio_damage": TRIO_DAMAGE, "share_829_min": SHARE_829_MIN, "share_953_max": SHARE_953_MAX, "plural_ratio": PLURAL_RATIO, "instrument_tol": INSTRUMENT_TOL}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    fw = L.ManualForward(backend); forwards = 0
    def run(edits):
        nonlocal forwards; out = []
        for start in range(0, len(rows), v1.BATCH):
            out.extend(dod_units.forward_margins(backend, fw, rows[start:start + v1.BATCH], LAYER, edits, readers)); forwards += 1
        return out
    native = run(None)
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    cong = [i for i, r in enumerate(rows) if r.construction in CONGRUENT]; incong = [i for i, r in enumerate(rows) if r.construction not in CONGRUENT]
    def summarize(arm, idx):
        d = [(native[i]["answer"] - native[i]["foil"]) - (arm[i]["answer"] - arm[i]["foil"]) for i in idx]
        m = sum(d) / len(d); nm = sum(native[i]["answer"] - native[i]["foil"] for i in idx) / len(idx)
        return {"damage": m, "native_margin": nm, "fraction": m / nm, "positive": sum(1 for x in d if x > 0) / len(d), **{f"{k}_abs_move": sum(abs(arm[i][k] - native[i][k]) for i in idx) / len(idx) for k in readers}}
    cells = sorted({r.construction for r in rows})
    def arm(units):
        out = run((units, cue_of))
        return {"congruent": summarize(out, cong), "incongruent": summarize(out, incong), "per_cell": {c: summarize(out, [i for i, r in enumerate(rows) if r.construction == c]) for c in cells}}
    arms = {"829": arm((829,)), "953": arm((953,))}
    for k, a in arms.items():
        print(k, "congruent", round(a["congruent"]["damage"], 4), round(a["congruent"]["fraction"], 4), "share of trio", round(a["congruent"]["damage"] / TRIO_DAMAGE, 3), "per cell", {c: (round(s_["damage"], 3), s_["positive"]) for c, s_ in a["per_cell"].items()})
    c829, c953 = arms["829"]["per_cell"], arms["953"]["per_cell"]
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_829_alone_carries_070": arms["829"]["congruent"]["damage"] >= SHARE_829_MIN * TRIO_DAMAGE, "pred_c_953_alone_carries_at_most_030": arms["953"]["congruent"]["damage"] <= SHARE_953_MAX * TRIO_DAMAGE,
                   "pred_d_829_is_plural_specific": c829["natural_plural_they"]["damage"] >= PLURAL_RATIO * c829["natural_singular_he"]["damage"] and c829["natural_plural_he"]["damage"] < 0,
                   "pred_e_953_is_singular_specific": c953["natural_singular_he"]["damage"] > 0 and c953["natural_plural_they"]["damage"] <= 0.5 * c953["natural_singular_he"]["damage"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp8_units_natural_result_v197", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "arms": arms,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
