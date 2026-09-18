#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_1030_alone_carries_at_most_020 pred_c_1030_congruent_positive pred_d_1030_plural_leaning pred_e_sum_of_singles_within_020_of_trio
"""Pronoun number they/he DoD (v240): the third MLP-8 number unit, 1030, alone on the natural rows -- the document's open item. v197: the trio {829, 953, 1030}
zeroed at the cue removes 3.3% of the congruent margin (0.1181 logits); v198: 829 alone 0.069 (plural-only), 953 alone 0.032 (singular-only), sum 0.101 -- leaving
0.017 for 1030 and interactions. v170: 1030 is a weak plural detector (36 / 40 pairs, output cosine +0.35 with 9.6's direction). Arms: 1030 alone at the cue, and the
three singles' sum against the trio.
PREDICTIONS (scored as written; failures preserved; priors from v170 / v198)
    pred_a_instrument_replays_native      hooked forward with no edit = producer native <= 1e-4
    pred_b_1030_alone_carries_at_most_020 1030 alone: congruent damage <= 0.20 x the trio's 0.1181
    pred_c_1030_congruent_positive        1030 alone: congruent damage > 0
    pred_d_1030_plural_leaning            1030 alone: damage on plural-they > damage on singular-he
    pred_e_sum_of_singles_within_020_of_trio  |0.069 + 0.032 + (1030 alone) - 0.1181| <= 0.20 x 0.1181 (the three units add; small interaction)
PRICE (registered maximum): 4 batches x (native + 1 arm) + producer replay 4 = 12 forwards; 0 backwards; 0 fits. Bar <= 14.
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
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp8_unit1030_natural_v240_result.json"
SOURCES = {"fineweb": ROOT / "circuits/followups/pronoun_number_dod_natural_rows_v77.json", "pile": ROOT / "circuits/followups/pronoun_number_dod_pile_rows_v78.json"}
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp8_unit1030_natural_v240"
LAYER, UNITS, CONGRUENT = 8, (829, 953, 1030), ("natural_plural_they", "natural_singular_he")
NULL_SEEDS = tuple(range(7901, 7917))
TRIO_DAMAGE, D829, D953, SHARE_MAX, SUM_TOL, INSTRUMENT_TOL = 0.1181, 0.0693, 0.0321, 0.20, 0.20, 1e-4
FORWARDS_MAX = 14
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_1030_alone_carries_at_most_020": "<= 0.20 x trio", "pred_c_1030_congruent_positive": "> 0", "pred_d_1030_plural_leaning": "plural > singular", "pred_e_sum_of_singles_within_020_of_trio": "<= 0.20 x trio"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"trio_damage": TRIO_DAMAGE, "d829": D829, "d953": D953, "share_max": SHARE_MAX, "sum_tol": SUM_TOL, "instrument_tol": INSTRUMENT_TOL}}
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
    arms = {"1030": arm((1030,))}
    for k, a in arms.items():
        print(k, "congruent", round(a["congruent"]["damage"], 4), round(a["congruent"]["fraction"], 4), "share of trio", round(a["congruent"]["damage"] / TRIO_DAMAGE, 3), "per cell", {c: (round(s_["damage"], 3), s_["positive"]) for c, s_ in a["per_cell"].items()})
    c = arms["1030"]["per_cell"]; d1030 = arms["1030"]["congruent"]["damage"]
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_1030_alone_carries_at_most_020": d1030 <= SHARE_MAX * TRIO_DAMAGE, "pred_c_1030_congruent_positive": d1030 > 0,
                   "pred_d_1030_plural_leaning": c["natural_plural_they"]["damage"] > c["natural_singular_he"]["damage"], "pred_e_sum_of_singles_within_020_of_trio": abs(D829 + D953 + d1030 - TRIO_DAMAGE) <= SUM_TOL * TRIO_DAMAGE}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp8_units_natural_result_v197", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "arms": arms,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
