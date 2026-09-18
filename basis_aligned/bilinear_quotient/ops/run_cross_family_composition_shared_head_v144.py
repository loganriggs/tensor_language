#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_own_sets_live_at_their_positions pred_c_other_family_removal_is_inert pred_d_joint_removal_is_additive_per_position
"""Cross-family COMPOSITION in one sentence with a SHARED HEAD (v144): a correlative decision at the object and a person decision at the end, whose sets both
contain head 8.1 (removed along or-but at the object and along myself-yourself at the end).

Rows: "{I|you} praised {either|not} the {obj}" scored at the object for or / but (row set A, 64 rows: 16 objects x 2 persons x 2 cues) and the same
sentence continued "... and then blamed" scored for myself / yourself (row set B, 64 rows). Objects fresh from `dod_lexicon.OBJECT_POOL` (the 16
after v143's); frames unused by either module. Sets: the person set {8.1, 13.1, 10.5, 15.1} along O_h^T(u_myself − u_yourself) and the correlative
set {16.8, 14.8, 7.8, 8.1} along O_h^T(u_or − u_but); the two block-8 components are named apart so head 8.1 carries both directions. Arms per row set: own set, other family's set, both. better_circuits §1 COMPOSES at the collection level: components of different
families in one sentence should act at their own positions, not at each other's, and jointly add.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native          <= 1e-4 on both row sets
    pred_b_own_sets_live_at_their_positions   own removal: fraction >= 0.10 and positive >= 0.75 on each row set (prior: unsure for the gender set
                                              after a correlative object -- the frames are new to it)
    pred_c_other_family_removal_is_inert      the other family's set removes <= 0.10 x the own set's damage, on each row set
    pred_d_joint_removal_is_additive_per_position   |both - own - other| <= 0.25 x own, on each row set

PRICE (registered maximum): 2 row sets x 2 batches x (native + producer + 3 arms) = 20 forwards; 0 backwards; 0 fits. Bar <= 24.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import dod_battery, dod_lexicon

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/cross_family_composition_shared_head_v144_result.json"
CANDIDATE_ID = "corpus.cross_family_composition_shared_head_v144"
GENDER_HEADS, CORR_HEADS = ((8, 1), (13, 1), (10, 5), (15, 1)), ((16, 8), (14, 8), (7, 8), (8, 1))   # GENDER_HEADS = the person set here (names kept from v143)
LIVE_FRACTION, LIVE_POSITIVE, INERT, ADD, INSTRUMENT_TOL = 0.10, 0.75, 0.10, 0.25, 1e-4
FORWARDS_MAX = 24
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_own_sets_live_at_their_positions": "live x 2", "pred_c_other_family_removal_is_inert": "<= 0.10 x own", "pred_d_joint_removal_is_additive_per_position": "<= 0.25 x own"}
L.READERS = {"tense_will_would": (" will", " would"), "animacy_who_which": (" who", " which"), "canonical_night_day": (" night", " day")}
L.UNRELATED = ("tense_will_would", "animacy_who_which", "canonical_night_day")


def build():
    objects = dod_lexicon.fresh(dod_lexicon.OBJECT_POOL, 32)[16:]   # the 16 after v143's
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    A, B = [], []
    for g in range(16):
        for gender in (True, False):   # here: first person (I) vs second (you)
            for cue in (True, False):
                noun = "I" if gender else "You"; obj = objects[g]
                textA = f"{noun} praised {'either' if cue else 'not'} the {obj}"; textB = textA + " and then blamed"
                for rows, text, present, ans, foil, tag in ((A, textA, cue, " or", " but", "corr"), (B, textB, gender, " myself", " yourself", "person")):
                    ids = L.ENCODING.encode(text); a, f = (ans, foil) if present else (foil, ans); a_id, f_id = L._single(a), L._single(f)
                    if L.ENCODING.encode(text + a) != ids + [a_id] or L.ENCODING.encode(text + f) != ids + [f_id]:
                        raise L.RowError(f"joint tokenization changed for {text!r}")
                    rows.append(L.Row(hashlib.sha256(json.dumps(["xfam_v144", tag, g, gender, cue, text]).encode()).hexdigest()[:24], f"{tag}_{'either' if cue else 'not'}", g, present, text, tuple(ids), a, f, a_id, f_id, len(ids) - 1, (), reader_ids))
    return A, B


def main() -> None:
    A, B = build()
    plan = {"candidate_id": CANDIDATE_ID, "rows": {"A_correlative": len(A), "B_person": len(B)}, "rows_sha256": {"A": L.rows_sha256(A), "B": L.rows_sha256(B)}, "person_heads": list(GENDER_HEADS), "correlative_heads": list(CORR_HEADS),
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "inert": INERT, "add": ADD}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    he, she, or_, but = L._single(" myself"), L._single(" yourself"), L._single(" or"), L._single(" but")
    gset = dod_battery.LineSpec(CANDIDATE_ID, OUT.name, A, he, she, GENDER_HEADS).set_components()
    cset = tuple(L.Component("corr_" + c.name, c.layer, c.kind, c.heads, c.where) for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, A, or_, but, CORR_HEADS).set_components())
    fw.directions = {**L.readout_directions(backend.model, gset, he, she), **L.readout_directions(backend.model, cset, or_, but)}
    forwards, report, instrument = 0, {}, 0.0
    for name, rows, own, other in (("A_correlative", A, cset, gset), ("B_person", B, gset, cset)):
        native, n = v1._run_arm(fw, rows); forwards += n
        ref, n = v1._producer_native(backend, rows); forwards += n
        instrument = max(instrument, max(max(abs(x["answer"] - r[0]), abs(x["foil"] - r[1])) for x, r in zip(native, ref)))
        arms = {}
        for label, comps in (("own", own), ("other", other), ("both", own + other)):
            arm, n = v1._run_arm(fw, rows, components=comps, mode="project"); forwards += n
            arms[label] = L.summarize(rows, native, arm)
        d = {k: v["target_damage_mean"] for k, v in arms.items()}
        report[name] = {"arms": arms, "own_live": arms["own"]["target_damage_fraction"] >= LIVE_FRACTION and arms["own"]["target_damage_positive_fraction"] >= LIVE_POSITIVE,
                        "other_ratio": d["other"] / d["own"] if d["own"] else None, "additivity_gap": abs(d["both"] - d["own"] - d["other"]), "additivity_bar": ADD * d["own"]}
        print(name, {k: (round(v["target_damage_mean"], 3), round(v["target_damage_fraction"], 3), round(v["target_damage_positive_fraction"], 3)) for k, v in arms.items()}, "other/own", round(report[name]["other_ratio"], 3), "gap", round(report[name]["additivity_gap"], 3))
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_own_sets_live_at_their_positions": all(r["own_live"] for r in report.values()),
                   "pred_c_other_family_removal_is_inert": all(r["other_ratio"] is not None and abs(r["other_ratio"]) <= INERT for r in report.values()),
                   "pred_d_joint_removal_is_additive_per_position": all(r["additivity_gap"] <= r["additivity_bar"] for r in report.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "cross_family_composition_result_v143", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
