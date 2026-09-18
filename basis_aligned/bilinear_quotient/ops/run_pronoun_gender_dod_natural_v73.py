#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_congruent_native_capability pred_c_congruent_removal_shifts_away_from_label pred_d_congruent_removal_beats_null pred_e_congruent_removal_selective pred_f_incongruent_removal_shifts_toward_label
"""Pronoun gender he/she DoD (v73): the set {10.1, 9.6, 12.4, 15.1} on NATURAL FineWeb rows (PREDICTS OOD, better_circuits §1).

Lane: Claude circuit lane. Parents: v71 (fresh synthetic rows, 0.96-1.11 of the margin), v72 (random-set null). Rows:
`pronoun_gender_dod_natural_rows.py` (outcome-blind miner, receipt `circuits/followups/pronoun_gender_dod_natural_rows_v73.json`,
64 rows: 16 per (noun gender x next token) cell; the noun filter is any-sense -- 'female dorms', 'groom a successor' pass it --
so the congruent cells are the readout test and the incongruent cells are natural counter-cases, both reported as mined).
FineWeb is the training corpus (in-distribution text, out-of-panel rows). Removal along `O_h^T (u_he - u_she)` at the position
before the pronoun; null = norm-matched random direction x 16 seeds; readers was/were, who/which, night/day. Damage is the
oriented drop of (actual next token - other pronoun); a NEGATIVE damage on incongruent rows means the removal moved the
model toward what the text actually says and away from the noun's gender.

PREDICTIONS (scored as written; failures preserved; bars frozen from the aspectual natural line v20)
    pred_a_instrument_replays_native                 <= 1e-4
    pred_b_congruent_native_capability               >= 0.75 correct in each congruent cell (male/he, female/she)
    pred_c_congruent_removal_shifts_away_from_label  congruent rows pooled: mean damage >= 0.15 logits and >= 60% rows positive
    pred_d_congruent_removal_beats_null              congruent mean damage > max of the 16 null means
    pred_e_congruent_removal_selective               three reader gates on congruent rows (null + 0.25 x damage)
    pred_f_incongruent_removal_shifts_toward_label   incongruent rows pooled: mean damage <= 0. Prior: unsure.

PRICE (registered maximum): 2 batches x (native + producer + set + 16 nulls) = 38 forwards; 0 backwards; 0 fits. Bar <= 40.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_pronoun_gender_dod_battery_v71 as v71
import dod_battery

ROOT = Path(__file__).resolve().parent.parent
ROWS = ROOT / "circuits/followups/pronoun_gender_dod_natural_rows_v73.json"
OUT = ROOT / "circuits/followups/pronoun_gender_dod_natural_v73_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_natural_v73"
NULL_SEEDS = tuple(range(3101, 3117))
CAPABILITY_MIN, SHIFT_MIN, DIRECTION_MIN, GATE_RATIO, INSTRUMENT_TOL = 0.75, 0.15, 0.60, 0.25, 1e-4
FORWARDS_MAX = 40
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_congruent_native_capability": ">= 0.75 per congruent cell",
               "pred_c_congruent_removal_shifts_away_from_label": "mean >= 0.15, >= 60% positive", "pred_d_congruent_removal_beats_null": "> null max",
               "pred_e_congruent_removal_selective": "three gates", "pred_f_incongruent_removal_shifts_toward_label": "mean <= 0"}


def load_rows():
    doc = json.loads(ROWS.read_text())
    if hashlib.sha256(json.dumps(doc["rows"], sort_keys=True).encode()).hexdigest() != doc["rows_sha256"]:
        raise SystemExit("natural rows changed")
    he, she = L._single(" he"), L._single(" she")
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for r in doc["rows"]:
        ids = tuple(r["ids"]); present = r["label"] == "he"
        rows.append(L.Row(hashlib.sha256(json.dumps([r["doc_index"], r["position"]]).encode()).hexdigest()[:24], f"natural_{r['gender']}_{r['label']}",
                          r["doc_index"], present, r["text"], ids, " he" if present else " she", " she" if present else " he",
                          he if present else she, she if present else he, len(ids) - 1, (), reader_ids))
    return rows, doc["rows_sha256"], he, she


def main() -> None:
    rows, sha, he, she = load_rows()
    SET = dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, v71.HEADS).set_components()
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": sha, "set": list(v71.HEADS), "null_seeds": list(NULL_SEEDS), "forwards_max": FORWARDS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"capability_min": CAPABILITY_MIN, "shift_min": SHIFT_MIN, "direction_min": DIRECTION_MIN, "gate_ratio": GATE_RATIO}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, SET, he, she)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    arm, n = v1._run_arm(fw, rows, components=SET, mode="project"); forwards += n
    nulls = []
    for seed in NULL_SEEDS:
        na, n = v1._run_arm(fw, rows, components=SET, mode="project_random", seed=seed); forwards += n; nulls.append(na)
    congruent = [i for i, r in enumerate(rows) if r.construction in ("natural_male_he", "natural_female_she")]
    incongruent = [i for i, r in enumerate(rows) if i not in congruent]

    def score(idx):
        sub = [rows[i] for i in idx]; nat = [native[i] for i in idx]
        s = L.summarize(sub, nat, [arm[i] for i in idx])
        ns = [L.summarize(sub, nat, [na[i] for i in idx]) for na in nulls]
        null_max = max(x["target_damage_mean"] for x in ns)
        null_moves = {name: sum(x[f"{name}_abs_move_mean"] for x in ns) / len(ns) for name in L.UNRELATED}
        gates = {name: s[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * s["target_damage_mean"] for name in L.UNRELATED}
        return {"n": len(idx), "removal": s, "null_damage_max": null_max, "null_unrelated_abs_move_mean": null_moves, "gates": gates}

    capability = {}
    for c in sorted({r.construction for r in rows}):
        cell = [1.0 if native[i]["answer"] > native[i]["foil"] else 0.0 for i, r in enumerate(rows) if r.construction == c]
        capability[c] = sum(cell) / len(cell)
    report = {"congruent": score(congruent), "incongruent": score(incongruent), "by_cell": {c: score([i for i, r in enumerate(rows) if r.construction == c]) for c in sorted({r.construction for r in rows})}}
    C, I = report["congruent"]["removal"], report["incongruent"]["removal"]
    print("capability", capability); print("congruent", round(C["target_damage_mean"], 3), "positive", C["target_damage_positive_fraction"], "null max", round(report["congruent"]["null_damage_max"], 3), "gates", report["congruent"]["gates"])
    print("incongruent", round(I["target_damage_mean"], 3), "positive", I["target_damage_positive_fraction"])
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
                   "pred_b_congruent_native_capability": capability["natural_male_he"] >= CAPABILITY_MIN and capability["natural_female_she"] >= CAPABILITY_MIN,
                   "pred_c_congruent_removal_shifts_away_from_label": C["target_damage_mean"] >= SHIFT_MIN and C["target_damage_positive_fraction"] >= DIRECTION_MIN,
                   "pred_d_congruent_removal_beats_null": C["target_damage_mean"] > report["congruent"]["null_damage_max"],
                   "pred_e_congruent_removal_selective": all(report["congruent"]["gates"].values()),
                   "pred_f_incongruent_removal_shifts_toward_label": I["target_damage_mean"] <= 0.0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_gender_dod_natural_result_v73", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "capability": capability,
                               "report": report, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
