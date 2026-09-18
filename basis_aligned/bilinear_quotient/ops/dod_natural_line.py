#!/usr/bin/env python3
# BQGATE: LIBRARY -- generic natural-row (FineWeb / Pile) removal runner for one readout set (review-7 efficiency item).
"""`run(spec, rows_path, out_name, congruent_constructions, candidate_id)`: the v73/v75 body for any line. Rows come from a
miner receipt (`{"rows": [{doc_index, position, label, ids, text, ...}], "rows_sha256"}`) turned into L.Row by `rows_from_receipt`;
the caller names which constructions are the congruent (readout-test) cells; the rest are counter-cases. Arms: native + producer
(instrument), set removal along weight-only readout directions, 16 norm-matched nulls, three readers. Bars frozen from the
aspectual natural line v20: shift >= 0.15, >= 60% positive, > null max, three gates on congruent rows; counter-case mean <= 0.

GATE NOTE: the RUNNER file must carry a literal `PREDICTIONS = {...}` dict with the keys scored here (see gate.py)."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1

ROOT = Path(__file__).resolve().parent.parent
NULL_SEEDS = tuple(range(3101, 3117))
BARS = {"capability_min": 0.75, "shift_min": 0.15, "direction_min": 0.60, "gate_ratio": 0.25, "instrument_tol": 1e-4}
PREDICTION_KEYS = ("pred_a_instrument_replays_native", "pred_b_congruent_native_capability", "pred_c_congruent_removal_shifts_away_from_label",
                   "pred_d_congruent_removal_beats_null", "pred_e_congruent_removal_selective", "pred_f_incongruent_removal_shifts_toward_label")


def rows_from_receipt(path: Path, positive_label: str, positive_token: str, negative_token: str, construction_of):
    """construction_of(record) -> construction name; present = record['label'] == positive_label."""
    doc = json.loads(path.read_text())
    if hashlib.sha256(json.dumps(doc["rows"], sort_keys=True).encode()).hexdigest() != doc["rows_sha256"]:
        raise SystemExit("natural rows changed")
    pos, neg = L._single(positive_token), L._single(negative_token)
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for r in doc["rows"]:
        ids = tuple(r["ids"]); present = r["label"] == positive_label
        rows.append(L.Row(hashlib.sha256(json.dumps([r["doc_index"], r["position"]]).encode()).hexdigest()[:24], construction_of(r), r["doc_index"], present,
                          r["text"], ids, positive_token if present else negative_token, negative_token if present else positive_token,
                          pos if present else neg, neg if present else pos, len(ids) - 1, (), reader_ids))
    return rows, doc["rows_sha256"], pos, neg


def run(candidate_id: str, out_name: str, rows, sha: str, pos: int, neg: int, heads, congruent_constructions) -> None:
    import dod_battery
    out = ROOT / f"circuits/followups/{out_name}"
    SET = dod_battery.LineSpec(candidate_id, out_name, rows, pos, neg, tuple(heads)).set_components()
    batches = (len(rows) + v1.BATCH - 1) // v1.BATCH
    forwards_max = batches * (3 + len(NULL_SEEDS))
    plan = {"candidate_id": candidate_id, "rows": len(rows), "rows_sha256": sha, "set": list(heads), "congruent": list(congruent_constructions), "null_seeds": list(NULL_SEEDS),
            "forwards_max": forwards_max, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only", "bars": BARS}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, SET, pos, neg)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    arm, n = v1._run_arm(fw, rows, components=SET, mode="project"); forwards += n
    nulls = []
    for seed in NULL_SEEDS:
        na, n = v1._run_arm(fw, rows, components=SET, mode="project_random", seed=seed); forwards += n; nulls.append(na)
    congruent = [i for i, r in enumerate(rows) if r.construction in congruent_constructions]
    incongruent = [i for i in range(len(rows)) if i not in congruent]

    def score(idx):
        sub = [rows[i] for i in idx]; nat = [native[i] for i in idx]
        s = L.summarize(sub, nat, [arm[i] for i in idx])
        ns = [L.summarize(sub, nat, [na[i] for i in idx]) for na in nulls]
        null_max = max(x["target_damage_mean"] for x in ns)
        null_moves = {name: sum(x[f"{name}_abs_move_mean"] for x in ns) / len(ns) for name in L.UNRELATED}
        gates = {name: s[f"{name}_abs_move_mean"] <= null_moves[name] + BARS["gate_ratio"] * s["target_damage_mean"] for name in L.UNRELATED}
        return {"n": len(idx), "removal": s, "null_damage_max": null_max, "null_unrelated_abs_move_mean": null_moves, "gates": gates}

    constructions = sorted({r.construction for r in rows})
    capability = {c: (lambda cell: sum(cell) / len(cell))([1.0 if native[i]["answer"] > native[i]["foil"] else 0.0 for i, r in enumerate(rows) if r.construction == c]) for c in constructions}
    report = {"congruent": score(congruent), "incongruent": score(incongruent) if incongruent else None, "by_cell": {c: score([i for i, r in enumerate(rows) if r.construction == c]) for c in constructions}}
    C = report["congruent"]["removal"]; I = report["incongruent"]["removal"] if incongruent else None
    print("capability", capability); print("congruent", round(C["target_damage_mean"], 3), "positive", C["target_damage_positive_fraction"], "null max", round(report["congruent"]["null_damage_max"], 3), "gates", report["congruent"]["gates"])
    if I: print("incongruent", round(I["target_damage_mean"], 3), "positive", I["target_damage_positive_fraction"])
    predictions = {"pred_a_instrument_replays_native": instrument <= BARS["instrument_tol"],
                   "pred_b_congruent_native_capability": all(capability[c] >= BARS["capability_min"] for c in congruent_constructions),
                   "pred_c_congruent_removal_shifts_away_from_label": C["target_damage_mean"] >= BARS["shift_min"] and C["target_damage_positive_fraction"] >= BARS["direction_min"],
                   "pred_d_congruent_removal_beats_null": C["target_damage_mean"] > report["congruent"]["null_damage_max"],
                   "pred_e_congruent_removal_selective": all(report["congruent"]["gates"].values()),
                   "pred_f_incongruent_removal_shifts_toward_label": (I["target_damage_mean"] <= 0.0) if I else None}
    if forwards > forwards_max:
        raise SystemExit(f"price exceeded: {forwards} > {forwards_max}")
    out.write_text(json.dumps({"schema": "dod_natural_line_result_v1", "candidate_id": candidate_id, "plan": plan, "instrument_max_abs_error": instrument, "capability": capability,
                               "report": report, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))
