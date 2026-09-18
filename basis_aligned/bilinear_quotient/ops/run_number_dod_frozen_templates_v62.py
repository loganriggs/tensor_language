#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_two_of_three_templates_capable pred_c_set_fraction_within_015_of_frozen pred_d_orthogonalized_set_selective_everywhere pred_e_pair_keep_retains_most_everywhere
"""Number family DoD, step 8 (v62): frozen prediction on three new templates (Predicts OOD).

Frozen before access: F* = 0.70 ± 0.15 for the full set {11.3, 5.7, 7.8, 9.7} along the raw were−was directions; selectivity
scored with the ORTHOGONALIZED directions (v57) on has−had / who−which / night−day; keep-only at the pair {11.3, 7.8}
(v61) >= 0.80. Templates (same 16 subjects with single-token plurals — declared reuse — and 16 places new to the line):
    pp_first    "Near the P the S(s)"              -> was/were
    relative    "The S(s) that lived near the P"   -> was/were
    yesterday   "Yesterday the S(s) by the P"      -> was/were

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   <= 1e-4
    pred_b_two_of_three_templates_capable   >= 2 constructions with both number cells >= 0.85
    pred_c_set_fraction_within_015_of_frozen   |fraction − 0.70| <= 0.15 in every capable construction
    pred_d_orthogonalized_set_selective_everywhere   three reader gates (raw 0.25 x damage; no null arms here) pass in every
                                       capable construction for the orthogonalized removal
    pred_e_pair_keep_retains_most_everywhere   keep-only at {11.3, 7.8} >= 0.80 in every capable construction

PRICE (registered maximum): 96 rows in 3 batches: native 3 + producer 3 + raw set 3 + orthogonalized set 3 + pair zero 3 +
pair keep 3 = 18 forwards; 0 backwards; 0 fits. Bar <= 24.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_number_dod_battery_v55 as v55

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/number_family_dod_frozen_templates_v62_result.json"
CANDIDATE_ID = "lexical_number.pp_intervener.dod_frozen_templates_v62"
FROZEN, BAND, CAPABILITY_MIN, LIVE_POSITIVE, GATE_RATIO, RETAIN_MIN, INSTRUMENT_TOL = 0.70, 0.15, 0.85, 0.75, 0.25, 0.80, 1e-4
FORWARDS_MAX = 24
PLACES = ('canal', 'lodge', 'inn', 'mine', 'farm', 'arena', 'bakery', 'brewery', 'cellar', 'clinic', 'depot', 'gallery', 'hangar', 'plaza', 'reef', 'shrine')
TEMPLATES = {"pp_first": lambda n, p: f"Near the {p} the {n}", "relative": lambda n, p: f"The {n} that lived near the {p}", "yesterday": lambda n, p: f"Yesterday the {n} by the {p}"}
PAIR = (v55.SINGLES[0], v55.SINGLES[2])


def build(subjects):
    for w in PLACES:
        L._single(" " + w)
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for name, make in TEMPLATES.items():
        for g in range(16):
            for plural in (True, False):
                noun = subjects[g] + ("s" if plural else ""); text = make(noun, PLACES[g])
                ids = L.ENCODING.encode(text); answer, foil = (" were", " was") if plural else (" was", " were")
                a_id, f_id = L._single(answer), L._single(foil)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise SystemExit(f"joint tokenization failed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["number_v62", name, g, plural, text]).encode()).hexdigest()[:24], name, g, plural, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows


def main() -> None:
    _, subjects = v55.build()
    rows = build(subjects)
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "frozen": {"fraction": FROZEN, "band": BAND}, "forwards_max": FORWARDS_MAX, "model_backwards": 0,
                          "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    raw = L.readout_directions(backend.model, v55.SINGLES, v55.WERE, v55.WAS)
    tense = L.readout_directions(backend.model, v55.SINGLES, L._single(" has"), L._single(" had"))
    orth = {}
    for k in raw:
        a, b = raw[k].float(), tense[k].float(); bh = b / b.norm(); orth[k] = a - (a @ bh) * bh
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    fw.directions = dict(raw); raw_arm, n = v1._run_arm(fw, rows, components=v55.SET, mode="project"); forwards += n
    fw.directions = dict(orth); orth_arm, n = v1._run_arm(fw, rows, components=v55.SET, mode="project"); forwards += n
    fw.directions = dict(raw)
    zero_pair, n = v1._run_arm(fw, rows, components=PAIR, mode="zero"); forwards += n
    keep_pair, n = v1._run_arm(fw, rows, components=PAIR, mode="keep_only"); forwards += n
    capability, report = {}, {}
    for c in TEMPLATES:
        idx = [i for i, r in enumerate(rows) if r.construction == c]; sub, subn = [rows[i] for i in idx], [native[i] for i in idx]
        for plural in (True, False):
            cell = [1.0 if native[i]["answer"] > native[i]["foil"] else 0.0 for i in idx if rows[i].present == plural]
            capability[f"{c}/{'plural' if plural else 'singular'}"] = sum(cell) / len(cell)
        rs = L.summarize(sub, subn, [raw_arm[i] for i in idx]); os_ = L.summarize(sub, subn, [orth_arm[i] for i in idx])
        gates = {name: os_[f"{name}_abs_move_mean"] <= GATE_RATIO * os_["target_damage_mean"] for name in L.UNRELATED}
        zd = L.summarize(sub, subn, [zero_pair[i] for i in idx])["target_damage_mean"]; kd = L.summarize(sub, subn, [keep_pair[i] for i in idx])["target_damage_mean"]
        capable = all(capability[f"{c}/{k}"] >= CAPABILITY_MIN for k in ("plural", "singular"))
        report[c] = {"capable": capable, "raw_fraction": rs["target_damage_fraction"], "raw_positive": rs["target_damage_positive_fraction"], "orth": os_, "gates": gates, "pair_keep_retention": 1.0 - kd / zd if zd else None,
                     "within_band": abs(rs["target_damage_fraction"] - FROZEN) <= BAND}
        print(c, "capable", capable, "raw fraction", round(rs["target_damage_fraction"], 3), "pos", round(rs["target_damage_positive_fraction"], 3), "orth gates", gates, "pair keep", round(report[c]["pair_keep_retention"], 3))
    capable = [c for c in TEMPLATES if report[c]["capable"]]
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_two_of_three_templates_capable": len(capable) >= 2,
                   "pred_c_set_fraction_within_015_of_frozen": bool(capable) and all(report[c]["within_band"] and report[c]["raw_positive"] >= LIVE_POSITIVE for c in capable),
                   "pred_d_orthogonalized_set_selective_everywhere": bool(capable) and all(all(report[c]["gates"].values()) for c in capable),
                   "pred_e_pair_keep_retains_most_everywhere": bool(capable) and all(report[c]["pair_keep_retention"] >= RETAIN_MIN for c in capable)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "number_family_dod_frozen_templates_result_v62", "candidate_id": CANDIDATE_ID, "rows_sha256": L.rows_sha256(rows), "instrument_max_abs_error": instrument, "capability": capability,
                               "templates": report, "capable": capable, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "capability": capability, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
