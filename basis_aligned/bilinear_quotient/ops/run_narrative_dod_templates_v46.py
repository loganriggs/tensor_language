#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_two_of_three_templates_capable pred_c_set_removal_transfers_to_every_capable_template pred_d_set_removal_selective_on_every_capable_template
"""Narrative tense DoD battery, step 5 (v46): template-varying transfer of the set removal (as v5/v29).

Three new constructions on the v43 lexicon, varying the tense carrier and the frame:
    years_ago      "Years ago the S stood nearby. The main reason for the F P"       / "These days the S stands nearby. …"
    once_still     "Once the S stood nearby and drew crowds. The main reason …"       / "Still the S stands nearby and draws crowds. …"
    back_then_now  "Back then the S stood nearby. The main reason …"                   / "Right now the S stands nearby. …"
Set {15.5, 11.3, 9.4, 9.1} removed along `O_h^T(u_was - u_is)` with 16 norm-matched nulls and three readers per
construction; frozen band [0.5, 1.5] x 0.74.

PREDICTIONS: pred_a instrument <= 1e-4; pred_b >= 2 of 3 constructions capable (both cells >= 0.85); pred_c in every capable
construction fraction >= 0.10, positive >= 0.75, > max null, fraction within [0.37, 1.11]; pred_d three reader gates over the null.
PRICE (registered maximum): 96 rows in 3 batches x (native + producer + set + 16 nulls) = 57 forwards; bar <= 64.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_narrative_dod_sweep_and_set_v42 as v42
import run_narrative_dod_confirm_v43 as v43

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/narrative_tense_dod_templates_v46_result.json"
CANDIDATE_ID = "narrative_tense.past_vs_present.dod_templates_v46"
NULL_SEEDS = tuple(range(1301, 1317))
CAPABILITY_MIN, LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, BAND, REF, INSTRUMENT_TOL = 0.85, 0.10, 0.75, 0.25, (0.5, 1.5), 0.74, 1e-4
FORWARDS_MAX = 64
TEMPLATES = {
    "years_ago": (lambda s, f, p: f"Years ago the {s} stood nearby. The main reason for the {f} {p}", lambda s, f, p: f"These days the {s} stands nearby. The main reason for the {f} {p}"),
    "once_still": (lambda s, f, p: f"Once the {s} stood nearby and drew crowds. The main reason for the {f} {p}", lambda s, f, p: f"Still the {s} stands nearby and draws crowds. The main reason for the {f} {p}"),
    "back_then_now": (lambda s, f, p: f"Back then the {s} stood nearby. The main reason for the {f} {p}", lambda s, f, p: f"Right now the {s} stands nearby. The main reason for the {f} {p}"),
}


def build():
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for name, (past_make, present_make) in TEMPLATES.items():
        for g in range(16):
            s_, p_, f_ = v43.SUBJECTS[g], v43.PLACES[g], v43.FOCUS[g]
            for past in (True, False):
                text = (past_make if past else present_make)(s_, f_, p_)
                ids = L.ENCODING.encode(text); answer, foil = (" was", " is") if past else (" is", " was")
                a_id, f_id = L._single(answer), L._single(foil)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise SystemExit(f"joint tokenization failed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["narrative_templates", name, g, past, text]).encode()).hexdigest()[:24], name, g, past, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows


def main() -> None:
    rows = build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
                          "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, v43.SET, v42.WAS, v42.IS)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    set_arm, n = v1._run_arm(fw, rows, components=v43.SET, mode="project"); forwards += n
    nulls = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=v43.SET, mode="project_random", seed=seed); forwards += n
        nulls.append(arm)
    report, capability = {}, {}
    for c in TEMPLATES:
        idx = [i for i, r in enumerate(rows) if r.construction == c]
        sub, subn = [rows[i] for i in idx], [native[i] for i in idx]
        for past in (True, False):
            cell = [1.0 if native[i]["answer"] > native[i]["foil"] else 0.0 for i in idx if rows[i].present == past]
            capability[f"{c}/{'past' if past else 'present'}"] = sum(cell) / len(cell)
        s = L.summarize(sub, subn, [set_arm[i] for i in idx])
        ns = [L.summarize(sub, subn, [nl[i] for i in idx]) for nl in nulls]
        null_max = max(x["target_damage_mean"] for x in ns); null_moves = {name: sum(x[f"{name}_abs_move_mean"] for x in ns) / len(ns) for name in L.UNRELATED}
        gates = {name: s[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * s["target_damage_mean"] for name in L.UNRELATED}
        capable = all(capability[f"{c}/{k}"] >= CAPABILITY_MIN for k in ("past", "present"))
        transfers = s["target_damage_fraction"] >= LIVE_FRACTION and s["target_damage_positive_fraction"] >= LIVE_POSITIVE and s["target_damage_mean"] > null_max and BAND[0] * REF <= s["target_damage_fraction"] <= BAND[1] * REF
        report[c] = {"capable": capable, "set": s, "null_damage_max": null_max, "gates": gates, "selective": all(gates.values()), "transfers": transfers}
        print(c, "capable", capable, "fraction", round(s["target_damage_fraction"], 3), "pos", round(s["target_damage_positive_fraction"], 3), "null_max", round(null_max, 4), "gates", gates)
    capable = [c for c in TEMPLATES if report[c]["capable"]]
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_two_of_three_templates_capable": len(capable) >= 2,
                   "pred_c_set_removal_transfers_to_every_capable_template": bool(capable) and all(report[c]["transfers"] for c in capable),
                   "pred_d_set_removal_selective_on_every_capable_template": bool(capable) and all(report[c]["selective"] for c in capable)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "narrative_tense_dod_templates_result_v46", "candidate_id": CANDIDATE_ID, "rows_sha256": L.rows_sha256(rows), "instrument_max_abs_error": instrument, "capability": capability,
              "templates": report, "capable": capable, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "capability": capability, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
