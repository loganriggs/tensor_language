#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_keep_only_readout_retains_most pred_c_random_keep_retains_little pred_d_two_of_three_templates_capable pred_e_set_removal_transfers_to_every_capable_template pred_f_set_removal_selective_on_every_capable_template
"""Temporal will/had DoD battery, step 2 (v29): keep-only sufficiency and template-varying transfer.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v28 (S = {11.3, 9.1, 15.5, 9.4} along
`O_h^T(u_will - u_had)`: 81% of the fresh-row margin, selective, 70x null).

WHY. Two rungs of the battery at once (they share nothing but the model load): EXTRACTED at the head
boundary -- keep only the readout projection at the four heads (discard everything else they write at the
final query) vs keep a random unit direction (16 seeds) vs zero the four slices; and PREDICTS OOD on
templates -- the S removal with its norm-matched null and three readers on three new constructions that
move the cue, add a comma, or add an adverbial phrase.

ROWS. Keep-only: the 64 v28 fresh rows (opened by v28 for removal only). Templates (96 rows, same lexicon):
    adverb_phrase   "Tomorrow morning the A near the P"          / "Earlier today the A near the P"
    expected_that   "It is expected that tomorrow the A near the P" / "It was reported that earlier the A near the P"
    agent_first_comma "The A near the P, tomorrow,"               / "The A near the P, earlier,"   (comma-final)

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native      <= 1e-4 on both panels
    pred_b_keep_only_readout_retains_most keep-only retention >= 0.70 of the four-slice zero damage
    pred_c_random_keep_retains_little     every random keep <= 0.30
    pred_d_two_of_three_templates_capable >= 2 constructions with both cue cells >= 0.85
    pred_e_set_removal_transfers_to_every_capable_template   in each capable construction: fraction >= 0.10,
                                          positive >= 0.75, > max null, and fraction within [0.5, 1.5] x 0.807
    pred_f_set_removal_selective_on_every_capable_template   three reader gates over the null in each

PRICE (registered maximum): keep panel 2 batches x (native 1 + producer 1 + zero 1 + keep 1 + 16 random) = 40;
template panel 3 batches x (native 1 + producer 1 + set 1 + 16 nulls) = 57 -> 97 forwards; 0 backwards; 0 fits.
Bar <= 110.
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
import run_temporal_dod_removal_v28 as v28

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_keep_and_templates_v29_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_keep_and_templates_v29"
WILL, HAD = v28.WILL, v28.HAD
KEEP_SEEDS, NULL_SEEDS = tuple(range(901, 917)), tuple(range(951, 967))
RETAIN_MIN, RANDOM_MAX, CAPABILITY_MIN, LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, BAND, V28_FRACTION, INSTRUMENT_TOL = 0.70, 0.30, 0.85, 0.10, 0.75, 0.25, (0.5, 1.5), 0.807, 1e-4
FORWARDS_MAX = 110
SET = v28.SET
TEMPLATES = {
    "adverb_phrase": (lambda p, a: f"Tomorrow morning the {a} near the {p}", lambda p, a: f"Earlier today the {a} near the {p}"),
    "expected_that": (lambda p, a: f"It is expected that tomorrow the {a} near the {p}", lambda p, a: f"It was reported that earlier the {a} near the {p}"),
    "agent_first_comma": (lambda p, a: f"The {a} near the {p}, tomorrow,", lambda p, a: f"The {a} near the {p}, earlier,"),
}


def build_templates():
    rows = L.build_rows_lexicon(v28.AGENTS, v28.PLACES, TEMPLATES, "temporal_templates")
    return [L.Row(r.row_id, r.construction, r.group, r.present, r.text, r.ids, " will" if r.present else " had", " had" if r.present else " will",
                  WILL if r.present else HAD, HAD if r.present else WILL, r.final, (), r.reader_ids) for r in rows]


def main() -> None:
    rows = v28.build(); templates = build_templates()
    for r in rows + templates:
        if L.ENCODING.encode(r.text + r.answer) != list(r.ids) + [r.answer_id] or L.ENCODING.encode(r.text + r.foil) != list(r.ids) + [r.foil_id]:
            raise SystemExit(f"joint tokenization failed for {r.text!r}")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "templates": len(templates), "rows_sha256": L.rows_sha256(rows), "templates_sha256": L.rows_sha256(templates),
                          "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
                          "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, SET, WILL, HAD)
    forwards = 0
    # keep-only panel
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    zero, n = v1._run_arm(fw, rows, components=SET, mode="zero"); forwards += n
    keep, n = v1._run_arm(fw, rows, components=SET, mode="keep_only"); forwards += n
    zd = L.summarize(rows, native, zero)["target_damage_mean"]
    retention = 1.0 - L.summarize(rows, native, keep)["target_damage_mean"] / zd
    random_ret = []
    for seed in KEEP_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=SET, mode="keep_only_random", seed=seed); forwards += n
        random_ret.append(1.0 - L.summarize(rows, native, arm)["target_damage_mean"] / zd)
    print("zero", round(zd, 4), "keep retention", round(retention, 4), "random max", round(max(random_ret), 4))
    # template panel
    nat_t, n = v1._run_arm(fw, templates); forwards += n
    ref_t, n = v1._producer_native(backend, templates); forwards += n
    instrument = max(instrument, max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(nat_t, ref_t)))
    set_t, n = v1._run_arm(fw, templates, components=SET, mode="project"); forwards += n
    nulls_t = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, templates, components=SET, mode="project_random", seed=seed); forwards += n
        nulls_t.append(arm)
    report, capability = {}, {}
    for c in TEMPLATES:
        idx = [i for i, r in enumerate(templates) if r.construction == c]
        sub, subn = [templates[i] for i in idx], [nat_t[i] for i in idx]
        for present in (True, False):
            cell = [1.0 if nat_t[i]["answer"] > nat_t[i]["foil"] else 0.0 for i in idx if templates[i].present == present]
            capability[f"{c}/{'tomorrow' if present else 'earlier'}"] = sum(cell) / len(cell)
        s = L.summarize(sub, subn, [set_t[i] for i in idx])
        ns = [L.summarize(sub, subn, [nl[i] for i in idx]) for nl in nulls_t]
        null_max = max(x["target_damage_mean"] for x in ns)
        null_moves = {name: sum(x[f"{name}_abs_move_mean"] for x in ns) / len(ns) for name in L.UNRELATED}
        gates = {name: s[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * s["target_damage_mean"] for name in L.UNRELATED}
        capable = all(capability[f"{c}/{k}"] >= CAPABILITY_MIN for k in ("tomorrow", "earlier"))
        transfers = (s["target_damage_fraction"] >= LIVE_FRACTION and s["target_damage_positive_fraction"] >= LIVE_POSITIVE and s["target_damage_mean"] > null_max
                     and BAND[0] * V28_FRACTION <= s["target_damage_fraction"] <= BAND[1] * V28_FRACTION)
        report[c] = {"capable": capable, "set": s, "null_damage_max": null_max, "gates": gates, "selective": all(gates.values()), "transfers": transfers}
        print(c, "capable", capable, "fraction", round(s["target_damage_fraction"], 3), "pos", round(s["target_damage_positive_fraction"], 3), "null_max", round(null_max, 4), "gates", gates)
    capable = [c for c in TEMPLATES if report[c]["capable"]]
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_keep_only_readout_retains_most": retention >= RETAIN_MIN,
        "pred_c_random_keep_retains_little": all(r <= RANDOM_MAX for r in random_ret),
        "pred_d_two_of_three_templates_capable": len(capable) >= 2,
        "pred_e_set_removal_transfers_to_every_capable_template": bool(capable) and all(report[c]["transfers"] for c in capable),
        "pred_f_set_removal_selective_on_every_capable_template": bool(capable) and all(report[c]["selective"] for c in capable),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_keep_and_templates_result_v29", "candidate_id": CANDIDATE_ID, "rows_sha256": L.rows_sha256(rows), "templates_sha256": L.rows_sha256(templates),
              "instrument_max_abs_error": instrument, "keep_only": {"zero_damage": zd, "retention": retention, "random_retention": random_ret},
              "capability": capability, "templates": report, "capable": capable, "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "capability": capability, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
