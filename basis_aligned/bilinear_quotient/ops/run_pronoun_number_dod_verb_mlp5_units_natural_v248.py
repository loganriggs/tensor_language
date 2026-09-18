#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_congruent_damage_live pred_c_units_beat_random_unit_triples pred_d_units_selective pred_e_incongruent_shifts_toward_label
"""Pronoun number they/he DoD (v248): the VERB-SITE MLP-5 units {1036, 715} OUT OF THE PANEL. v245 / v246: at the verb these two units carry 82% of MLP 5's part of
MLP-6 unit 69 and, zeroed at the verb, remove 29% of the plural detector's verb contrast and 1.05% of the margin. The natural rows annotate the noun (cue) but not the
verb; the token after the cue is the verb in many but not all rows, so the edit is applied at cue + 1 and the registered bar is set low (LIVE 0.003). v197 / v199 / v200's
design otherwise (16 random 2-unit sets of MLP 5 at cue + 1, number-free readers, incongruent counter-case).
PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native        hooked forward with no edit = producer native <= 1e-4
    pred_b_congruent_damage_live            congruent rows: damage fraction >= 0.003 and positive >= 0.55. Prior: unsure (cue + 1 is not always the verb).
    pred_c_units_beat_random_unit_triples   congruent damage > max of the 16 random 2-unit sets
    pred_d_units_selective                  each reader |move| <= null mean |move| + 0.25 x damage
    pred_e_incongruent_shifts_toward_label  incongruent rows: mean oriented damage < 0
PRICE (registered maximum): 4 batches x (native + cue+1 + all-positions + 16 nulls) + producer replay 4 = 80 forwards; 0 backwards; 0 fits. Bar <= 84.
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
OUT = ROOT / "circuits/followups/pronoun_number_dod_verb_mlp5_units_natural_v248_result.json"
SOURCES = {"fineweb": ROOT / "circuits/followups/pronoun_number_dod_natural_rows_v77.json", "pile": ROOT / "circuits/followups/pronoun_number_dod_pile_rows_v78.json"}
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_verb_mlp5_units_natural_v248"
LAYER, UNITS, CONGRUENT = 5, (1036, 715), ("natural_plural_they", "natural_singular_he")
NULL_SEEDS = tuple(range(8101, 8117))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, INSTRUMENT_TOL = 0.003, 0.55, 0.25, 1e-4
FORWARDS_MAX = 84
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_congruent_damage_live": ">= 0.003, positive >= 0.55", "pred_c_units_beat_random_unit_triples": "> random max", "pred_d_units_selective": "three gates", "pred_e_incongruent_shifts_toward_label": "< 0"}


def main() -> None:
    rows, cue, shas = [], {}, {}
    for name, path in SOURCES.items():
        rs, sha, they, he = N.rows_from_receipt(path, "they", " they", " he", lambda r: f"natural_{r['number']}_{r['label']}")
        recs = json.loads(path.read_text())["rows"]
        for row, rec in zip(rs, recs):
            if tuple(rec["ids"]) != row.ids: raise SystemExit("row order mismatch")
            cue[row.row_id] = rec["cue_offset"]
        rows.extend(rs); shas[name] = sha
    cue_of = lambda row: min(cue[row.row_id] + 1, len(row.ids) - 2)     # the token AFTER the cue (the verb in most rows; never the final token)
    readers = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": shas, "units": list(UNITS), "congruent": list(CONGRUENT), "null_seeds": list(NULL_SEEDS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "gate_ratio": GATE_RATIO, "instrument_tol": INSTRUMENT_TOL}}
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
    cue_arm, all_arm = run((UNITS, cue_of)), run((UNITS, lambda r: None))
    arms = {"cue_congruent": summarize(cue_arm, cong), "cue_incongruent": summarize(cue_arm, incong), "all_positions_congruent": summarize(all_arm, cong), "all_positions_incongruent": summarize(all_arm, incong),
            "per_cell_cue": {c: summarize(cue_arm, [i for i, r in enumerate(rows) if r.construction == c]) for c in sorted({r.construction for r in rows})}}
    pool = [j for j in range(backend.model.transformer.h[LAYER].mlp.Down.weight.shape[1]) if j not in UNITS]; nulls = []
    for seed in NULL_SEEDS:
        g = torch.Generator().manual_seed(seed); units = tuple(pool[k] for k in torch.randperm(len(pool), generator=g)[:len(UNITS)].tolist())
        nulls.append({"units": units, **summarize(run((units, cue_of)), cong)})
    null_max = max(s["damage"] for s in nulls); null_moves = {k: sum(s[f"{k}_abs_move"] for s in nulls) / len(nulls) for k in readers}
    two = arms["cue_congruent"]; gates = {k: two[f"{k}_abs_move"] <= null_moves[k] + GATE_RATIO * two["damage"] for k in readers}
    print("instrument", instrument, "congruent", {k: round(v, 4) for k, v in two.items()}, "incongruent", round(arms["cue_incongruent"]["damage"], 4), round(arms["cue_incongruent"]["fraction"], 4), "all-pos congruent", round(arms["all_positions_congruent"]["fraction"], 4), "null max", round(null_max, 4), "gates", gates)
    print("per cell", {c: (round(s["damage"], 3), round(s["fraction"], 3), s["positive"]) for c, s in arms["per_cell_cue"].items()})
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_congruent_damage_live": two["fraction"] >= LIVE_FRACTION and two["positive"] >= LIVE_POSITIVE, "pred_c_units_beat_random_unit_triples": two["damage"] > null_max,
                   "pred_d_units_selective": all(gates.values()), "pred_e_incongruent_shifts_toward_label": arms["cue_incongruent"]["damage"] < 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp8_units_natural_result_v197", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "arms": arms, "null_damage_max": null_max, "null_moves": null_moves, "nulls": nulls, "gates": gates,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
