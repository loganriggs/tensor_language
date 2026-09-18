#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays_v206 pred_b_verb_only_removes_004 pred_c_noun_plus_verb_matches_all_positions pred_d_verb_edit_beats_random_sets pred_e_verb_edit_selective
"""Pronoun number they/he DoD (v215): the MLP-8 number units {829, 953, 1030} zeroed at the VERB only, and at NOUN + VERB. v169b: at the noun 5.3% of the margin,
at all positions 11.8%; v203 / v214: the same detectors re-fire at the verb and the readers take 33% of their coefficients from the verb. If the noun and
the verb are the two sites, noun + verb should reproduce the all-positions number. Null: 16 seeded random 3-unit sets of MLP 8 zeroed at the verb.
Readers will-would / who-which / night-day (number-free), gate move <= null mean + 0.25 x damage.
PREDICTIONS (scored as written; failures preserved)
    pred_a_baseline_replays_v206             native pooled they - he margin = 196.62 within relative 1e-3
    pred_b_verb_only_removes_004             the verb-only edit removes >= 0.04 of the margin (noun-only was 0.053)
    pred_c_noun_plus_verb_matches_all_positions  noun + verb removes >= 0.80 x the all-positions edit measured in the same run
    pred_d_verb_edit_beats_random_sets       verb-only damage > max of the 16 random 3-unit sets at the verb
    pred_e_verb_edit_selective               each unrelated reader |move| <= null mean |move| + 0.25 x damage
PRICE (registered maximum): 3 batches x (native + verb + noun + verb + all + 16 nulls) = 60 forwards; 0 backwards; 0 fits. Bar <= 63.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp8_units_verb_edit_v215_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp8_units_verb_edit_v215"
UNITS, LAYER, V206_MARGIN, VERB_MIN, MATCH, GATE_RATIO, N_NULL, SEED, BATCH = (829, 953, 1030), 8, 196.62, 0.04, 0.80, 0.25, 16, 215, 32
FORWARDS_MAX = 63
L.READERS = {"tense_will_would": (" will", " would"), "animacy_who_which": (" who", " which"), "canonical_night_day": (" night", " day")}
PREDICTIONS = {"pred_a_baseline_replays_v206": "<= 1e-3", "pred_b_verb_only_removes_004": ">= 0.04", "pred_c_noun_plus_verb_matches_all_positions": ">= 0.80 x all", "pred_d_verb_edit_beats_random_sets": "> null max", "pred_e_verb_edit_selective": "three gates"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": UNITS, "layer": LAYER, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"verb_min": VERB_MIN, "match": MATCH, "gate_ratio": GATE_RATIO}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); blocks = model.transformer.h
    readers = {name: (L._single(a_), L._single(b_)) for name, (a_, b_) in L.READERS.items()}; readers["target"] = (he, she)
    import random
    rng = random.Random(SEED); pool = [j for j in range(blocks[LAYER].mlp.Down.weight.shape[1]) if j not in UNITS]
    null_sets = [tuple(sorted(rng.sample(pool, len(UNITS)))) for _ in range(N_NULL)]
    verb_of = lambda row: noun_of(row) + 1
    conditions = [("native", None), ("verb", (UNITS, verb_of)), ("noun_verb", (UNITS, lambda r: [noun_of(r), verb_of(r)])), ("all", (UNITS, lambda r: None))] + [(f"null{k}", (s_, verb_of)) for k, s_ in enumerate(null_sets)]
    forwards, per_row = 0, {name: [] for name, _ in conditions}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        for name, edits in conditions:
            if edits is not None and name == "noun_verb":
                units, pf = edits; edits = (units, lambda r, pf=pf: pf(r))
            per_row[name].extend(dod_units.forward_margins(backend, fw, chunk, LAYER, edits, readers)); forwards += 1
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; pairs = [(i, partner[(row.construction, row.group, False)]) for i, row in enumerate(rows) if row.present]
    pooled = lambda name: sum(per_row[name][i]["target"] - per_row[name][j]["target"] for i, j in pairs)
    native = pooled("native"); damage = {name: (native - pooled(name)) / native for name in per_row if name != "native"}
    unrel = [r_ for r_ in readers if r_ != "target"]
    moves = {name: {r_: sum(abs(a[r_] - n[r_]) for n, a in zip(per_row["native"], per_row[name])) / len(rows) for r_ in unrel} for name in per_row if name != "native"}
    null_moves = {r_: sum(moves[f"null{k}"][r_] for k in range(N_NULL)) / N_NULL for r_ in unrel}; null_max = max(damage[f"null{k}"] for k in range(N_NULL))
    dmg_row = damage["verb"] * native / len(pairs); gates = {r_: moves["verb"][r_] <= null_moves[r_] + GATE_RATIO * dmg_row for r_ in unrel}
    report = {"native_pooled": native, "damage": {k: damage[k] for k in ("verb", "noun_verb", "all")}, "null_damage_max": null_max, "moves_verb": moves["verb"], "null_moves": null_moves, "gates": gates, "null_sets": null_sets}
    print("native", round(native, 2), "damage verb", round(damage["verb"], 4), "noun+verb", round(damage["noun_verb"], 4), "all", round(damage["all"], 4), "null max", round(null_max, 4)); print("moves", {r_: round(v, 4) for r_, v in moves["verb"].items()}, "null", {r_: round(v, 4) for r_, v in null_moves.items()}, gates)
    predictions = {"pred_a_baseline_replays_v206": abs(native - V206_MARGIN) / V206_MARGIN <= 1e-3, "pred_b_verb_only_removes_004": damage["verb"] >= VERB_MIN, "pred_c_noun_plus_verb_matches_all_positions": damage["noun_verb"] >= MATCH * damage["all"],
                   "pred_d_verb_edit_beats_random_sets": damage["verb"] > null_max, "pred_e_verb_edit_selective": all(gates.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp8_units_verb_edit_result_v215", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
