#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_congruent_native_capability pred_c_congruent_removal_shifts_away_from_label pred_d_congruent_removal_beats_null pred_e_congruent_removal_selective pred_f_incongruent_removal_shifts_toward_label
"""Pronoun number they/he DoD (v77): the set {9.6, 12.4, 15.1, 10.5} on natural FineWeb rows (training corpus, out-of-panel) (PREDICTS OOD).

Lane: Claude circuit lane. Parent: v76 (fresh rows 7/7, 76%). Body: `dod_natural_line.run` (the v73/v75 body lifted into a module).
Rows: `pronoun_number_dod_natural_rows.py` (outcome-blind miner, receipt `circuits/followups/pronoun_number_dod_natural_rows_v77.json`, 64 rows, 16 per
(noun number x next token) cell; any-sense noun filter stated: 'Visitors Bureau', 'head coach' pass it). Congruent cells
(plural/they, singular/he) are the readout test; incongruent cells are natural counter-cases (a singular noun followed by
'they' is often generic or a distant plural). Damage = oriented drop of (actual next token - other pronoun).

PREDICTIONS (scored as written by dod_natural_line; bars frozen from the aspectual natural line v20; failures preserved)
    pred_a_instrument_replays_native                 <= 1e-4
    pred_b_congruent_native_capability               >= 0.75 in each congruent cell
    pred_c_congruent_removal_shifts_away_from_label  congruent mean damage >= 0.15 and >= 60% rows positive
    pred_d_congruent_removal_beats_null              > max of 16 null means
    pred_e_congruent_removal_selective               three reader gates on congruent rows
    pred_f_incongruent_removal_shifts_toward_label   incongruent mean damage <= 0. Prior: unsure.

PRICE (registered maximum): 2 batches x (native + producer + set + 16 nulls) = 38 forwards; 0 backwards; 0 fits.
"""
from __future__ import annotations
import dod_natural_line as N
import run_pronoun_number_dod_battery_v76 as v76

CANDIDATE_ID = "pronoun_number.they_vs_he.dod_natural_v77"
ROWS = N.ROOT / "circuits/followups/pronoun_number_dod_natural_rows_v77.json"
OUT = "pronoun_number_dod_natural_v77_result.json"
CONGRUENT = ("natural_plural_they", "natural_singular_he")
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_congruent_native_capability": ">= 0.75 per congruent cell",
               "pred_c_congruent_removal_shifts_away_from_label": "mean >= 0.15, >= 60% positive", "pred_d_congruent_removal_beats_null": "> null max",
               "pred_e_congruent_removal_selective": "three gates", "pred_f_incongruent_removal_shifts_toward_label": "mean <= 0"}


def main() -> None:
    rows, sha, they, he = N.rows_from_receipt(ROWS, "they", " they", " he", lambda r: f"natural_{r['number']}_{r['label']}")
    N.run(CANDIDATE_ID, OUT, rows, sha, they, he, v76.HEADS, CONGRUENT)


if __name__ == "__main__":
    main()
