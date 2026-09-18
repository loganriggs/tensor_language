#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_congruent_native_capability pred_c_congruent_removal_shifts_away_from_label pred_d_congruent_removal_beats_null pred_e_congruent_removal_selective pred_f_incongruent_removal_shifts_toward_label
"""Pronoun gender he/she DoD (v75): the set {10.1, 9.6, 12.4, 15.1} on PILE rows (out-of-corpus; PREDICTS OOD, better_circuits §1).

Lane: Claude circuit lane. Parent: v73 (natural FineWeb rows: 77% of the congruent margin, 6/6). Same runner body (imported from
v73), same frozen bars, rows from `pronoun_gender_dod_natural_rows.py pile` (NeelNanda/pile-10k in stream order, receipt
`circuits/followups/pronoun_gender_dod_pile_rows_v75.json`, 64 rows, 16 per (noun gender x next token) cell; the any-sense
noun filter is stated: 'Cowbell', 'Manx' pass it). The Pile is not the training corpus, so this is the out-of-distribution panel.

PREDICTIONS (identical to v73, scored as written; failures preserved)
    pred_a_instrument_replays_native                 <= 1e-4
    pred_b_congruent_native_capability               >= 0.75 correct in each congruent cell
    pred_c_congruent_removal_shifts_away_from_label  congruent mean damage >= 0.15 logits and >= 60% rows positive
    pred_d_congruent_removal_beats_null              > max of the 16 null means
    pred_e_congruent_removal_selective               three reader gates
    pred_f_incongruent_removal_shifts_toward_label   incongruent mean damage <= 0

PRICE (registered maximum): 2 batches x (native + producer + set + 16 nulls) = 38 forwards; 0 backwards; 0 fits. Bar <= 40.
"""
from __future__ import annotations
import run_pronoun_gender_dod_natural_v73 as v73

PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_congruent_native_capability": ">= 0.75 per congruent cell",
               "pred_c_congruent_removal_shifts_away_from_label": "mean >= 0.15, >= 60% positive", "pred_d_congruent_removal_beats_null": "> null max",
               "pred_e_congruent_removal_selective": "three gates", "pred_f_incongruent_removal_shifts_toward_label": "mean <= 0"}
v73.ROWS = v73.ROOT / "circuits/followups/pronoun_gender_dod_pile_rows_v75.json"
v73.OUT = v73.ROOT / "circuits/followups/pronoun_gender_dod_pile_v75_result.json"
v73.CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_pile_v75"

if __name__ == "__main__":
    v73.main()
