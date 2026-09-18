#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_congruent_native_capability pred_c_congruent_removal_shifts_away_from_label pred_d_congruent_removal_beats_null pred_e_congruent_removal_selective pred_f_incongruent_removal_shifts_toward_label
"""Reflexive person DoD (v111): the set {8.1, 13.1, 10.5, 15.1} on Pile rows (out-of-corpus) (PREDICTS OOD). Body: `dod_natural_line.run`; rows
`person_dod_natural_rows.py` (generic miner; receipt `circuits/followups/person_reflexive_dod_pile_rows_v111.json`; an I / you token within 12 tokens, next token myself /
yourself; other person pronouns excluded from the context). Congruent cells I/myself and you/yourself are the readout test; the incongruent
cells are the few corpus counter-cases the miner found (a quoted speaker: 'ask yourself' after 'I'), recorded at whatever size they are.
Readers: will−would, who−which, night−day (set by importing v104).
PREDICTIONS (bars frozen from v20; failures preserved): pred_a instrument <= 1e-4; pred_b congruent capability >= 0.75 per cell; pred_c congruent
mean damage >= 0.15 and >= 60% positive; pred_d > null max; pred_e three gates; pred_f incongruent mean damage <= 0 (small n; prior: unsure).
PRICE (registered maximum): 2 batches x (native + producer + set + 16 nulls) = 38 forwards; 0 backwards; 0 fits.
"""
from __future__ import annotations
import dod_natural_line as N
import run_person_reflexive_dod_battery_v104 as line

CANDIDATE_ID = "reflexive_person.i_vs_you.dod_pile_v111"
ROWS = N.ROOT / "circuits/followups/person_reflexive_dod_pile_rows_v111.json"
OUT = "person_reflexive_dod_pile_v111_result.json"
CONGRUENT = ("natural_I_myself", "natural_you_yourself")
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_congruent_native_capability": ">= 0.75 per congruent cell",
               "pred_c_congruent_removal_shifts_away_from_label": "mean >= 0.15, >= 60% positive", "pred_d_congruent_removal_beats_null": "> null max",
               "pred_e_congruent_removal_selective": "three gates", "pred_f_incongruent_removal_shifts_toward_label": "mean <= 0"}


def main() -> None:
    rows, sha, my, your = N.rows_from_receipt(ROWS, "myself", " myself", " yourself", lambda r: f"natural_{r['cue']}_{r['label']}")
    N.run(CANDIDATE_ID, OUT, rows, sha, my, your, line.HEADS, CONGRUENT)


if __name__ == "__main__":
    main()
