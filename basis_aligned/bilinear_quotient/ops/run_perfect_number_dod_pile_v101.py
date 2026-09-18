#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_congruent_native_capability pred_c_congruent_removal_shifts_away_from_label pred_d_congruent_removal_beats_null pred_e_congruent_removal_selective pred_f_incongruent_removal_shifts_toward_label
"""Perfect have/has DoD (v101): the set {11.3, 7.8, 5.3, 9.7} on Pile rows (out-of-corpus) (PREDICTS OOD). Body: `dod_natural_line.run`; rows
`perfect_number_dod_natural_rows.py` (generic miner; receipt `circuits/followups/perfect_number_dod_pile_rows_v101.json`; a list noun in singular / plural form within 12
tokens, next token have / has; pronouns and have/has/had excluded from the context; any-sense filter stated). Congruent cells plural/have
and singular/has are the readout test; the incongruent cells are corpus counter-cases (the cue noun is not the subject, e.g. a possessive
or a relative clause). Readers: will−would, who−which, night−day (set by importing v97).
PREDICTIONS (bars frozen from v20; failures preserved): pred_a instrument <= 1e-4; pred_b congruent capability >= 0.75 per cell; pred_c
congruent mean damage >= 0.15 and >= 60% positive; pred_d > null max; pred_e three gates; pred_f incongruent mean damage <= 0. Prior for f:
unsure -- v77/v78 (pronoun number) falsified the same reading twice; registered again because the two lines read number differently.
PRICE (registered maximum): 2 batches x (native + producer + set + 16 nulls) = 38 forwards; 0 backwards; 0 fits.
"""
from __future__ import annotations
import dod_natural_line as N
import run_perfect_number_dod_battery_v97 as line

CANDIDATE_ID = "perfect_number.have_vs_has.dod_pile_v101"
ROWS = N.ROOT / "circuits/followups/perfect_number_dod_pile_rows_v101.json"
OUT = "perfect_number_dod_pile_v101_result.json"
CONGRUENT = ("natural_plural_have", "natural_singular_has")
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_congruent_native_capability": ">= 0.75 per congruent cell",
               "pred_c_congruent_removal_shifts_away_from_label": "mean >= 0.15, >= 60% positive", "pred_d_congruent_removal_beats_null": "> null max",
               "pred_e_congruent_removal_selective": "three gates", "pred_f_incongruent_removal_shifts_toward_label": "mean <= 0"}


def main() -> None:
    rows, sha, have, has = N.rows_from_receipt(ROWS, "have", " have", " has", lambda r: f"natural_{r['cue']}_{r['label']}")
    N.run(CANDIDATE_ID, OUT, rows, sha, have, has, line.HEADS, CONGRUENT)


if __name__ == "__main__":
    main()
