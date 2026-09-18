#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_congruent_native_capability pred_c_congruent_removal_shifts_away_from_label pred_d_congruent_removal_beats_null pred_e_congruent_removal_selective pred_f_incongruent_removal_shifts_toward_label
"""Lexical number were/was DoD (v139): the v55 set {11.3, 5.7, 7.8, 9.7} on Pile rows (out-of-corpus) (PREDICTS OOD; the number family's first line had no natural
rows). Body: `dod_natural_line.run`; rows `lexical_number_dod_natural_rows.py` (generic miner; receipt `circuits/followups/lexical_number_dod_pile_rows_v139.json`; a list noun in
singular / plural form within 12 tokens, next token were / was; pronouns and be-forms excluded from the context; any-sense filter stated). Congruent
cells plural/were and singular/was are the readout test; incongruent cells are corpus counter-cases (v77/v78/v100/v101 falsified the "toward the text"
reading on number lines each time; registered again as the standing bar). Readers: has-had (tense), who-which, night-day, set by importing v55.
PREDICTIONS (bars frozen from v20): pred_a instrument <= 1e-4; pred_b congruent capability >= 0.75 per cell; pred_c congruent mean damage >= 0.15 and
>= 60% positive; pred_d > null max; pred_e three gates; pred_f incongruent mean damage <= 0 (prior: expected FALSE, as on every number line so far).
PRICE (registered maximum): 2 batches x (native + producer + set + 16 nulls) = 38 forwards; 0 backwards; 0 fits.
"""
from __future__ import annotations
import dod_natural_line as N
import run_number_dod_battery_v55 as v55

CANDIDATE_ID = "lexical_number.pp_intervener.dod_pile_v139"
ROWS = N.ROOT / "circuits/followups/lexical_number_dod_pile_rows_v139.json"
OUT = "lexical_number_dod_pile_v139_result.json"
HEADS = ((11, 3), (5, 7), (7, 8), (9, 7))
CONGRUENT = ("natural_plural_were", "natural_singular_was")
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_congruent_native_capability": ">= 0.75 per congruent cell",
               "pred_c_congruent_removal_shifts_away_from_label": "mean >= 0.15, >= 60% positive", "pred_d_congruent_removal_beats_null": "> null max",
               "pred_e_congruent_removal_selective": "three gates", "pred_f_incongruent_removal_shifts_toward_label": "mean <= 0"}


def main() -> None:
    rows, sha, were, was = N.rows_from_receipt(ROWS, "were", " were", " was", lambda r: f"natural_{r['cue']}_{r['label']}")
    N.run(CANDIDATE_ID, OUT, rows, sha, were, was, HEADS, CONGRUENT)


if __name__ == "__main__":
    main()
