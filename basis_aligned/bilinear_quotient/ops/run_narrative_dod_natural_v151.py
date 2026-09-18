#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_congruent_native_capability pred_c_congruent_removal_shifts_away_from_label pred_d_congruent_removal_beats_null pred_e_congruent_removal_selective pred_f_incongruent_removal_shifts_toward_label
"""Narrative was/is DoD (v151): the v42 / v43 set {15.5, 11.3, 9.1, 9.4} on natural FineWeb rows (training corpus, out-of-panel) (PREDICTS OOD). Body: `dod_natural_line.run`; rows
`narrative_dod_natural_rows.py` (receipt `circuits/followups/narrative_dod_natural_rows_v151.json`): a single temporal adverb (yesterday / ago / formerly / previously /
earlier vs today / now / currently / nowadays / presently) within 12 tokens of a next token was / is; be-forms excluded from the context. This is
a natural ANALOGUE of the panel's cue (a tensed clause + 'last winter' / 'every winter'), stated as such. Readers set explicitly: who-which,
night-day, pronoun-number they-he (was-were and has-had are tense- or target-adjacent).
PREDICTIONS (bars frozen from v20; failures preserved): pred_a instrument <= 1e-4; pred_b congruent capability >= 0.75 per cell; pred_c congruent
mean damage >= 0.15 and >= 60% positive; pred_d > null max; pred_e three gates; pred_f incongruent mean damage <= 0 (prior: unsure -- the temporal
family's 8.1 is a token reader, but this set reads a contextual tense state).
PRICE (registered maximum): 2 batches x (native + producer + set + 16 nulls) = 38 forwards; 0 backwards; 0 fits.
"""
from __future__ import annotations
import aspectual_dod_lib as L
import dod_natural_line as N

L.READERS = {"animacy_who_which": (" who", " which"), "canonical_night_day": (" night", " day"), "pronoun_number_they_he": (" they", " he")}
L.UNRELATED = ("animacy_who_which", "canonical_night_day", "pronoun_number_they_he")
CANDIDATE_ID = "narrative_tense.past_vs_present.dod_natural_v151"
ROWS = N.ROOT / "circuits/followups/narrative_dod_natural_rows_v151.json"
OUT = "narrative_dod_natural_v151_result.json"
HEADS = ((15, 5), (11, 3), (9, 1), (9, 4))
CONGRUENT = ("natural_past_was", "natural_present_is")
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_congruent_native_capability": ">= 0.75 per congruent cell",
               "pred_c_congruent_removal_shifts_away_from_label": "mean >= 0.15, >= 60% positive", "pred_d_congruent_removal_beats_null": "> null max",
               "pred_e_congruent_removal_selective": "three gates", "pred_f_incongruent_removal_shifts_toward_label": "mean <= 0"}


def main() -> None:
    rows, sha, was, is_ = N.rows_from_receipt(ROWS, "was", " was", " is", lambda r: f"natural_{r['cue']}_{r['label']}")
    N.run(CANDIDATE_ID, OUT, rows, sha, was, is_, HEADS, CONGRUENT)


if __name__ == "__main__":
    main()
