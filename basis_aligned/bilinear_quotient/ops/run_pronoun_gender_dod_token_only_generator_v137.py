#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_token_only_term_carries_the_three_heads pred_c_pattern_is_stable_within_cue pred_d_constant_pattern_transfers_pooled pred_e_constant_pattern_transfers_in_every_construction
"""Pronoun gender he/she DoD (v137): token-only generator for the gender set's two token readers {10.1, 15.1} on the v71 fresh rows (v113's body).
Parent: v80 (10.1 98% from the gendered noun, 72% token-only; 15.1 58% / 50%; 9.6 and 12.4 contextual and left native here). Cue tokens: the
20 fresh gender nouns of v71. Arms: zero the two slices; replace by p x lamb x v1(noun) with the native pattern; replace with a constant pattern per
(head, cue side) from the other two constructions. NOTE on the constants: v113 keys the constant by (head, construction, present), i.e. by the
male / female SIDE, not by the individual noun -- a two-entry table per head over the side, a stricter test than per-token.
PREDICTIONS: as v113 (instrument; native token-only retention >= 0.80 -- prior: unsure, 15.1 is only half token-only; pattern CV <= 0.35;
constant-pattern retention >= 0.60 pooled and >= 0.50 per held-out construction). PRICE: 2 batches x (3 + 2 folds + [2 captures + 1 arm] x 2) = 22 forwards; bar <= 48 (v113's).
"""
from __future__ import annotations
import run_person_dod_token_only_generator_v113 as v113
import run_pronoun_gender_dod_battery_v71 as g

PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_token_only_term_carries_the_three_heads": ">= 0.80 (two heads here)", "pred_c_pattern_is_stable_within_cue": "cv <= 0.35",
               "pred_d_constant_pattern_transfers_pooled": ">= 0.60", "pred_e_constant_pattern_transfers_in_every_construction": ">= 0.50 x 3"}
v113.OUT = v113.ROOT / "circuits/followups/pronoun_gender_dod_token_only_generator_v137_result.json"
v113.CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_token_only_generator_v137"
v113.PRONOUN_TOKENS = tuple(" " + w for p in g.PAIRS for w in p)
v113.HEADS = ((10, 1), (15, 1))
v113.line = type("Line", (), {"build": staticmethod(lambda: (*g.build(), None, None)), "HEADS": g.HEADS, "CANDIDATE_ID": g.CANDIDATE_ID})

if __name__ == "__main__":
    v113.main()
