#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_token_only_term_carries_the_three_heads pred_c_pattern_is_stable_within_cue pred_d_constant_pattern_transfers_pooled pred_e_constant_pattern_transfers_in_every_construction
"""Reflexive object control DoD (v117): token-only generator for {8.1, 13.1, 15.1} on the v105 fresh rows (v113's body with me / you as the cue
tokens). Parent: v115 (8.1 / 13.1 95% token-only readers of me / you; 15.1 64%). Same arms and bars as v113: zero the three slices; replace by
p x lamb x v1(pronoun) with the native pattern; replace with a constant pattern per (head, cue) from the other two constructions.
PREDICTIONS: as v113 (instrument; native token-only retention >= 0.80; pattern CV <= 0.35; constant-pattern retention >= 0.60 pooled and >= 0.50
per held-out construction). PRICE: 42 forwards; bar <= 48.
"""
from __future__ import annotations
import run_person_dod_token_only_generator_v113 as v113
import run_person_object_control_dod_battery_v105 as line

PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_token_only_term_carries_the_three_heads": ">= 0.80", "pred_c_pattern_is_stable_within_cue": "cv <= 0.35",
               "pred_d_constant_pattern_transfers_pooled": ">= 0.60", "pred_e_constant_pattern_transfers_in_every_construction": ">= 0.50 x 3"}
v113.OUT = v113.ROOT / "circuits/followups/person_object_control_dod_token_only_generator_v117_result.json"
v113.CANDIDATE_ID = "reflexive_object_control.me_vs_you.dod_token_only_generator_v117"
v113.PRONOUN_TOKENS = (" me", " you")
v113.line = line

if __name__ == "__main__":
    v113.main()
