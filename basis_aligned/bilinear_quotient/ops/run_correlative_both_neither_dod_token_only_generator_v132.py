#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_token_only_term_carries_the_three_heads pred_c_pattern_is_stable_within_cue pred_d_constant_pattern_transfers_pooled pred_e_constant_pattern_transfers_in_every_construction
"""correlative_both_neither DoD (v132): token-only generator for ALL FOUR heads {8.1, 16.8, 14.8, 7.8} on the fresh rows (v113's body; cue tokens (" both", " neither")). Parents:
v129 / v130 (every head reads the cue token; token-only branch 56-99%). Arms: zero the four slices; replace by p x lamb x v1(cue) with the native
pattern; replace with a constant pattern per (head, cue) from the other two constructions (leave-one-construction-out; a median, not a fit).
PREDICTIONS: as v113 (instrument; native token-only retention >= 0.80 -- the pred name says "three heads", the set here is four; pattern CV <= 0.35;
constant-pattern retention >= 0.60 pooled and >= 0.50 per held-out construction). Prior for b: unsure -- 7.8's token-only share is only 0.56-0.59.
PRICE (registered maximum): 3 batches x (native + producer + zero + 4 pattern folds + [4 captures + 1 arm] x 2) = 51 forwards; bar <= 54.
"""
from __future__ import annotations
import run_person_dod_token_only_generator_v113 as v113
import run_correlative_both_neither_dod_battery_v120 as line

PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_token_only_term_carries_the_three_heads": ">= 0.80 (four heads here)", "pred_c_pattern_is_stable_within_cue": "cv <= 0.35",
               "pred_d_constant_pattern_transfers_pooled": ">= 0.60", "pred_e_constant_pattern_transfers_in_every_construction": ">= 0.50 x 3"}
v113.OUT = v113.ROOT / "circuits/followups/correlative_both_neither_dod_token_only_generator_v132_result.json"
v113.CANDIDATE_ID = line.CANDIDATE_ID.rsplit(".", 1)[0] + ".dod_token_only_generator_v132"
v113.PRONOUN_TOKENS = (" both", " neither")
v113.HEADS = line.HEADS
v113.FORWARDS_MAX = 54
v113.line = line

if __name__ == "__main__":
    v113.main()
