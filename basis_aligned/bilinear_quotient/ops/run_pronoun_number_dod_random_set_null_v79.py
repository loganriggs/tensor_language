#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_set_beats_every_random_quadruple pred_c_random_quadruples_are_not_live pred_d_set_fraction_within_band_of_v76
"""Pronoun number they/he DoD (v79): matched-count random four-head-set null on the v76 fresh rows (SIMPLE). Body: v72's, with
the number set and rows; 16 random quadruples from the 158 other heads, each along its own weight-only they-he direction.
PREDICTIONS: pred_a instrument <= 1e-4; pred_b set damage > max random quadruple; pred_c no random quadruple live; pred_d pooled
set fraction within 0.76 +/- 0.05 (replay on identical rows).
PRICE (registered maximum): 3 batches x (native + producer + set + 16 random) = 57 forwards; 0 backwards; 0 fits. Bar <= 60.
"""
from __future__ import annotations
import run_pronoun_gender_dod_random_set_null_v72 as v72
import run_pronoun_number_dod_battery_v76 as v76

PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_set_beats_every_random_quadruple": "> random max",
               "pred_c_random_quadruples_are_not_live": "none live", "pred_d_set_fraction_within_band_of_v76": "0.76 +/- 0.05"}
v72.OUT = v72.ROOT / "circuits/followups/pronoun_number_dod_random_set_null_v79_result.json"
v72.CANDIDATE_ID = "pronoun_number.they_vs_he.dod_random_set_null_v79"
v72.EXCLUDED = set(v76.HEADS)
v72.POOL = [(l, h) for l in range(18) for h in range(9) if (l, h) not in v72.EXCLUDED]
v72.SETS = [tuple(sorted(v72.random.Random(2026_09_18_79 + s).sample(v72.POOL, 4))) for s in range(16)]
v72.V71_FRACTION, v72.FORWARDS_MAX = 0.76, 60
v72.v71 = type("Line", (), {"build": staticmethod(lambda: v76.build()[:3]), "HEADS": v76.HEADS})

if __name__ == "__main__":
    v72.main()
