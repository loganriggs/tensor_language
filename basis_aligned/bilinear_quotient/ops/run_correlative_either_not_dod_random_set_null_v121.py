#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_set_beats_every_random_quadruple pred_c_random_quadruples_are_not_live pred_d_set_fraction_within_band_of_v71
"""correlative_either_not DoD (v121): matched-count random four-head-set null on the run_correlative_either_not_dod_battery_v119 fresh rows (SIMPLE). Body: v72's; 16 random quadruples from the 158
other heads, each removed along its own weight-only readout direction. Emitted by dod_line.py.
PREDICTIONS: pred_a instrument <= 1e-4; pred_b set damage > max random quadruple; pred_c no random quadruple live; pred_d pooled set fraction
within 0.44 +/- 0.05 (replay of the battery; the scored key keeps v72's literal name).
PRICE (registered maximum): batches x (native + producer + set + 16 random) forwards; 0 backwards; 0 fits. Bar <= 60.
"""
from __future__ import annotations
import run_pronoun_gender_dod_random_set_null_v72 as v72
import run_correlative_either_not_dod_battery_v119 as line

PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_set_beats_every_random_quadruple": "> random max",
               "pred_c_random_quadruples_are_not_live": "none live", "pred_d_set_fraction_within_band_of_v71": "0.44 +/- 0.05 (this line's band)"}
v72.OUT = v72.ROOT / "circuits/followups/correlative_either_not_dod_random_set_null_v121_result.json"
v72.CANDIDATE_ID = line.CANDIDATE_ID.rsplit(".", 1)[0] + ".dod_random_set_null_v121"
v72.EXCLUDED = set(line.HEADS)
v72.POOL = [(l, h) for l in range(18) for h in range(9) if (l, h) not in v72.EXCLUDED]
v72.SETS = [tuple(sorted(v72.random.Random(2026_09_18_121 + s).sample(v72.POOL, 4))) for s in range(16)]
v72.V71_FRACTION, v72.FORWARDS_MAX = 0.44, 60
v72.v71 = type("Line", (), {"build": staticmethod(lambda: line.build()[:3]), "HEADS": line.HEADS})

if __name__ == "__main__":
    v72.main()
