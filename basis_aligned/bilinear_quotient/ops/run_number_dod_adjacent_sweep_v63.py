#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_top4_set_live_and_beats_null pred_d_top4_set_selective pred_e_top4_overlaps_shared_family
"""Number family, step 9 (v63): blind readout sweep on the subject-ADJACENT frame ("Near the P the S(s)" -> was/were),
where the v62 set removed only 14%. Uses `dod_reuse_census.run` on 32 rows built here (v62 lexicon); family overlap is
scored against the census family (8.1/9.1/9.4/11.3/15.5) as a fixed reference; the real question is which heads lead.
PRICE: 181 forwards per 32-row batch (1 batch) registered in dod_reuse_census."""
import json, hashlib
import aspectual_dod_lib as L
import dod_reuse_census as R
import run_number_dod_battery_v55 as v55
import run_number_dod_frozen_templates_v62 as v62

REGISTERED_PREDICTIONS = (
    ("pred_a_instrument_replays_native", "no-edit forward matches producer.native <= 1e-4"),
    ("pred_b_native_capability", "both number cells >= 0.85"),
    ("pred_c_top4_set_live_and_beats_null", "fraction >= 0.10, positive >= 0.75, > max of 16 norm-matched nulls"),
    ("pred_d_top4_set_selective", "three readers within null + 0.25 x damage (readers: has-had, who-which, night-day)"),
    ("pred_e_top4_overlaps_shared_family", "at least two of the top four heads are in the census family (prior: no)"),
)
_, subjects = v55.build()
rows = [r for r in v62.build(subjects) if r.construction == "pp_first"]
if __name__ == "__main__":
    R.run("lexical_number.pp_intervener.dod_adjacent_sweep_v63", "number_family_dod_adjacent_sweep_v63_result.json", rows, v55.WERE, v55.WAS)
