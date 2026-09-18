#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_top4_set_live_and_beats_null pred_d_top4_set_selective pred_e_top4_overlaps_shared_family
"""Modal would/will reuse census (v49b): blind readout sweep + top-4 set on the line's authored A1/A2 contexts (opened rows).
Contrast `O_h^T(u_would - u_will)`. Predictions as in `dod_reuse_census.run`; prior for pred_e: yes (the 2026-09-06
greedy set for this line was {9.4, 11.3}). PRICE: 181 forwards per 32-row batch (128 rows -> 724); registered in dod_reuse_census."""
import circuit_fast_screen_candidate_modal_remoteness as m
import dod_reuse_census as R

REGISTERED_PREDICTIONS = (
    ("pred_a_instrument_replays_native", "no-edit forward matches producer.native <= 1e-4"),
    ("pred_b_native_capability", "all construction x side cells >= 0.85"),
    ("pred_c_top4_set_live_and_beats_null", "fraction >= 0.10, positive >= 0.75, > max of 16 norm-matched nulls"),
    ("pred_d_top4_set_selective", "three unrelated readers within null + 0.25 x damage"),
    ("pred_e_top4_overlaps_shared_family", "at least two of the top four heads are in {8.1, 9.1, 9.4, 11.3, 15.5}"),
)
rows, pos, neg = R.rows_from_candidate(m, " would", " will", "modal")
if __name__ == "__main__":
    R.run("modal_remoteness.would_vs_will.dod_reuse_census_v49b", "modal_remoteness_dod_reuse_census_v49b_result.json", rows, pos, neg)
