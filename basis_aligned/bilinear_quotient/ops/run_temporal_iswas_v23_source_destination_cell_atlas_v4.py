#!/usr/bin/env python3
"""Padding-domain repair of the frozen v23 source/destination cell atlas."""

# BQGATE: EXPERIMENT pred_a_authority_dependency_partitions_finiteness_enumeration_exact_price_and_no_v24 pred_b_exact_cell_closure_and_every_full_head_replay pred_c_changed_source_to_matched_suffix_destination_cell_is_reciprocal pred_d_at_least_three_heads_have_a_reciprocal_cell pred_e_a_reciprocal_cell_label_is_shared_by_at_least_two_heads
from pathlib import Path

import run_temporal_iswas_v23_source_destination_cell_atlas_v3 as atlas


PREDICTION_KEYS = (
    "pred_a_authority_dependency_partitions_finiteness_enumeration_exact_price_and_no_v24",
    "pred_b_exact_cell_closure_and_every_full_head_replay",
    "pred_c_changed_source_to_matched_suffix_destination_cell_is_reciprocal",
    "pred_d_at_least_three_heads_have_a_reciprocal_cell",
    "pred_e_a_reciprocal_cell_label_is_shared_by_at_least_two_heads",
)


atlas.AUTHORITY = atlas.ROOT.parent / "polynomial_causal/TEMPORAL_ISWAS_V23_SOURCE_DESTINATION_CELL_ATLAS_V4_PREREGISTRATION.md"
atlas.PRIOR = atlas.ROOT / "circuits/prior_art/temporal_iswas_v23_source_destination_cell_atlas_v4.json"
atlas.OUT = atlas.ROOT / "circuits/followups/temporal_iswas_v23_source_destination_cell_atlas_v4_result.json"
atlas.CANDIDATE_ID = "cross_task.temporal_iswas.v23_source_destination_cell_atlas_v4"
atlas.SCHEMA = "temporal_iswas_v23_source_destination_cell_atlas_result_v4"
atlas.EXPECTED["authority"] = "9d7292d61931776724be53e76c0c0988947484ead0cfd60420806a200f979cd1"
atlas.EXPECTED["prior"] = "34453c79e59705dff3bb0208aac5edd82121361d48250e0aa5475c87921b7c44"


if __name__ == "__main__":
    atlas.main()
