#!/usr/bin/env python3
"""Post-outcome exact repair of the fresh-writer block11 component cube."""

# BQGATE: EXPERIMENT pred_a_repaired_exactness pred_b_direct_and_entry_recurrence pred_c_full_cube_reconstructs_direct pred_d_quarantined_attention_shape_replays pred_e_exact_price
from __future__ import annotations

import json
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_auxiliary_will_had_fresh_writer_block11_crossing_cube_v1 as base


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_writer_block11_crossing_cube_v2.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_block11_crossing_cube_v2_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_writer_block11_crossing_cube_v2"
EXPECTED = dict(base.EXPECTED)
EXPECTED.update({
    "prior": "86068c7a8033c597231e20f002f20e9de4d1de16b7ea2addba7fe08b952f97da",
    "crossing": "b4ed1057e84e060fc2b01dfd7d056cfc4a8554f13d7a338b28048781b10e6f19",
})
ENTRY_TARGET = {"A1": 0.006125655754336495, "A2": 0.0024376408871950436}


def repair_result(result):
    entry = result["summaries"]["entry"]
    old = result["predictions"]
    predictions = {
        "pred_a_repaired_exactness": old["pred_a_authority_capability_exact_full_sequence_cube"],
        "pred_b_direct_and_entry_recurrence": bool(
            old["pred_b_boundary11_direct_ceiling_recurrence"]
            and all(abs(entry[family]["mean_recovery"] - ENTRY_TARGET[family]) <= 1e-6 for family in ("A1", "A2"))
        ),
        "pred_c_full_cube_reconstructs_direct": old["pred_c_full_sequence_cube_recovers_crossing"],
        "pred_d_quarantined_attention_shape_replays": old["pred_d_attention_is_dominant_transfer"],
        "pred_e_exact_price": old["pred_e_exact_zero_fit_price"],
    }
    terminal = "screen" if all(predictions.values()) else (
        "null" if predictions["pred_a_repaired_exactness"]
        and predictions["pred_b_direct_and_entry_recurrence"]
        and predictions["pred_c_full_cube_reconstructs_direct"]
        and predictions["pred_e_exact_price"] else "invalid"
    )
    result.update({
        "schema": "temporal_auxiliary_will_had_fresh_writer_block11_crossing_cube_result_v2",
        "candidate_id": CANDIDATE_ID,
        "evidence_class": "post_outcome_exact_instrument_repair_not_prospective_head_identification",
        "authority_sha256": EXPECTED,
        "predictions": predictions,
        "terminal": terminal,
        "reason": {
            "screen": "hybrid_consumed_entry_repair_validated_attention_shape_requires_disjoint_confirmation",
            "null": "exact_repair_valid_but_quarantined_attention_shape_did_not_replay",
            "invalid": "repaired_exactness_recurrence_coverage_or_price_invalid",
        }[terminal],
        "next_action": (
            "run disjoint prospective block11 head selection and confirmation"
            if terminal == "screen" else
            "retain the corrected accounting and do not promote the quarantined attention mechanism"
        ),
    })
    result["dryrun"]["candidate_id"] = CANDIDATE_ID
    return result


def main():
    base.PRIOR = PRIOR
    base.OUT = OUT
    base.CANDIDATE_ID = CANDIDATE_ID
    base.EXPECTED = EXPECTED
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        base.main()
        return
    captured = {}
    original_create = base.atomic_create_json

    def capture(_path, result):
        captured["result"] = result

    base.atomic_create_json = capture
    try:
        base.main()
    finally:
        base.atomic_create_json = original_create
    result = repair_result(captured["result"])
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "terminal": result["terminal"],
        "reason": result["reason"],
        "predictions": result["predictions"],
        "entry": result["summaries"]["entry"],
        "attention_retained": result["attention_singleton_retained_fraction"],
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
