#!/usr/bin/env python3
"""Fresh confirmation of the L13H8 OPEN-vs-POST construction interaction."""

from __future__ import annotations

import json

import circuit_fast_screen_candidate_bracket_l13h8_source_regions as shared


CANDIDATE_ID = "bracket.pending_opener.l13h8_open_post_family_confirm"
PATCH_LAYER, PATCH_HEAD = shared.PATCH_LAYER, shared.PATCH_HEAD
REGIONS = shared.REGIONS
CORNERS = (("OPEN",), ("POST",), ("OPEN", "POST"))
TARGET_FAMILIES = shared.TARGET_FAMILIES
CONTROL_FAMILIES = shared.CONTROL_FAMILIES
FAMILIES = shared.FAMILIES
PARENT_BASIC_SHA256 = "e5471038a18c2b7b285723e1bbe41d56c0236cab9d4e665edc2daa7595e6205d"

def build_rows() -> list[dict]:
    prefixes = ("The beekeeper", "A cartographer", "The drummer", "One forester", "Our jeweler", "The potter")
    words = (
        ("apron", "blossom", "cello", "dock", "evergreen"),
        ("fountain", "glacier", "heron", "inkwell", "junction"),
        ("kiwi", "loom", "mosaic", "napkin", "obelisk"),
        ("paddle", "quarry", "radish", "seashell", "tulip"),
        ("utensil", "vine", "windmill", "yacht", "zephyr"),
        ("basket", "crystal", "daisy", "falcon", "geyser"),
    )
    ordered = [(left, right) for left in shared.DELIMITERS for right in shared.DELIMITERS if left != right]
    rows = []
    for index, (left, right) in enumerate(ordered):
        prefix, (w0, w1, w2, w3, w4) = prefixes[index], words[index]
        lo, lc, lname = left
        ro, rc, rname = right
        group = shared.digest({"candidate": CANDIDATE_ID, "pair": (lname, rname), "index": index})
        tail = f"the {w0}, the {w1}, the {w2}, the {w3}, and the {w4} remained"
        rows.extend((
            shared._row(group, "direct_type", f"{prefix} recorded {lo} {tail}",
                        f"{prefix} recorded {ro} {tail}", lc, rc),
            shared._row(group, "completed_then_reopened",
                        f"{prefix} marked {lo} the {w0} {lc}, then recorded {ro} {tail}",
                        f"{prefix} marked {ro} the {w0} {rc}, then recorded {lo} {tail}", rc, lc),
            shared._row(group, "same_state_surface",
                        f"{prefix} recorded {lo} the {w0} near the {w1}, the {w2}, the {w3}, and the {w4}",
                        f"{prefix} recorded {lo} the {w4} near the {w3}, the {w2}, the {w1}, and the {w0}", lc, lc),
            shared._row(group, "same_state_punctuation", f"{prefix} paused, then recorded {lo} {tail}",
                        f"{prefix} paused: then recorded {lo} {tail}", lc, lc),
        ))
    for row in rows:
        row["split"] = "FRESH_CONFIRM"
    assert len(rows) == 24 and len({row["row_id"] for row in rows}) == 24
    return rows


ROWS = build_rows()
ROWS_SHA256 = shared.digest(ROWS)
EXPECTED_ROWS_SHA256 = "4f282eafbd809b0a83e7c273574353ae8dc8dbbc2f80d15c449bd781650aea5b"
assert ROWS_SHA256 == EXPECTED_ROWS_SHA256


def compile_plan(batch_size: int = 24) -> dict:
    conditions = ("native", "native_replay", "complete_head", "payload_OPEN", "payload_POST", "payload_OPEN+POST")
    forwards = 2 * len(conditions) * ((len(ROWS) + batch_size - 1) // batch_size)
    return {
        "schema": "bracket_l13h8_open_post_family_confirm_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "parent_basic_sha256": PARENT_BASIC_SHA256,
        "rows_sha256": ROWS_SHA256,
        "rows": len(ROWS), "groups": 6, "regions": list(REGIONS),
        "corners": [list(corner) for corner in CORNERS], "conditions": list(conditions),
        "batch_size": batch_size,
        "price": {"model_forwards": forwards, "example_evaluations": forwards * len(ROWS),
                  "backwards": 0, "parameter_updates": 0},
        "opened_splits": ["FRESH_CONFIRM"],
        "closed_splits": ["FINAL_TEST", "OOD"], "outcome_reads": [],
        "frozen_predictions": {
            "opener_payload": "OPEN approximates OPEN+POST in both families/directions; POST and exact OPENxPOST are negligible; completed/reopened OPEN exceeds direct OPEN",
            "post_or_synergy": "POST or OPENxPOST is material, or the completed-vs-direct OPEN magnitude gap disappears",
            "instrument_null": "native capability, replay, complete-head ceiling, or same-state controls fail"
        },
        "bars": {
            "native_positive_fraction_each_family_direction_min": 0.75,
            "native_replay_max_absolute_logit_error_max": 1e-5,
            "complete_head_target_positive_fraction_min": 0.75,
            "direct_open_median_recovery_min": 0.75,
            "completed_open_median_recovery_min": 1.25,
            "open_between_family_gap_min": 0.25,
            "post_mean_absolute_recovery_max": 0.10,
            "open_post_interaction_mean_absolute_max": 0.01,
            "open_vs_open_post_mean_absolute_difference_max": 0.10,
            "control_mean_absolute_closer_margin_change_max": 0.10,
            "control_mean_absolute_fraction_of_complete_head_max": 0.25
        }
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
