#!/usr/bin/env python3
"""Fresh held-out SELECT confirmation of the BASIC_SCREEN family interaction."""

from __future__ import annotations

import json

import circuit_fast_screen_candidate_bracket_l13h8_source_regions as shared


CANDIDATE_ID = "bracket.pending_opener.l13h8_source_region_family_interaction_select"
PATCH_LAYER, PATCH_HEAD = shared.PATCH_LAYER, shared.PATCH_HEAD
REGIONS = shared.REGIONS
# Confirmation only: no repeat of the eight-corner discovery sweep.
CORNERS = (("PREFIX",), ("OPEN", "POST"))
TARGET_FAMILIES = shared.TARGET_FAMILIES
CONTROL_FAMILIES = shared.CONTROL_FAMILIES
FAMILIES = shared.FAMILIES
PARENT_RESULT_SHA256 = "e5471038a18c2b7b285723e1bbe41d56c0236cab9d4e665edc2daa7595e6205d"

def build_rows() -> list[dict]:
    prefixes = ("The archivist", "A carpenter", "The geologist", "One violinist", "Our astronomer", "The pharmacist")
    words = (
        ("anchor", "breeze", "cabin", "depot", "ember"),
        ("fossil", "garden", "hinge", "ivory", "kernel"),
        ("lilac", "marble", "nectar", "oar", "plume"),
        ("quill", "river", "stencil", "timber", "vase"),
        ("willow", "xylophone", "yeast", "zipper", "basin"),
        ("copper", "delta", "easel", "fjord", "goblet"),
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
            shared._row(group, "same_state_punctuation",
                        f"{prefix} paused, then recorded {lo} {tail}",
                        f"{prefix} paused: then recorded {lo} {tail}", lc, lc),
        ))
    for row in rows:
        row["split"] = "SELECT"
    assert len(rows) == 24 and len({row["row_id"] for row in rows}) == 24
    return rows


ROWS = build_rows()
ROWS_SHA256 = shared.digest(ROWS)
EXPECTED_ROWS_SHA256 = "c2d9e82977c98bff95125ac329024d053a4ccbb3174abc58ed33c345899a270d"
assert ROWS_SHA256 == EXPECTED_ROWS_SHA256


def compile_plan(batch_size: int = 24) -> dict:
    chunks = (len(ROWS) + batch_size - 1) // batch_size
    conditions = ("native", "native_replay", "complete_head", "payload_PREFIX", "payload_OPEN+POST")
    forwards = chunks * 2 * len(conditions)
    return {
        "schema": "bracket_source_region_family_interaction_select_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "parent_result_sha256": PARENT_RESULT_SHA256,
        "rows_sha256": ROWS_SHA256,
        "rows": len(ROWS),
        "groups": 6,
        "regions": list(REGIONS),
        "corners": [list(corner) for corner in CORNERS],
        "conditions": list(conditions),
        "batch_size": batch_size,
        "price": {"model_forwards": forwards, "example_evaluations": forwards * len(ROWS),
                  "backwards": 0, "parameter_updates": 0},
        "opened_splits": ["SELECT"],
        "closed_splits": ["FINAL_TEST", "OOD"],
        "outcome_reads": [],
        "frozen_predictions": {
            "family_interaction_holds": "direct PREFIX remains null while completed/reopened PREFIX is negative; OPEN+POST remains positive and larger for completed/reopened",
            "family_interaction_fails": "either PREFIX effects converge across constructions or the OPEN+POST construction gap disappears",
            "instrument_null": "native capability, replay, complete-head ceiling, or same-state controls fail",
        },
        "bars": {
            "native_positive_fraction_each_family_direction_min": 0.75,
            "native_replay_max_absolute_logit_error_max": 1e-5,
            "complete_head_target_positive_fraction_min": 0.75,
            "direct_prefix_mean_absolute_recovery_max": 0.10,
            "completed_prefix_mean_recovery_max": -0.50,
            "prefix_between_family_gap_min": 0.50,
            "direct_open_post_median_recovery_min": 0.75,
            "completed_open_post_median_recovery_min": 1.25,
            "open_post_between_family_gap_min": 0.25,
            "open_post_positive_fraction_each_family_direction_min": 0.75,
            "control_mean_absolute_closer_margin_change_max": 0.10,
            "control_mean_absolute_fraction_of_complete_head_max": 0.25
        }
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
