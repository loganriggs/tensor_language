#!/usr/bin/env python3
"""Fresh held-out authority for exact semantic-OPEN contribution zero-removal."""

from __future__ import annotations

import json

import circuit_fast_screen_candidate_bracket_l13h8_source_regions as shared


CANDIDATE_ID = "bracket.pending_opener.l13h8_semantic_open_zero_removal"
PATCH_LAYER, PATCH_HEAD = shared.PATCH_LAYER, shared.PATCH_HEAD
REGIONS = shared.REGIONS
CORNERS = (("OPEN",),)
TARGET_FAMILIES = shared.TARGET_FAMILIES
CONTROL_FAMILIES = shared.CONTROL_FAMILIES
FAMILIES = shared.FAMILIES
PARENT_CONFIRM_SHA256 = "39d75e204bdc428f9f2c574eaff22483ede1a95c976c78ab409d58f242d66124"


def build_rows() -> list[dict]:
    prefixes = ("The clockmaker", "A diver", "The herbalist", "One illustrator", "Our mason", "The ranger")
    words = (
        ("amber", "bridge", "cactus", "dagger", "echo"),
        ("fable", "garnet", "hammock", "igloo", "jigsaw"),
        ("kayak", "lichen", "magnet", "noodle", "opal"),
        ("parchment", "quiver", "raspberry", "silo", "tapestry"),
        ("uniform", "valley", "wheel", "yew", "zucchini"),
        ("beacon", "canyon", "domino", "fiddle", "gazebo"),
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
        row["split"] = "FRESH_HELDOUT_REMOVAL"
    assert len(rows) == 24 and len({row["row_id"] for row in rows}) == 24
    return rows


ROWS = build_rows()
ROWS_SHA256 = shared.digest(ROWS)
EXPECTED_ROWS_SHA256 = "79b16e1b431716a16e8f919b70d54ea132d03f05d5808c2641e993c51b97aa6e"
assert ROWS_SHA256 == EXPECTED_ROWS_SHA256


def compile_plan(batch_size: int = 24) -> dict:
    conditions = ("native", "native_replay", "zero_complete_head", "zero_semantic_open")
    forwards = 2 * len(conditions) * ((len(ROWS) + batch_size - 1) // batch_size)
    return {
        "schema": "bracket_l13h8_semantic_open_zero_removal_plan_v1",
        "candidate_id": CANDIDATE_ID, "parent_confirm_sha256": PARENT_CONFIRM_SHA256,
        "rows_sha256": ROWS_SHA256, "rows": len(ROWS), "groups": 6,
        "conditions": list(conditions), "batch_size": batch_size,
        "price": {"model_forwards": forwards, "example_evaluations": forwards * len(ROWS),
                  "backwards": 0, "parameter_updates": 0},
        "opened_splits": ["FRESH_HELDOUT_REMOVAL"], "closed_splits": ["FINAL_TEST", "OOD"],
        "outcome_reads": [],
        "frozen_predictions": {
            "selective_necessity": "zeroing the exact semantic-OPEN term damages both target constructions/endpoints relative to full-head zero, while same-state controls retain their answers with low collateral",
            "not_necessary_or_not_selective": "target damage is weak in any family/endpoint, or same-state collateral/answer loss is material",
            "instrument_null": "native capability, exact replay, live OPEN term, or full-head removal ceiling fails"
        },
        "bars": {
            "native_positive_fraction_each_family_direction_min": 0.75,
            "native_replay_max_absolute_logit_error_max": 1e-5,
            "semantic_open_term_norm_min": 1e-4,
            "full_head_target_damage_positive_fraction_min": 0.75,
            "target_median_normalized_damage_min_each_family_direction": 0.50,
            "target_positive_damage_fraction_min_each_family_direction": 0.75,
            "control_mean_absolute_margin_damage_max": 0.10,
            "control_mean_absolute_normalized_damage_max": 0.25,
            "control_answer_preservation_fraction_min_each_family_direction": 0.75
        }
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
