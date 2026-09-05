#!/usr/bin/env python3
"""Fresh same-template delimiter triplets for exact opener effect coding."""

from __future__ import annotations

import json

import circuit_fast_screen_candidate_bracket_l13h8_source_regions as shared


CANDIDATE_ID = "bracket.pending_opener.l13h8_semantic_open_shared_contrast"
PATCH_LAYER, PATCH_HEAD = shared.PATCH_LAYER, shared.PATCH_HEAD
DELIMITERS = shared.DELIMITERS
FAMILIES = ("direct_type", "completed_then_reopened", "same_state_surface", "same_state_punctuation")
TARGET_FAMILIES = FAMILIES[:2]
CONTROL_FAMILIES = FAMILIES[2:]


def build_rows() -> list[dict]:
    bundles = (
        ("The glassblower", ("anvil", "cedar", "flute", "harbor", "iris")),
        ("A milliner", ("juniper", "ladle", "meteor", "nest", "piano")),
    )
    rows = []
    for bundle, (prefix, words) in enumerate(bundles):
        w0, w1, w2, w3, w4 = words
        tail = f"the {w0}, the {w1}, the {w2}, the {w3}, and the {w4} remained"
        direct_group = shared.digest({"candidate": CANDIDATE_ID, "bundle": bundle, "family": "direct_type"})
        for family in FAMILIES:
            group = shared.digest({"candidate": CANDIDATE_ID, "bundle": bundle, "family": family})
            for delimiter_index, (opener, closer, delimiter_name) in enumerate(DELIMITERS):
                if family == "direct_type":
                    text = f"{prefix} recorded {opener} {tail}"
                elif family == "completed_then_reopened":
                    text = f"{prefix} marked ( the {w0} ), then recorded {opener} {tail}"
                elif family == "same_state_surface":
                    text = f"{prefix} carefully noted {opener} the {w4}, the {w3}, the {w2}, the {w1}, and the {w0} remained"
                else:
                    text = f"{prefix} paused: then recorded {opener} {tail}"
                ids, answer = shared.encode(text), shared.answer_id(closer)
                position = shared.semantic_open_position(ids, answer)
                rows.append({
                    "row_id": shared.digest({"group": group, "delimiter": delimiter_name}),
                    "group_id": group, "bundle_id": bundle, "matched_direct_group_id": direct_group,
                    "split": "FRESH_BASIC", "family_id": family,
                    "role": "target" if family in TARGET_FAMILIES else "invariance_control",
                    "delimiter_index": delimiter_index, "delimiter_name": delimiter_name,
                    "text": text, "ids": ids, "answer_id": answer,
                    "open_position": position, "final_position": len(ids) - 1,
                })
    assert len(rows) == 24 and len({row["row_id"] for row in rows}) == 24
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group_id"], []).append(row)
    assert len(by_group) == 8
    for triplet in by_group.values():
        assert {row["delimiter_index"] for row in triplet} == {0, 1, 2}
        assert len({len(row["ids"]) for row in triplet}) == 1
        assert len({row["open_position"] for row in triplet}) == 1
        assert len({row["final_position"] for row in triplet}) == 1
    return rows


ROWS = build_rows()
ROWS_SHA256 = shared.digest(ROWS)
EXPECTED_ROWS_SHA256 = "5c44fa3519e8c2675fab1772b7a1c9e41cd333336c003ff3d58c97b1e198858d"
assert ROWS_SHA256 == EXPECTED_ROWS_SHA256


def compile_plan() -> dict:
    conditions = ("native", "native_replay", "natural_contrast_swap", "contrast_removed", "shared_removed")
    return {
        "schema": "bracket_l13h8_shared_contrast_plan_v1", "candidate_id": CANDIDATE_ID,
        "rows_sha256": ROWS_SHA256, "rows": 24, "triplet_groups": 8,
        "conditions": list(conditions),
        "price": {"model_forwards": 5, "example_evaluations": 120,
                  "backwards": 0, "parameter_updates": 0},
        "opened_splits": ["FRESH_BASIC"], "closed_splits": ["TEST", "OOD"], "outcome_reads": [],
        "bars": {
            "native_positive_fraction_each_family_min": 0.75,
            "native_replay_max_absolute_logit_error_max": 1e-5,
            "semantic_open_term_norm_min": 1e-4,
            "natural_swap_positive_type_transfer_fraction_each_target_family_min": 0.75,
            "contrast_removal_positive_type_damage_fraction_each_target_family_min": 0.75,
            "contrast_removal_type_to_common_median_ratio_each_target_family_min": 2.0,
            "shared_removal_common_to_type_median_ratio_min": 2.0,
            "matched_invariance_median_absolute_ratio_difference_max": 0.25
        },
        "frozen_predictions": {
            "pred_a": "shared plus contrast: natural contrast swap transfers type; contrast removal moves centered type more than common support; shared removal moves common support more than centered type; matched rewrites preserve ratios",
            "pred_b": "pure shared support: natural swap or contrast removal has weak type effect while shared removal dominates",
            "pred_c": "entangled context: natural swap is live but type/common separation or matched-context invariance fails"
        }
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
