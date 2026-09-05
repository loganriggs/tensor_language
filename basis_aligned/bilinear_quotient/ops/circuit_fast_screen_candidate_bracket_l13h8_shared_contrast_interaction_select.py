#!/usr/bin/env python3
"""Fresh SELECT triplets for the exact shared/contrast interaction factorial."""

from __future__ import annotations

import json

import circuit_fast_screen_candidate_bracket_l13h8_shared_contrast as parent
import circuit_fast_screen_candidate_bracket_l13h8_source_regions as shared


CANDIDATE_ID = "bracket.pending_opener.l13h8_shared_contrast_interaction_factorial"
PATCH_LAYER, PATCH_HEAD = parent.PATCH_LAYER, parent.PATCH_HEAD
DELIMITERS, FAMILIES = parent.DELIMITERS, parent.FAMILIES
TARGET_FAMILIES, CONTROL_FAMILIES = parent.TARGET_FAMILIES, parent.CONTROL_FAMILIES
PARENT_RESULT_SHA256 = "37014b88e9341d63626d5152fca87e2df7de90c7dc76e2b309935bc330806c3a"


def build_rows() -> list[dict]:
    bundles = (
        ("The perfumer", ("acacia", "bell", "compass", "dahlia", "eclipse")),
        ("A roofer", ("fresco", "ginger", "helmet", "inlet", "jasmine")),
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
                rows.append({"row_id": shared.digest({"group": group, "delimiter": delimiter_name}),
                             "group_id": group, "bundle_id": bundle, "matched_direct_group_id": direct_group,
                             "split": "SELECT", "family_id": family,
                             "role": "target" if family in TARGET_FAMILIES else "invariance_control",
                             "delimiter_index": delimiter_index, "delimiter_name": delimiter_name,
                             "text": text, "ids": ids, "answer_id": answer,
                             "open_position": shared.semantic_open_position(ids, answer),
                             "final_position": len(ids) - 1})
    groups = {}
    for row in rows: groups.setdefault(row["group_id"], []).append(row)
    assert len(rows) == 24 and len(groups) == 8
    for triplet in groups.values():
        assert {row["delimiter_index"] for row in triplet} == {0, 1, 2}
        assert len({len(row["ids"]) for row in triplet}) == 1
        assert len({row["open_position"] for row in triplet}) == 1
        assert len({row["final_position"] for row in triplet}) == 1
    return rows


ROWS = build_rows()
ROWS_SHA256 = shared.digest(ROWS)
EXPECTED_ROWS_SHA256 = "2ce11bfc0f236da3bcf115726d8280a259ed8c9f5ff7e84a7bdb72806e5ee2cb"
assert ROWS_SHA256 == EXPECTED_ROWS_SHA256


def compile_plan() -> dict:
    conditions = ("native", "native_replay", "natural_contrast_swap", "contrast_removed",
                  "shared_removed", "opener_zero_both_removed")
    return {"schema": "bracket_l13h8_shared_contrast_interaction_select_plan_v1",
            "candidate_id": CANDIDATE_ID, "parent_result_sha256": PARENT_RESULT_SHA256,
            "rows_sha256": ROWS_SHA256, "rows": 24, "triplet_groups": 8,
            "conditions": list(conditions),
            "price": {"model_forwards": 6, "example_evaluations": 144,
                      "backwards": 0, "parameter_updates": 0},
            "opened_splits": ["SELECT"], "closed_splits": ["TEST", "OOD"], "outcome_reads": [],
            "interaction": "I_A=A_remove_both-A_remove_mu-A_remove_delta+A_native for A in {centered_type, common_support}",
            "bars": {"native_positive_fraction_each_family_min": 0.75,
                     "native_replay_max_absolute_logit_error_max": 1e-5,
                     "semantic_open_term_norm_min": 1e-4,
                     "natural_swap_positive_type_transfer_fraction_each_target_family_min": 0.75,
                     "median_normalized_interaction_each_family_max": 0.25},
            "frozen_predictions": {
                "pred_a_instrument_live": "native capability, replay, live term, and natural donor swap all pass",
                "pred_b_additive_oblique": "median normalized 2-axis Mobius interaction <=0.25 in every family",
                "pred_c_nonlinear_interaction": "with a live instrument, median normalized interaction >0.25 in either target family"
            }}


if __name__ == "__main__": print(json.dumps(compile_plan(), indent=2, sort_keys=True))
