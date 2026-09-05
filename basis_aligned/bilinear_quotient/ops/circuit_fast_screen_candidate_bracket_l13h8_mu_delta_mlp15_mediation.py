#!/usr/bin/env python3
"""Fresh BASIC triplets for fixed-hypothesis L13H8 factor mediation by MLP15."""

from __future__ import annotations

import json

import circuit_fast_screen_candidate_bracket_l13h8_shared_contrast as parent
import circuit_fast_screen_candidate_bracket_l13h8_source_regions as shared


CANDIDATE_ID = "bracket.pending_opener.l13h8_mu_delta_mlp15_mediation"
PATCH_LAYER, PATCH_HEAD, MEDIATOR_LAYER = parent.PATCH_LAYER, parent.PATCH_HEAD, 15
DELIMITERS, FAMILIES = parent.DELIMITERS, parent.FAMILIES
TARGET_FAMILIES, STABILITY_FAMILIES = parent.TARGET_FAMILIES, parent.CONTROL_FAMILIES
PRIOR_ART_SHA256 = "916c38eaf884be3e498b0a53e2e7c3d48fd249e02373c1f4e8d1970608bbd17f"


def build_rows() -> list[dict]:
    bundles = (
        ("The watchmaker", ("amber", "birch", "cello", "delta", "elm")),
        ("A navigator", ("falcon", "garnet", "heron", "ivory", "kelp")),
    )
    rows = []
    for bundle, (prefix, words) in enumerate(bundles):
        w0, w1, w2, w3, w4 = words
        tail = f"the {w0}, the {w1}, the {w2}, the {w3}, and the {w4} remained"
        direct_group = shared.digest({"candidate": CANDIDATE_ID, "bundle": bundle,
                                      "family": "direct_type"})
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
                rows.append({
                    "row_id": shared.digest({"group": group, "delimiter": delimiter_name}),
                    "group_id": group, "bundle_id": bundle,
                    "matched_direct_group_id": direct_group, "split": "FRESH_BASIC",
                    "family_id": family,
                    "role": "target" if family in TARGET_FAMILIES else "stability_rewrite",
                    "delimiter_index": delimiter_index, "delimiter_name": delimiter_name,
                    "text": text, "ids": ids, "answer_id": answer,
                    "open_position": shared.semantic_open_position(ids, answer),
                    "final_position": len(ids) - 1,
                })
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    assert len(rows) == 24 and len(groups) == 8
    for triplet in groups.values():
        assert {row["delimiter_index"] for row in triplet} == {0, 1, 2}
        assert len({len(row["ids"]) for row in triplet}) == 1
        assert len({row["open_position"] for row in triplet}) == 1
        assert len({row["final_position"] for row in triplet}) == 1
    return rows


ROWS = build_rows()
ROWS_SHA256 = shared.digest(ROWS)
EXPECTED_ROWS_SHA256 = "61f8db59d41863f2cc48b141f7c14f73a6774c63bbbca3b41ae0d873dbd18ba7"
assert ROWS_SHA256 == EXPECTED_ROWS_SHA256


def compile_plan() -> dict:
    return {
        "schema": "bracket_l13h8_mu_delta_mlp15_mediation_plan_v1",
        "candidate_id": CANDIDATE_ID, "prior_art_sha256": PRIOR_ART_SHA256,
        "rows_sha256": ROWS_SHA256, "rows": 24, "triplet_groups": 8,
        "conditions": ["native", "native_replay", "remove_mu", "remove_delta",
                       "remove_mu_restore_native_mlp15_final",
                       "remove_delta_restore_native_mlp15_final"],
        "price": {"model_forwards": 6, "example_evaluations": 144,
                  "backwards": 0, "parameter_updates": 0},
        "opened_splits": ["FRESH_BASIC"], "closed_splits": ["TEST", "OOD"],
        "outcome_reads": [],
        "estimand": "d=C(native)-C(remove); r=C(restore)-C(remove); projection=dot(r,d)/(dot(d,d)+1e-8); cosine=dot(r,d)/(norm(r)*norm(d)+1e-8)",
        "bars": {
            "native_positive_fraction_each_family_min": 0.75,
            "native_replay_max_absolute_logit_error_max": 1e-5,
            "median_live_centered_effect_norm_each_factor_family_min": 0.05,
            "median_projection_recovery_each_factor_family_min": 0.25,
            "median_rescue_cosine_each_factor_family_min": 0.50,
            "positive_projection_fraction_each_factor_family_min": 0.75,
        },
        "frozen_predictions": {
            "pred_a": "native capability and exact replay hold, and both mu and delta removals have live centered closer-vector effects in every construction and stability rewrite",
            "pred_b": "MLP15 substantially mediates both live factors: projection recovery >=0.25, rescue cosine >=0.50, and positive projection fraction >=0.75 separately in every target construction and stability rewrite",
            "pred_c": "with a live instrument, at least one factor in a target construction misses an MLP15 mediation bar, supporting distributed downstream use or bypass",
        },
        "interpretation_limit": "BASIC mediation screen only; stability rewrites test robustness, not selectivity",
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
