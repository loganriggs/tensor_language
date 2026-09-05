#!/usr/bin/env python3
"""Frozen R545 FINAL_TEST authority for exact pair-centered opener terms."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_source_regions as shared


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "bracket.pending_opener.l13h8_pair_centered_open_term_final_test_v1"
ROWS_PATH = ROOT / "pending_opener_three_value_fresh_rows_rung545.json"
RECEIPT_PATH = ROOT / "circuits/prior_art/bracket_l13h8_pair_centered_open_term_final_test_v1.json"
ROWS_FILE_SHA256 = "07b64d2e48a6ca67685c81d3475a064daba612d6fe7ff233efd5b6c157b940a9"
RECEIPT_SHA256 = "8910899470d14cd7190c290a506307a562017dde0dfc63105447d57ff8b85f63"
ROWS_SHA256 = "6ed5098cab797c5ef372a2dc111e57753917090c246f870d776e6fc4d51e1608"
PATCH_LAYER, PATCH_HEAD = 13, 8
CLOSERS = (8, 60, 1)
TARGET_FAMILIES = ("direct_three_value_type_substitution",
                   "completed_then_reopened_three_value_order")
CONTROL_FAMILIES = ("pending_type_preserved_surface_rewrite",
                    "pending_type_preserved_distance_extension",
                    "pending_type_preserved_nonopener_punctuation")
FAMILIES = TARGET_FAMILIES + CONTROL_FAMILIES


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows() -> list[dict]:
    if _file_sha(ROWS_PATH) != ROWS_FILE_SHA256 or _file_sha(RECEIPT_PATH) != RECEIPT_SHA256:
        raise ValueError("frozen R545 authority or prior-art receipt changed")
    rows = [dict(row) for row in json.loads(ROWS_PATH.read_text())["rows"]
            if row["split"] == "FINAL_TEST"]
    if len(rows) != 180 or len({row["row_id"] for row in rows}) != 180 \
            or shared.digest(rows) != ROWS_SHA256:
        raise ValueError("R545 FINAL_TEST identity changed")
    counts = {}
    endpoints = set()
    for row in rows:
        family = row["family_id"]
        if family not in FAMILIES:
            raise ValueError("unexpected family")
        row["role"] = "target" if family in TARGET_FAMILIES else "control"
        for side in ("base", "donor"):
            ids, answer = row[f"{side}_ids"], row[f"{side}_answer_id"]
            if tuple(ids) in endpoints or answer not in CLOSERS:
                raise ValueError("duplicate endpoint or unexpected closer")
            endpoints.add(tuple(ids))
            position = shared.semantic_open_position(ids, answer)
            if not 0 < position < len(ids) - 1:
                raise ValueError("semantic opener source is invalid")
            row[f"{side}_open_position"] = position
        if (row["base_answer_id"] != row["donor_answer_id"]) != (row["role"] == "target"):
            raise ValueError("answer-change role mismatch")
        key = (family, row["base_answer_id"], row["donor_answer_id"])
        counts[key] = counts.get(key, 0) + 1
    if len(endpoints) != 360 or any(sum(v for (f, _b, _d), v in counts.items() if f == family) != 36
                                    for family in FAMILIES):
        raise ValueError("FINAL_TEST balance changed")
    for family in TARGET_FAMILIES:
        if sorted(v for (f, _b, _d), v in counts.items() if f == family) != [6] * 6:
            raise ValueError("target ordered-pair balance changed")
    return rows


ROWS = build_rows()


def compile_plan() -> dict:
    return {
        "schema": "bracket_l13h8_pair_centered_open_term_final_test_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "receipt_sha256": RECEIPT_SHA256,
        "rows_file_sha256": ROWS_FILE_SHA256,
        "final_test_rows_sha256": ROWS_SHA256,
        "rows": 180, "endpoints": 360,
        "opened_splits": ["FINAL_TEST"], "closed_splits": ["OOD"],
        "conditions": ["native", "native_replay", "complete_head_donor_swap",
                       "open_term_donor_swap", "pair_centered_contrast_removed"],
        "capability_first": True,
        "price": {"model_forwards": 5, "example_evaluations": 1800,
                  "backwards": 0, "parameter_updates": 0},
        "active_price_if_capability_fails": {"model_forwards": 1,
                                              "example_evaluations": 360,
                                              "backwards": 0, "parameter_updates": 0},
        "bars": {
            "native_target_cell_accuracy_min": .75,
            "native_control_cell_accuracy_min": .75,
            "native_target_family_mean_margin_min": 0.0,
            "native_replay_max_absolute_logit_error_max": 1e-5,
            "complete_head_target_cell_positive_fraction_min": .75,
            "open_swap_target_cell_positive_fraction_min": .75,
            "open_swap_median_fraction_complete_each_family_direction_min": .50,
            "midpoint_positive_margin_and_ce_damage_fraction_each_family_direction_min": .75,
            "midpoint_median_margin_fraction_complete_each_family_direction_min": .50,
            "control_answer_preservation_each_family_direction_arm_min": .75,
            "control_effect_fraction_of_smallest_target_cell_max": .25,
        },
        "frozen_predictions": {
            "pred_a": "native capability, exact replay, and complete-head target ceilings pass",
            "pred_b": "opener-term interchange transfers delimiter type and pair-centered contrast removal is selectively necessary",
            "pred_c": "interchange transfers but selective pair-centered necessity fails",
            "pred_d": "live instrument but no held-out opener-term transfer",
        },
        "outcome_reads": [],
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
