#!/usr/bin/env python3
"""Frozen SELECT exporter and still-closed OOD authority for a six-vector bracket program."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_source_regions as shared


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "bracket.pending_opener.l13h8_ordered_pair_displacement_program_v1"
PRIOR_ART = ROOT / "circuits/prior_art/bracket_l13h8_ordered_pair_displacement_program_v1.json"
ROWS_PATH = ROOT / "pending_opener_three_value_fresh_rows_rung545.json"
PRIOR_ART_SHA256 = "caa0731326de41c0c36d3054203ae28947a7505ebdecd87d55e03d026e821007"
ROWS_FILE_SHA256 = "07b64d2e48a6ca67685c81d3475a064daba612d6fe7ff233efd5b6c157b940a9"
TARGET_FAMILIES = ("direct_three_value_type_substitution",
                   "completed_then_reopened_three_value_order")
CONTROL_FAMILIES = ("pending_type_preserved_surface_rewrite",
                    "pending_type_preserved_distance_extension",
                    "pending_type_preserved_nonopener_punctuation")
CLOSERS = (1, 8, 60)
ORDERED_PAIRS = tuple((a, b) for a in CLOSERS for b in CLOSERS if a != b)
PATCH_LAYER, PATCH_HEAD = 13, 8


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_rows() -> list[dict]:
    if _sha(PRIOR_ART) != PRIOR_ART_SHA256 or _sha(ROWS_PATH) != ROWS_FILE_SHA256:
        raise ValueError("frozen prior art or R545 row authority changed")
    return json.loads(ROWS_PATH.read_text())["rows"]


def _decorate(row: dict) -> dict:
    row = dict(row)
    row["program_role"] = "target" if row["family_id"] in TARGET_FAMILIES else "control"
    for side in ("base", "donor"):
        ids, answer = row[f"{side}_ids"], row[f"{side}_answer_id"]
        source = shared.semantic_open_position(ids, answer)
        if answer not in CLOSERS or not 0 < source < len(ids) - 1:
            raise ValueError("invalid closer or semantic opener position")
        row[f"{side}_open_position"] = source
    return row


def build_export_rows() -> list[dict]:
    rows = [_decorate(row) for row in _all_rows()
            if row["split"] == "SELECT" and row["family_id"] in TARGET_FAMILIES]
    counts = Counter()
    for row in rows:
        counts[(row["base_answer_id"], row["donor_answer_id"])] += 1
        counts[(row["donor_answer_id"], row["base_answer_id"])] += 1
    if len(rows) != 72 or set(counts) != set(ORDERED_PAIRS) or set(counts.values()) != {24}:
        raise ValueError("SELECT ordered-pair balance changed")
    return rows


def build_ood_rows() -> list[dict]:
    rows = [_decorate(row) for row in _all_rows() if row["split"] == "OOD"]
    families = Counter(row["family_id"] for row in rows)
    if len(rows) != 180 or set(families) != set(TARGET_FAMILIES + CONTROL_FAMILIES) \
            or set(families.values()) != {36} or len({row["row_id"] for row in rows}) != 180:
        raise ValueError("OOD authority balance changed")
    return rows


def compile_export_plan() -> dict:
    rows = build_export_rows()
    return {
        "schema": "bracket_l13h8_ordered_pair_displacement_export_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "rows_file_sha256": ROWS_FILE_SHA256,
        "split": "SELECT",
        "rows": len(rows), "endpoints": 2 * len(rows),
        "target_families": list(TARGET_FAMILIES),
        "ordered_pairs": [f"{a}->{b}" for a, b in ORDERED_PAIRS],
        "construction": "mean exact donor opener-term minus recipient opener-term per ordered closer pair",
        "prototype_width": 1152, "stored_scalars": 6 * 1152,
        "ood_rows_consumed": 0, "ood_activations_consumed": 0,
        "ood_logits_consumed": 0, "ood_outcomes_consumed": 0,
        "price": {"model_forwards": 1, "example_evaluations": 144,
                  "backwards": 0, "parameter_updates": 0},
    }


if __name__ == "__main__":
    print(json.dumps(compile_export_plan(), indent=2, sort_keys=True))
