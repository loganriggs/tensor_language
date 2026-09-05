#!/usr/bin/env python3
"""Fresh CPU-buildable authority for the L13H8 source-region payload screen."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations

import tiktoken


CANDIDATE_ID = "bracket.pending_opener.l13h8_source_region_payload_factorial"
PATCH_LAYER, PATCH_HEAD = 13, 8
REGIONS = ("PREFIX", "OPEN", "POST")
CORNERS = tuple(tuple(region for bit, region in enumerate(REGIONS) if mask & (1 << bit))
                for mask in range(8))
TARGET_FAMILIES = ("direct_type", "completed_then_reopened")
CONTROL_FAMILIES = ("same_state_surface", "same_state_punctuation")
FAMILIES = TARGET_FAMILIES + CONTROL_FAMILIES
DELIMITERS = (("(", ")", "parenthesis"), ("[", "]", "square"), ('"', '"', "quote"))
ENC = tiktoken.get_encoding("gpt2")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def encode(text: str) -> list[int]:
    ids = ENC.encode(text)
    assert ENC.decode(ids) == text
    return ids


def answer_id(mark: str) -> int:
    ids = encode(mark)
    assert len(ids) == 1
    return ids[0]


def semantic_open_position(ids: list[int], answer: int) -> int:
    opener = {answer_id(")"): 357, answer_id("]"): 685, answer_id('"'): 366}[answer]
    positions = [index for index, token in enumerate(ids) if token == opener]
    assert positions
    return positions[-1]


def _row(group: str, family: str, base: str, donor: str, base_answer: str, donor_answer: str) -> dict:
    base_ids, donor_ids = encode(base), encode(donor)
    assert len(base_ids) == len(donor_ids), (family, len(base_ids), len(donor_ids))
    ba, da = answer_id(base_answer), answer_id(donor_answer)
    bo, do = semantic_open_position(base_ids, ba), semantic_open_position(donor_ids, da)
    assert bo == do and 0 < bo < len(base_ids) - 1
    regions = {
        "PREFIX": list(range(bo)),
        "OPEN": [bo],
        "POST": list(range(bo + 1, len(base_ids))),
    }
    assert sorted(sum(regions.values(), [])) == list(range(len(base_ids)))
    role = "target" if family in TARGET_FAMILIES else "control"
    assert (ba != da) == (role == "target")
    return {
        "row_id": digest({"group": group, "family": family, "base": base, "donor": donor}),
        "group_id": group,
        "split": "BASIC_SCREEN",
        "family_id": family,
        "role": role,
        "base_text": base,
        "donor_text": donor,
        "base_ids": base_ids,
        "donor_ids": donor_ids,
        "base_answer_id": ba,
        "donor_answer_id": da,
        "base_open_position": bo,
        "donor_open_position": do,
        "regions": regions,
    }


def build_rows() -> list[dict]:
    """Build 24 fresh pairs: every ordered delimiter pair in every family."""
    prefixes = ("The curator", "A botanist", "The surveyor", "One sculptor", "Our navigator", "The locksmith")
    words = (
        ("acorn", "brook", "canvas", "drum", "elm"),
        ("flask", "grove", "harp", "island", "kettle"),
        ("lantern", "meadow", "notebook", "orchard", "pebble"),
        ("quartz", "reef", "saddle", "thimble", "urn"),
        ("velvet", "well", "yarn", "zebra", "badge"),
        ("comet", "dune", "engine", "fern", "granite"),
    )
    ordered = [(left, right) for left in DELIMITERS for right in DELIMITERS if left != right]
    rows = []
    for index, (left, right) in enumerate(ordered):
        prefix, (w0, w1, w2, w3, w4) = prefixes[index], words[index]
        lo, lc, lname = left
        ro, rc, rname = right
        group = digest({"candidate": CANDIDATE_ID, "pair": (lname, rname), "index": index})
        tail = f"the {w0}, the {w1}, the {w2}, the {w3}, and the {w4} remained"
        rows.append(_row(group, "direct_type", f"{prefix} recorded {lo} {tail}",
                         f"{prefix} recorded {ro} {tail}", lc, rc))
        rows.append(_row(group, "completed_then_reopened",
                         f"{prefix} marked {lo} the {w0} {lc}, then recorded {ro} {tail}",
                         f"{prefix} marked {ro} the {w0} {rc}, then recorded {lo} {tail}", rc, lc))
        rows.append(_row(group, "same_state_surface",
                         f"{prefix} recorded {lo} the {w0} near the {w1}, the {w2}, the {w3}, and the {w4}",
                         f"{prefix} recorded {lo} the {w4} near the {w3}, the {w2}, the {w1}, and the {w0}", lc, lc))
        rows.append(_row(group, "same_state_punctuation",
                         f"{prefix} paused, then recorded {lo} {tail}",
                         f"{prefix} paused: then recorded {lo} {tail}", lc, lc))
    assert len(rows) == 24 and len({row["row_id"] for row in rows}) == 24
    return rows


ROWS = build_rows()
ROWS_SHA256 = digest(ROWS)
EXPECTED_ROWS_SHA256 = "98252680b5c407b30195f6f8dfecd993d967cb47f4b59b2069e7f1606b351c50"
assert ROWS_SHA256 == EXPECTED_ROWS_SHA256


def compile_plan(batch_size: int = 24) -> dict:
    chunks = (len(ROWS) + batch_size - 1) // batch_size
    conditions = ("native", "native_replay", "complete_head", *("payload_" + "+".join(c or ("NONE",)) for c in CORNERS))
    forwards = chunks * 2 * len(conditions)
    return {
        "schema": "bracket_source_region_payload_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "rows_sha256": ROWS_SHA256,
        "rows": len(ROWS),
        "groups": 6,
        "regions": list(REGIONS),
        "corners": [list(corner) for corner in CORNERS],
        "conditions": list(conditions),
        "batch_size": batch_size,
        "price": {"model_forwards": forwards, "example_evaluations": forwards * len(ROWS),
                  "backwards": 0, "parameter_updates": 0},
        "closed_splits": ["SELECT", "FINAL_TEST", "OOD"],
        "outcome_reads": [],
        "frozen_predictions": {
            "localized_payload": "OPEN+POST passes each target family/direction; PREFIX recovery <=0.25",
            "broad_or_distributed_payload": "PREFIX recovery >=0.50, or only ALL passes",
            "instrument_null": "ALL payload fails despite a live complete-head ceiling",
        },
        "bars": {
            "native_answer_positive_fraction_min": 0.75,
            "complete_head_target_positive_fraction_min": 0.75,
            "open_post_target_median_recovery_min": 0.50,
            "open_post_target_positive_fraction_min": 0.75,
            "prefix_target_mean_absolute_recovery_max": 0.25,
            "control_mean_absolute_closer_margin_change_max": 0.10,
            "control_mean_absolute_fraction_of_complete_head_max": 0.25,
            "native_replay_max_absolute_logit_error_max": 1e-5,
        },
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
