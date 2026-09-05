#!/usr/bin/env python3
# BQLANE: cpu
"""Create-only v3 authority repairing the step-two registered preference."""

from __future__ import annotations

from collections import Counter
import copy
import json

import circuit_fast_screen_candidate_attn8_h3_h7_cross_behavior_factor_interchange as v2


SCHEMA = "attn8_h3_h7_cross_behavior_factor_interchange_authority_v3_control_repair"
EXPECTED_ROWS_SHA256 = "a560839c18c2cd1b9203d24e7f19cd0553ae9ea315ab69c89ef04803d2f45d7e"
canonical_sha256 = v2.canonical_sha256
ENC = v2.ENC


def _build_unvalidated():
    rows = copy.deepcopy(v2.build_rows())
    for row in rows:
        row["controls"] = copy.deepcopy(row["controls"])
        row["schema"] = SCHEMA
        row["row_id"] = canonical_sha256([
            SCHEMA, row["group_id"], row["split"], row["recipient_format"], row["direction"]])
        step = row["controls"]["step_two"]
        old_answer = step["answer_id"]
        step["answer_id"] = step["preference_foil_id"]
        step["preference_foil_id"] = old_answer
        step["answer_text"] = ENC.decode([step["answer_id"]])
    return rows


def validate_rows(rows, verify_frozen_hash=True):
    rows = list(rows)
    if len(rows) != 32 or len({row["row_id"] for row in rows}) != 32:
        raise ValueError("v3 authority must have 32 unique directional rows")
    counts = Counter((row["split"], row["recipient_format"], row["direction"]) for row in rows)
    if len(counts) != 8 or set(counts.values()) != {4}:
        raise ValueError("v3 split/format/direction balance changed")
    for row in rows:
        if row["schema"] != SCHEMA:
            raise ValueError("v3 authority schema changed")
        step = row["controls"]["step_two"]
        visible = int(ENC.decode([step["ids"][step["source_positions"][-1]]]).strip())
        if step["answer_text"].strip() != str(visible+1) \
                or ENC.decode([step["preference_foil_id"]]).strip() != str(visible+2):
            raise ValueError("step-two answer is not final label +1 against arithmetic +2")
        if ENC.encode(step["text"] + step["answer_text"]) != step["ids"] + [step["answer_id"]]:
            raise ValueError("repaired step-two answer does not tokenize jointly")
        for control_id in ("repeated_list_copy", "digit_copy"):
            current = row["controls"][control_id]
            if ENC.encode(current["text"] + current["answer_text"]) != current["ids"] + [current["answer_id"]]:
                raise ValueError("unchanged control tokenization changed")
    digest = canonical_sha256(rows)
    if verify_frozen_hash and digest != EXPECTED_ROWS_SHA256:
        raise ValueError(f"v3 authority digest changed: {digest}")
    return digest


def build_rows():
    rows = _build_unvalidated(); validate_rows(rows); return rows


def compile_plan():
    rows = build_rows()
    return {"schema": SCHEMA, "row_count": len(rows), "authority_sha256": validate_rows(rows),
            "repair": "step_two_answer_is_final_visible_label_plus_one_and_foil_is_plus_two",
            "model_loaded": False, "outcomes_opened": []}


if __name__ == "__main__": print(json.dumps(compile_plan(), indent=2, sort_keys=True))
