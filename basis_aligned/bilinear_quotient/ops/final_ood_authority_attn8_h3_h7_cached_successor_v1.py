#!/usr/bin/env python3
# BQLANE: cpu
"""Frozen FINAL_TEST/OOD full-family authority for the L8 H3+H7 cached-value test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numbered_list_semantic_positions_rung573 as r573
import numeric_sequence_semantic_positions_rung577 as r577


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "increment_two_hypothesis_rows_rung567.json"
STEP_TWO = ROOT / "numbered_list_conflict_confirmation_rung572_results.json"
LIST_MAP = ROOT / "numbered_list_semantic_positions_rung573.json"
NUMERIC_MAP = ROOT / "numeric_sequence_semantic_positions_rung577.json"
HASHES = {
    ROWS: "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053",
    STEP_TWO: "3df046bdcc4fa4387a2dbef084ed732c5f6a05232b7fa64072af3cd4939daea1",
    LIST_MAP: "b4f4f8e58c03deb3a015141656572dcf1fd0fe0c12027c358b441459c245c16b",
    NUMERIC_MAP: "a6a98715617cf91971655c252553f42d45b59937ecfbf46722b518333721de1d",
}
POSITIVE = "list_step_two_conflict"
NEGATIVE = ("sequence_digit_copy_control", "sequence_word_copy_control")
SECONDARY = "list_repeated_index_control"
FAMILIES = (POSITIVE,) + NEGATIVE + (SECONDARY,)
SPLITS = ("FINAL_TEST", "OOD")
ROW_ID_DIGESTS = {
    ("list_step_two_conflict", "FINAL_TEST"): "6b1342ba4f04952f5a719242b2cbcac3ce80d9dfa5ae5c43e149cb26bdbd8a68",
    ("list_step_two_conflict", "OOD"): "b343c50c1a762c270088befbe715cf5b9aa13e8587d06940273c02dfc908b70f",
    ("sequence_digit_copy_control", "FINAL_TEST"): "b625486b459522fc85f8bdb27a4e7fb26146ee62151469470b9f70d9b597b809",
    ("sequence_digit_copy_control", "OOD"): "24c7cea6b6abfdddbe62f9568b5eb915edfb527000866cd993f9a0ff0eee56e4",
    ("sequence_word_copy_control", "FINAL_TEST"): "dcd271d72b52e98340fbd909db85c22c983095b78280b403bf01406800c9916a",
    ("sequence_word_copy_control", "OOD"): "ee0d611de4d71f093735f918d7f9f45bbd57d5419ef40625c91a46d3f79dc136",
    ("list_repeated_index_control", "FINAL_TEST"): "5ddae5cfa9eaf57d4e3868224a734ceb4af2b1c6512d0b9efa0a6d8b1ad4cab3",
    ("list_repeated_index_control", "OOD"): "2df78070704b4369c72d433a2448fd9670780ddacaa4c7ae086e100369e8019f",
}
EXPECTED_PAIRING_SHA256 = "e407d790de9f5a1e7fe17597bb7175d1b5e2ddfe56f3ac4cd702396ac846a869"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _endpoint(row, mapping, side):
    item = mapping["endpoints"][side]
    positions = item.get("label_positions") or item.get("value_positions")
    final = positions[-1]
    return {
        "ids": row[f"{side}_ids"], "text": row[f"{side}_text"],
        "answer_id": row[f"{side}_answer_id"], "answer_text": row[f"{side}_answer"],
        "source_positions": [entry["token_position"] for entry in positions],
        "query_position": item.get("final_query_position", item.get("query_position")),
        "final_value": final.get("label", final.get("numeric_value")),
    }


def _new_mapping(row):
    """Apply the frozen R573/R577 semantic parser to a previously unopened row."""
    if row["family_id"].startswith("list_"):
        endpoints = {}
        for side in ("base", "donor"):
            positions = r573.label_positions(row[f"{side}_text"], row[f"{side}_ids"])
            endpoints[side] = {"label_positions": positions,
                "final_query_position": len(row[f"{side}_ids"]) - 1}
    else:
        endpoints = {side: r577.endpoint_mapping(row[f"{side}_ids"])
                     for side in ("base", "donor")}
    return {"endpoints": endpoints}


def build_pairs(verify_frozen=True):
    for path, expected in HASHES.items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input changed: {path}")
    rows = json.loads(ROWS.read_text())["rows"]
    output = []
    for family in FAMILIES:
        for split in SPLITS:
            selected = [r for r in rows if r["family_id"] == family and r["split"] == split]
            digest = hashlib.sha256(json.dumps(
                [r["row_id"] for r in selected], separators=(",", ":")).encode()).hexdigest()
            if digest != ROW_ID_DIGESTS[(family, split)]:
                raise RuntimeError(f"full-family row order changed: {family} {split}")
            by_value = {}
            frozen = {row["row_id"]: _new_mapping(row) for row in selected}
            for row in selected:
                value = _endpoint(row, frozen[row["row_id"]], "base")["final_value"]
                by_value.setdefault(value, []).append(row)
            values = sorted(by_value)
            if len(values) != 2 or values[1] != values[0] + 1 \
                    or len(by_value[values[0]]) != len(by_value[values[1]]):
                raise RuntimeError(f"adjacent complete value sets changed: {family} {split}")
            lower = sorted(by_value[values[0]], key=lambda r: r["row_id"])
            higher = sorted(by_value[values[1]], key=lambda r: r["row_id"])
            for pair_index, (low, high) in enumerate(zip(lower, higher)):
                for side in ("base", "donor"):
                    low_ep = _endpoint(low, frozen[low["row_id"]], side)
                    high_ep = _endpoint(high, frozen[high["row_id"]], side)
                    if (len(low_ep["source_positions"]) != 3 or
                            len(high_ep["source_positions"]) != 3 or
                            low_ep["query_position"] != len(low_ep["ids"]) - 1 or
                            high_ep["query_position"] != len(high_ep["ids"]) - 1 or
                            high_ep["final_value"] != low_ep["final_value"] + 1):
                        raise RuntimeError("paired source maps or adjacent semantics changed")
                    for direction, recipient, donor in (
                        ("lower_to_higher", low_ep, high_ep),
                        ("higher_to_lower", high_ep, low_ep),
                    ):
                        output.append({
                            "pair_id": canonical([family, split, pair_index, side, direction]),
                            "family_id": family, "split": split, "surface_side": side,
                            "direction": direction, "lower_row_id": low["row_id"],
                            "higher_row_id": high["row_id"], "recipient": recipient,
                            "donor": donor,
                        })
    digest = canonical(output)
    if verify_frozen and digest != EXPECTED_PAIRING_SHA256:
        raise RuntimeError(f"FINAL_TEST/OOD pairing changed: {digest}")
    return output


def compile_plan():
    pairs = build_pairs()
    return {
        "schema": "attn8_h3_h7_cached_successor_final_ood_v1_authority",
        "pair_count": len(pairs), "pairing_sha256": canonical(pairs),
        "positive_family": POSITIVE, "negative_families": list(NEGATIVE),
        "secondary_family": SECONDARY, "splits": list(SPLITS),
        "model_loaded": False, "outcomes_opened": [],
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
