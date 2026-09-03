#!/usr/bin/env python3
"""R577 CPU-only semantic positions for the numeric-sequence circuit.

This extends the R575 final-source audit to all three comma-separated numeric
values and all nine FIT/SELECT families.  It opens no model or result.
"""

# BQLANE: cpu

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "increment_two_hypothesis_rows_rung567.json"
R575 = ROOT / "numeric_factor_removal_positions_rung575.json"
OUT = ROOT / "numeric_sequence_semantic_positions_rung577.json"
ROWS_SHA256 = "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053"
R575_SHA256 = "3663ebc48e5dca1ff336cb0627fc43c6db8d7d6e1666b81d7631ab150168dd4b"
HYPOTHESIS = "numeric_sequence_continuation"
FAMILIES = (
    "sequence_digit_state_shift",
    "sequence_word_state_shift",
    "sequence_cross_format_shift",
    "sequence_digit_surface_preserved",
    "sequence_word_surface_preserved",
    "sequence_middle_value_break",
    "sequence_digit_copy_control",
    "sequence_word_copy_control",
    "sequence_step_two_conflict",
)
ENC = tiktoken.get_encoding("gpt2")
COMMA_ID = ENC.encode(",")[0]
WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty",
)
WORD_VALUE = {word: value for value, word in enumerate(WORDS)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_value(token_id: int) -> tuple[int, str, str]:
    text = ENC.decode([token_id])
    stripped = text.strip().lower()
    if stripped.isdigit():
        return int(stripped), "digit", text
    if stripped in WORD_VALUE:
        return WORD_VALUE[stripped], "number_word", text
    raise RuntimeError(f"semantic source is not one numeric token: {token_id} {text!r}")


def endpoint_mapping(ids: list[int]) -> dict:
    commas = [index for index, token in enumerate(ids) if token == COMMA_ID]
    if len(commas) < 3 or commas[-1] != len(ids) - 1:
        raise RuntimeError("numeric sequence does not end in its third semantic comma")
    semantic_commas = commas[-3:]
    positions = [index - 1 for index in semantic_commas]
    if min(positions) < 0:
        raise RuntimeError("missing numeric token before comma")
    values = []
    for ordinal, position in enumerate(positions):
        value, representation, text = numeric_value(ids[position])
        values.append({
            "ordinal": ordinal,
            "token_position": position,
            "token_id": ids[position],
            "token_text": text,
            "numeric_value": value,
            "representation": representation,
        })
    return {
        "sequence_length": len(ids),
        "query_position": len(ids) - 1,
        "query_token_id": ids[-1],
        "value_positions": values,
        "final_value_position": positions[-1],
    }


def main() -> None:
    if sha256(ROWS) != ROWS_SHA256 or sha256(R575) != R575_SHA256:
        raise RuntimeError("frozen R567/R575 authority changed")
    document = json.loads(ROWS.read_text())
    r575 = {item["row_id"]: item for item in json.loads(R575.read_text())["records"]}
    rows = [row for row in document["rows"] if row["hypothesis_id"] == HYPOTHESIS
            and row["split"] in {"FIT", "SELECT"}]
    assert len(rows) == 432 and all(row["family_id"] in FAMILIES for row in rows)
    records = []
    overlap_checked = 0
    for row in rows:
        endpoints = {endpoint: endpoint_mapping(row[f"{endpoint}_ids"])
                     for endpoint in ("base", "donor")}
        if row["row_id"] in r575:
            for endpoint in ("base", "donor"):
                old = r575[row["row_id"]]["endpoints"][endpoint]
                new = endpoints[endpoint]
                assert old["source_position"] == new["final_value_position"]
                assert old["query_position"] == new["query_position"]
                assert old["source_token_id"] == new["value_positions"][-1]["token_id"]
                overlap_checked += 1
        records.append({
            "row_id": row["row_id"],
            "group_id": row["group_id"],
            "hypothesis_id": row["hypothesis_id"],
            "family_id": row["family_id"],
            "role": row["role"],
            "split": row["split"],
            "endpoints": endpoints,
        })
    counts = {
        split: {family: sum(item["split"] == split and item["family_id"] == family
                           for item in records) for family in FAMILIES}
        for split in ("FIT", "SELECT")
    }
    result = {
        "rung": 577,
        "schema": "numeric_sequence_semantic_positions_rung577_v1",
        "rows_sha256": ROWS_SHA256,
        "r575_sha256": R575_SHA256,
        "row_count": len(records),
        "counts": counts,
        "r575_endpoint_mappings_reproduced": overlap_checked,
        "records": records,
        "all_values_single_token": True,
        "all_queries_final_commas": True,
        "all_rows_have_three_semantic_values": True,
        "model_loaded": False,
        "model_forwards": 0,
        "outcomes_opened": [],
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: result[key] for key in (
        "row_count", "counts", "r575_endpoint_mappings_reproduced",
        "all_values_single_token", "all_queries_final_commas",
    )}, indent=2))


if __name__ == "__main__":
    main()
