#!/usr/bin/env python3
"""R575 CPU-only semantic source audit for active factor-removal families."""

# BQLANE: cpu

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "increment_two_hypothesis_rows_rung567.json"
LIST_POSITIONS = ROOT / "numbered_list_semantic_positions_rung573.json"
OUT = ROOT / "numeric_factor_removal_positions_rung575.json"
ROWS_SHA256 = "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053"
LIST_POSITIONS_SHA256 = "b4f4f8e58c03deb3a015141656572dcf1fd0fe0c12027c358b441459c245c16b"
LIST_FAMILIES = (
    "list_two_line_state_shift", "list_three_line_state_shift", "list_surface_preserved",
    "list_middle_index_break", "list_repeated_index_control", "list_step_two_conflict",
)
SEQUENCE_FAMILIES = (
    "sequence_digit_state_shift", "sequence_word_state_shift", "sequence_cross_format_shift",
    "sequence_digit_copy_control", "sequence_word_copy_control",
)
ENC = tiktoken.get_encoding("gpt2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(ROWS) != ROWS_SHA256 or sha256(LIST_POSITIONS) != LIST_POSITIONS_SHA256:
        raise RuntimeError("frozen semantic authority changed")
    document = json.loads(ROWS.read_text())
    old = {item["row_id"]: item for item in json.loads(LIST_POSITIONS.read_text())["mappings"]}
    selected = [row for row in document["rows"] if row["split"] in {"FIT", "SELECT"}
                and row["family_id"] in LIST_FAMILIES + SEQUENCE_FAMILIES]
    records = []
    for row in selected:
        endpoints = {}
        for endpoint in ("base", "donor"):
            ids = row[f"{endpoint}_ids"]
            if row["family_id"] in LIST_FAMILIES:
                authority = old[row["row_id"]]["endpoints"][endpoint]
                source = authority["final_label_position"]
                query = authority["final_query_position"]
                source_kind = "final_visible_list_label"
                assert query == len(ids) - 1 and ids[query] == ENC.encode("\n")[0]
            else:
                source, query = len(ids) - 2, len(ids) - 1
                source_kind = "final_visible_sequence_value"
                assert ids[query] == ENC.encode(",")[0]
                assert len(ENC.decode([ids[source]]).strip()) > 0
                if row["family_id"] in {"sequence_digit_copy_control", "sequence_word_copy_control"}:
                    assert ids[source] == row[f"{endpoint}_answer_id"]
            endpoints[endpoint] = {
                "sequence_length": len(ids), "source_position": source,
                "source_token_id": ids[source], "source_token_text": ENC.decode([ids[source]]),
                "query_position": query, "query_token_id": ids[query],
                "source_kind": source_kind,
            }
        records.append({"row_id": row["row_id"], "group_id": row["group_id"],
                        "hypothesis_id": row["hypothesis_id"], "family_id": row["family_id"],
                        "split": row["split"], "endpoints": endpoints})
    counts = {}
    for split in ("FIT", "SELECT"):
        counts[split] = {family: sum(record["split"] == split and record["family_id"] == family
                                    for record in records)
                         for family in LIST_FAMILIES + SEQUENCE_FAMILIES}
    result = {"rung": 575, "schema": "numeric_factor_removal_semantic_positions_v1",
              "rows_sha256": ROWS_SHA256, "list_positions_sha256": LIST_POSITIONS_SHA256,
              "row_count": len(records), "counts": counts, "records": records,
              "all_sources_single_token": True, "all_queries_are_final_separator": True,
              "model_loaded": False, "model_forwards": 0, "outcomes_opened": []}
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: result[key] for key in ("row_count", "counts",
                                                   "all_sources_single_token",
                                                   "all_queries_are_final_separator")}, indent=2))


if __name__ == "__main__":
    main()
