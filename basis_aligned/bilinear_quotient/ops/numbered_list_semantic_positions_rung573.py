#!/usr/bin/env python3
"""Freeze semantic list-label token positions for R573; CPU only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "bilinear_quotient"
ROWS = BQ / "increment_two_hypothesis_rows_rung567.json"
OUT = BQ / "numbered_list_semantic_positions_rung573.json"
ENC = tiktoken.get_encoding("gpt2")
EXPECTED_ROWS_SHA = "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_positions(text: str, ids: list[int]) -> list[dict]:
    lines = text.splitlines(keepends=True)
    assert lines and all(line.endswith("\n") for line in lines)
    positions, prefix = [], ""
    for line_index, line in enumerate(lines):
        label = line.split(".", 1)[0]
        assert label.isdigit()
        prefix_ids = ENC.encode(prefix)
        line_ids = ENC.encode(line)
        position = len(prefix_ids)
        assert ids[position] == line_ids[0]
        assert ENC.decode([ids[position]]).strip() == label
        positions.append({"line_index": line_index, "token_position": position,
                          "label": int(label), "token_id": ids[position]})
        prefix += line
    assert ENC.encode(prefix) == ids
    return positions


def main() -> None:
    assert sha256(ROWS) == EXPECTED_ROWS_SHA
    document = json.loads(ROWS.read_text())
    rows = [row for row in document["rows"] if row["hypothesis_id"] == "numbered_list_index_successor"
            and row["split"] in {"FIT", "SELECT"}]
    mappings = []
    for row in rows:
        endpoints = {}
        for endpoint in ("base", "donor"):
            positions = label_positions(row[f"{endpoint}_text"], row[f"{endpoint}_ids"])
            endpoints[endpoint] = {"sequence_length": len(row[f"{endpoint}_ids"]),
                                   "final_query_position": len(row[f"{endpoint}_ids"]) - 1,
                                   "label_positions": positions,
                                   "final_label_position": positions[-1]["token_position"]}
            assert row[f"{endpoint}_ids"][-1] == ENC.encode("\n")[0]
        assert len(endpoints["base"]["label_positions"]) == len(endpoints["donor"]["label_positions"])
        mappings.append({"row_id": row["row_id"], "group_id": row["group_id"], "split": row["split"],
                         "family_id": row["family_id"], "endpoints": endpoints})
    assert len(mappings) == 288
    assert sum(item["split"] == "FIT" for item in mappings) == 192
    assert sum(item["split"] == "SELECT" for item in mappings) == 96
    result = {"schema": "numbered_list_semantic_positions_rung573_v1",
              "rows_sha256": sha256(ROWS), "row_count": len(mappings),
              "fit_rows": 192, "select_rows": 96,
              "all_queries_are_final_newlines": True, "all_labels_are_single_semantic_tokens": True,
              "all_base_donor_line_counts_match": True, "mappings": mappings,
              "model_loaded": False, "model_forwards": 0, "outcomes_opened": []}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("row_count", "fit_rows", "select_rows",
                                                   "all_queries_are_final_newlines", "all_labels_are_single_semantic_tokens",
                                                   "model_forwards")}, indent=2))


if __name__ == "__main__":
    main()
