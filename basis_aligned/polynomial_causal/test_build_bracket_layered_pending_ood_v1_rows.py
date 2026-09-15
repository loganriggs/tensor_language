#!/usr/bin/env python3
from collections import Counter
import json
from pathlib import Path

import tiktoken

import build_bracket_layered_pending_ood_v1_rows as builder


def test_fifth_construction_is_balanced_aligned_and_disjoint():
    rows = builder.build_rows(tiktoken.get_encoding("gpt2").encode)
    root = Path(__file__).parent
    old_rows = []
    for name in ("BRACKET_NESTED_PENDING_OOD_V1_ROWS.json", "BRACKET_TRIPLE_PENDING_OOD_V1_ROWS.json"):
        old_rows.extend(json.loads((root / name).read_text())["rows"])
    assert len(rows) == 72 and len({row["row_id"] for row in rows}) == 72
    assert Counter(row["program_role"] for row in rows) == {"target": 36, "control": 36}
    assert not {row["base_text"] for row in rows} & {row["base_text"] for row in old_rows}
    pairs = Counter()
    for row in rows:
        assert len(row["base_ids"]) == len(row["donor_ids"])
        assert row["base_open_position"] == row["donor_open_position"]
        if row["program_role"] == "target":
            pairs[(row["base_answer_id"], row["donor_answer_id"])] += 1
        else:
            assert row["base_answer_id"] == row["donor_answer_id"]
    assert len(pairs) == 3 and set(pairs.values()) == {12}


def test_builder_is_model_blind():
    source = Path(builder.__file__).read_text()
    assert "load_bilin18" not in source and "torch" not in source
    assert builder.OUT.name == "BRACKET_LAYERED_PENDING_OOD_V1_ROWS.json"
