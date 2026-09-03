from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("numeric_sequence_semantic_positions_rung577.py")
SPEC = importlib.util.spec_from_file_location("r577_positions", MODULE_PATH)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def test_endpoint_mapping_uses_last_three_semantic_commas():
    ids = mod.ENC.encode("For the harp, the numbers are 8, 9, 10,")
    item = mod.endpoint_mapping(ids)
    assert [value["numeric_value"] for value in item["value_positions"]] == [8, 9, 10]
    assert [value["representation"] for value in item["value_positions"]] == ["digit"] * 3
    assert item["query_position"] == len(ids) - 1


def test_number_words_and_repeated_values_are_semantic_tokens():
    for text, expected in (
        ("The harp number sequence is eight, nine, ten,", [8, 9, 10]),
        ("The harp number sequence is eight, eight, eight,", [8, 8, 8]),
    ):
        item = mod.endpoint_mapping(mod.ENC.encode(text))
        assert [value["numeric_value"] for value in item["value_positions"]] == expected
        assert [value["representation"] for value in item["value_positions"]] == ["number_word"] * 3


def test_generated_authority_is_complete_and_outcome_free():
    data = json.loads(mod.OUT.read_text())
    assert data["row_count"] == 432
    assert data["r575_endpoint_mappings_reproduced"] == 480
    assert data["model_loaded"] is False and data["model_forwards"] == 0
    assert data["outcomes_opened"] == []
    assert all(data[key] is True for key in (
        "all_values_single_token", "all_queries_final_commas",
        "all_rows_have_three_semantic_values",
    ))
    for split, n in (("FIT", 32), ("SELECT", 16)):
        assert set(data["counts"][split]) == set(mod.FAMILIES)
        assert set(data["counts"][split].values()) == {n}
    for row in data["records"]:
        assert len(row["endpoints"]["base"]["value_positions"]) == 3
        assert len(row["endpoints"]["donor"]["value_positions"]) == 3
