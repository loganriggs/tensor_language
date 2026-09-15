import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BUILDER = HERE / "build_successor_pointer_prefixed_length6_v1_rows.py"
OUT = HERE / "SUCCESSOR_POINTER_PREFIXED_LENGTH6_V1_ROWS.json"
BASE = HERE / "SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_ROWS.json"


def test_frozen_rows_are_complete_and_disjoint():
    data = json.loads(OUT.read_text())
    base = json.loads(BASE.read_text())
    assert data["model_loaded"] is False and data["outcomes_opened"] is False
    assert data["row_count"] == 96 and data["endpoint_count"] == 192
    assert len({row["row_id"] for row in data["rows"]}) == 96
    assert not ({tuple(row["token_ids"]) for row in data["rows"]} & {tuple(row["token_ids"]) for row in base["rows"]})
    groups = {}
    for row in data["rows"]:
        key = (row["prefix_id"], row["family"], row["final_index"])
        groups.setdefault(key, set()).add(row["condition"])
        assert row["token_ids"][row["last_position"]] == row["recipient_token_id"]
        assert row["query_position"] == len(row["token_ids"]) - 1
    assert len(groups) == 32
    assert all(value == {"coherent", "early_swap_control", "late_swap_incoherent"} for value in groups.values())


def test_builder_refuses_to_overwrite_frozen_rows():
    completed = subprocess.run([sys.executable, str(BUILDER)], text=True, capture_output=True)
    assert completed.returncode != 0
    assert "FileExistsError" in completed.stderr
