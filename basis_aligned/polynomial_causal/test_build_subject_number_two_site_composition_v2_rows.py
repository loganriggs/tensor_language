import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE / "build_subject_number_two_site_composition_v2_rows.py"
OUT = HERE / "SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V2_ROWS.json"
V1 = HERE / "SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V1_ROWS.json"


def test_authority_is_strong_direction_and_well_formed():
    data = json.loads(OUT.read_text())
    assert data["outcomes_opened"] is False and data["model_loaded"] is False
    assert data["row_count"] == 16 and data["site_count"] == 32
    assert data["number_pair_counts"] == {
        "singular|singular": 16, "singular|plural": 0,
        "plural|singular": 0, "plural|plural": 0,
    }
    assert len({row["row_id"] for row in data["rows"]}) == 16
    assert len({tuple(row["token_ids"]) for row in data["rows"]}) == 16
    for row in data["rows"]:
        assert len(row["sites"]) == 2
        assert row["sites"][0]["position"] < row["sites"][1]["position"] == len(row["token_ids"]) - 1
        assert {site["direction"] for site in row["sites"]} == {"singular_to_plural"}


def test_lexicon_and_templates_are_disjoint_from_v1():
    current = json.loads(OUT.read_text())
    prior = json.loads(V1.read_text())
    assert set(current["templates"]).isdisjoint(prior["templates"])
    current_subjects = {site["subject"] for row in current["rows"] for site in row["sites"]}
    prior_subjects = {site["subject"] for row in prior["rows"] for site in row["sites"]}
    assert current_subjects.isdisjoint(prior_subjects)


def test_builder_refuses_overwrite():
    completed = subprocess.run([sys.executable, str(BUILDER)], text=True, capture_output=True)
    assert completed.returncode != 0 and "FileExistsError" in completed.stderr
