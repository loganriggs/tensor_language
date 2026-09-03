import collections
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "increment_counterfactual_authority_rung563.json"
RECEIPT = ROOT / "increment_counterfactual_authority_rung563_receipt.json"
FAMILIES = {
    "digit_coherent_shift", "word_coherent_shift", "cross_format_coherent_shift",
    "incoherent_middle_number_edit", "operation_preserved_surface_edit",
    "repeated_number_numeric_control", "step_two_numeric_control",
}


def docs():
    return json.loads(ROWS.read_text()), json.loads(RECEIPT.read_text())


def test_natural_authority_is_complete_and_outcome_free():
    rows, receipt = docs()
    assert rows["row_count"] == receipt["unique_prompt_pair_count"] == 1120
    assert rows["group_count"] == 160
    assert rows["family_revealing_prompt_labels"] is False
    assert receipt["family_revealing_prompt_labels"] is False
    assert rows["model_loaded"] is False and rows["outcomes_opened"] == []
    by_group, by_split = collections.defaultdict(set), collections.defaultdict(set)
    for row in rows["rows"]:
        by_group[row["group_id"]].add(row["family_id"])
        by_split[row["group_id"]].add(row["split"])
        assert "[" not in row["base_text"] and "[" not in row["donor_text"]
        assert all(row["construction_checks"].values())
    assert all(value == FAMILIES for value in by_group.values())
    assert all(len(value) == 1 for value in by_split.values())


def test_target_pairs_change_only_registered_state_or_representation():
    rows, _ = docs()
    for row in rows["rows"]:
        if row["family_id"] in {"digit_coherent_shift", "word_coherent_shift"}:
            assert row["base_operation"] == row["donor_operation"] == "+1"
            assert row["base_answer_id"] != row["donor_answer_id"]
        elif row["family_id"] == "cross_format_coherent_shift":
            assert row["base_operation"] == row["donor_operation"] == "+1"
            assert any(character.isdigit() for character in row["base_text"])
            assert not any(character.isdigit() for character in row["donor_text"])
        else:
            assert row["base_answer_id"] == row["donor_answer_id"]


def test_prompt_reuse_is_only_within_group():
    rows, receipt = docs()
    owners = collections.defaultdict(set)
    for row in rows["rows"]:
        owners[tuple(row["base_ids"])].add(row["group_id"])
        owners[tuple(row["donor_ids"])].add(row["group_id"])
    assert all(len(groups) == 1 for groups in owners.values())
    assert receipt["unique_token_sequence_count"] < receipt["token_sequence_count"]
