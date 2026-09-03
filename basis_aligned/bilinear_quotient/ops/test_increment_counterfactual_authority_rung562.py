import collections
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "increment_counterfactual_authority_rung562.json"
RECEIPT = ROOT / "increment_counterfactual_authority_rung562_receipt.json"
FAMILIES = {
    "digit_coherent_shift", "word_coherent_shift", "cross_format_coherent_shift",
    "incoherent_middle_number_edit", "operation_preserved_surface_edit",
    "repeated_number_numeric_control", "step_two_numeric_control",
}


def docs():
    return json.loads(ROWS.read_text()), json.loads(RECEIPT.read_text())


def test_authority_is_complete_group_disjoint_and_outcome_free():
    rows, receipt = docs()
    assert rows["row_count"] == receipt["unique_prompt_pair_count"] == 1120
    assert rows["group_count"] == 160
    assert rows["model_loaded"] is False and rows["outcomes_opened"] == []
    assert receipt["model_loaded"] is False and receipt["outcomes_opened"] == []
    by_group, splits = collections.defaultdict(set), collections.defaultdict(set)
    for row in rows["rows"]:
        by_group[row["group_id"]].add(row["family_id"])
        splits[row["group_id"]].add(row["split"])
        assert all(row["construction_checks"].values())
    assert all(families == FAMILIES for families in by_group.values())
    assert all(len(value) == 1 for value in splits.values())


def test_target_families_change_answers_and_controls_do_not():
    rows, _ = docs()
    changing = {"digit_coherent_shift", "word_coherent_shift", "cross_format_coherent_shift"}
    for row in rows["rows"]:
        assert row["answer_changes"] == (row["family_id"] in changing)
        if row["family_id"] in changing:
            assert row["base_operation"] == row["donor_operation"] == "+1"
        if row["family_id"] == "incoherent_middle_number_edit":
            assert row["base_state"][0] == row["donor_state"][0]
            assert row["base_state"][2] == row["donor_state"][2]


def test_held_out_pools_are_explicitly_disjoint():
    rows, _ = docs()
    starts = collections.defaultdict(set)
    words = collections.defaultdict(set)
    leads = collections.defaultdict(set)
    for row in rows["rows"]:
        if row["family_id"] == "digit_coherent_shift":
            starts[row["split"]].update((row["base_state"][0], row["donor_state"][0]))
            words[row["split"]].update(row["content_words"])
            leads[row["split"]].add(row["prompt_lead"])
    assert starts == {"FIT": {1, 2}, "SELECT": {6, 7}, "FINAL_TEST": {11, 12}, "OOD": {16, 17}}
    split_names = list(words)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            assert not (words[left] & words[right])
            assert not (leads[left] & leads[right])
