import collections
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "increment_two_hypothesis_rows_rung567.json"
RECEIPT = ROOT / "increment_two_hypothesis_rows_rung567_receipt.json"
LIST = {"list_two_line_state_shift", "list_three_line_state_shift", "list_surface_preserved",
        "list_middle_index_break", "list_repeated_index_control", "list_step_two_conflict"}
SEQUENCE = {"sequence_digit_state_shift", "sequence_word_state_shift", "sequence_cross_format_shift",
            "sequence_digit_surface_preserved", "sequence_word_surface_preserved", "sequence_middle_value_break",
            "sequence_digit_copy_control", "sequence_word_copy_control", "sequence_step_two_conflict"}


def docs():
    return json.loads(ROWS.read_text()), json.loads(RECEIPT.read_text())


def test_fresh_authority_is_complete_and_outcome_free():
    rows, receipt = docs()
    assert rows["row_count"] == receipt["unique_prompt_pair_count"] == 1200
    assert rows["group_count"] == 160 and receipt["development_sequence_overlap"] == 0
    assert rows["model_loaded"] is False and rows["outcomes_opened"] == []
    by_group, hypotheses, splits = collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set)
    for item in rows["rows"]:
        by_group[item["group_id"]].add(item["family_id"])
        hypotheses[item["group_id"]].add(item["hypothesis_id"])
        splits[item["group_id"]].add(item["split"])
        assert all(item["construction_checks"].values())
    assert all(len(value) == 1 for value in hypotheses.values())
    assert all(len(value) == 1 for value in splits.values())
    for group_id, families in by_group.items():
        expected = LIST if next(iter(hypotheses[group_id])) == "numbered_list_index_successor" else SEQUENCE
        assert families == expected


def test_conflicts_distinguish_structural_successor_from_arithmetic_step_two():
    rows, _ = docs()
    for item in rows["rows"]:
        details = item["semantic_details"]
        if item["family_id"] == "list_step_two_conflict":
            assert details["structural_successor_answer"] + 1 == details["arithmetic_step_two_answer"]
            assert item["base_answer"] == str(details["structural_successor_answer"])
        if item["family_id"] == "sequence_step_two_conflict":
            assert details["last_value_successor_answer"] + 1 == details["arithmetic_step_two_answer"]
            assert item["base_answer_id"] is None and item["donor_answer_id"] is None


def test_splits_hold_out_starts_and_lexical_pools():
    rows, _ = docs()
    group_words, group_starts = collections.defaultdict(set), collections.defaultdict(set)
    for item in rows["rows"]:
        if item["family_id"] == "sequence_digit_state_shift":
            values = item["semantic_details"]
            group_starts[item["split"]].update((values["base_values"][0], values["donor_values"][0]))
        group_words[item["split"]].update(
            token.lower().strip(".,") for token in item["base_text"].split() if token.lower().strip(".,").isalpha()
        )
    assert group_starts == {"FIT": {8, 9}, "SELECT": {11, 12}, "FINAL_TEST": {14, 15}, "OOD": {16, 17}}
    # The builder's declared content pools are checked via distinctive registered words, not shared grammar.
    markers = {"FIT": "acorn", "SELECT": "alcove", "FINAL_TEST": "archway", "OOD": "activation"}
    for split, marker in markers.items():
        assert marker in group_words[split]
        assert all(marker not in group_words[other] for other in markers if other != split)
