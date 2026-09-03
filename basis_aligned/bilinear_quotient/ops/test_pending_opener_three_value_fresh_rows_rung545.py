import collections
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "pending_opener_three_value_fresh_rows_rung545.json"
RECEIPT = ROOT / "pending_opener_three_value_fresh_rows_rung545_receipt.json"


def docs():
    return json.loads(ROWS.read_text()), json.loads(RECEIPT.read_text())


def test_rows_are_unique_complete_and_outcome_free():
    rows, receipt = docs()
    assert rows["row_count"] == receipt["unique_prompt_pair_count"] == 900
    assert receipt["unique_token_sequence_count"] == 1800
    assert rows["group_count"] == 180
    assert rows["model_loaded"] is False and rows["outcomes_opened"] == []
    assert receipt["model_loaded"] is False and receipt["outcomes_opened"] == []
    by_group = collections.defaultdict(set)
    for row in rows["rows"]:
        by_group[row["group_id"]].add(row["family_id"])
    assert all(len(families) == 5 for families in by_group.values())


def test_all_six_ordered_pairs_are_balanced_in_every_target_family_and_split():
    rows, _ = docs()
    counts = collections.Counter((row["split"], row["family_id"], row["base_answer"], row["donor_answer"])
                                 for row in rows["rows"] if row["role"] == "interchange")
    for split, expected in {"FIT": 12, "SELECT": 6, "FINAL_TEST": 6, "OOD": 6}.items():
        for family in ("direct_three_value_type_substitution", "completed_then_reopened_three_value_order"):
            cells = {(base, donor): count for (s, f, base, donor), count in counts.items()
                     if s == split and f == family}
            assert len(cells) == 6 and set(cells.values()) == {expected}


def test_no_r543_prefix_or_content_word_is_reused():
    rows, _ = docs()
    old = json.loads((ROOT / "pending_opener_unique_rows_rung543_v2.json").read_text())
    old_text = "\n".join(row["base_text"] + "\n" + row["donor_text"] for row in old["rows"])
    new_text = "\n".join(row["base_text"] + "\n" + row["donor_text"] for row in rows["rows"])
    old_markers = {"The editor", "A teacher", "The curator", "The architect", "During the audit the analyst"}
    new_markers = {"The baker", "A doctor", "The judge", "In theorem C the reviewer"}
    assert all(marker in old_text and marker not in new_text for marker in old_markers)
    assert all(marker in new_text and marker not in old_text for marker in new_markers)
