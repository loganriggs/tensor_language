import collections
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "pending_opener_unique_rows_rung543_v2.json"
RECEIPT = ROOT / "pending_opener_unique_rows_rung543_v2_receipt.json"


def docs():
    return json.loads(ROWS.read_text()), json.loads(RECEIPT.read_text())


def test_all_rows_and_sequences_remain_globally_unique():
    rows, receipt = docs()
    assert rows["row_count"] == receipt["unique_prompt_pair_count"] == 1200
    assert receipt["unique_token_sequence_count"] == 2400


def test_every_ordered_answer_pair_is_balanced_in_every_family_and_split():
    rows, _ = docs()
    counts = collections.Counter(
        (row["split"], row["family_id"], row["base_answer"], row["donor_answer"])
        for row in rows["rows"] if row["role"] == "interchange"
    )
    for split, expected in {"FIT": 8, "SELECT": 4, "FINAL_TEST": 4, "OOD": 4}.items():
        for family in ("direct_type_substitution", "completed_then_reopened_order"):
            cell = {(base_answer, donor_answer): count
                    for (s, f, base_answer, donor_answer), count in counts.items()
                    if s == split and f == family}
            assert len(cell) == 12
            assert set(cell.values()) == {expected}


def test_no_model_outcome_was_opened():
    rows, receipt = docs()
    assert rows["model_loaded"] is False
    assert rows["outcomes_opened"] == receipt["outcomes_opened"] == []
