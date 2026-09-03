import collections
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops" / "pending_opener_unique_rows_rung543.py"
spec = importlib.util.spec_from_file_location("r543", SCRIPT)
r543 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(r543)


def docs():
    return json.loads(r543.OUT.read_text()), json.loads(r543.RECEIPT.read_text())


def test_row_and_group_counts_are_exact():
    rows, receipt = docs()
    assert rows["row_count"] == 1200
    assert rows["group_count"] == 240
    assert receipt["unique_prompt_pair_count"] == 1200
    assert receipt["unique_token_sequence_count"] == 2400


def test_content_addressed_groups_bind_all_families_and_one_split():
    rows, _ = docs()
    families = collections.defaultdict(set)
    splits = collections.defaultdict(set)
    for row in rows["rows"]:
        families[row["group_id"]].add(row["family_id"])
        splits[row["group_id"]].add(row["split"])
    assert all(value == set(r543.FAMILIES) for value in families.values())
    assert all(len(value) == 1 for value in splits.values())


def test_roles_answers_and_single_token_edits_match_contract():
    rows, _ = docs()
    for row in rows["rows"]:
        assert row["answer_changes"] == (row["role"] == "interchange")
        assert len(r543.ENC.encode(row["base_answer"])) == 1
        assert len(r543.ENC.encode(row["donor_answer"])) == 1
        if row["family_id"] in {"direct_type_substitution", "nonopener_punctuation_substitution"}:
            assert row["construction_checks"]["equal_token_length"] is True
            assert row["construction_checks"]["single_token_difference"] is True


def test_receipt_hashes_exact_bytes_and_no_outcomes_exist():
    rows, receipt = docs()
    assert hashlib.sha256(r543.OUT.read_bytes()).hexdigest() == receipt["rows_sha256"]
    assert rows["model_loaded"] is False
    assert rows["model_forwards"] == rows["model_backwards"] == 0
    assert rows["outcomes_opened"] == []
