import collections
import hashlib
import importlib.util
import json
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
BUILDER = Path(__file__).with_name("pending_opener_multifamily_rows_rung537.py")
ROWS = BQ / "pending_opener_multifamily_rows_rung537.json"
RECEIPT = BQ / "pending_opener_multifamily_rows_rung537_receipt.json"


def load_builder():
    spec = importlib.util.spec_from_file_location("pending_rows", BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_receipt_binds_frozen_rows_and_no_outcomes():
    rows = json.loads(ROWS.read_text())
    receipt = json.loads(RECEIPT.read_text())
    assert hashlib.sha256(ROWS.read_bytes()).hexdigest() == receipt["rows_sha256"]
    assert rows["status"] == "rows_frozen_outcomes_unopened"
    assert receipt["outcomes_opened"] is False
    assert rows["model_forwards"] == receipt["model_forwards"] == 0
    assert rows["model_backwards"] == receipt["model_backwards"] == 0


def test_group_split_is_shared_across_all_three_families():
    rows = json.loads(ROWS.read_text())["rows"]
    assert len(rows) == 288
    by_group = collections.defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    assert len(by_group) == 96
    for group in by_group.values():
        assert len(group) == 3
        assert len({row["split"] for row in group}) == 1
        assert {row["family_id"] for row in group} == {
            "opener_type_substitution",
            "closed_then_reopened_type",
            "pending_state_preserved_surface_edit",
        }


def test_two_interchanges_are_structurally_different_and_endpoint_correct_by_construction():
    rows = json.loads(ROWS.read_text())["rows"]
    direct = [row for row in rows if row["family_id"] == "opener_type_substitution"]
    structural = [row for row in rows if row["family_id"] == "closed_then_reopened_type"]
    invariant = [row for row in rows if row["family_id"] == "pending_state_preserved_surface_edit"]
    assert len(direct) == len(structural) == len(invariant) == 96
    assert all(row["answer_changes"] and row["construction_checks"]["single_token_difference"] for row in direct)
    assert all(row["answer_changes"] and not row["construction_checks"]["single_token_difference"] for row in structural)
    assert all(row["construction_checks"]["same_lexical_token_multiset"] for row in direct + structural)
    assert all(not row["answer_changes"] and row["base_answer"] == row["donor_answer"] for row in invariant)


def test_lexical_pools_are_disjoint_between_splits():
    builder = load_builder()
    pools = {split: set(spec["words"]) for split, spec in builder.SPLITS.items()}
    for left, right in __import__("itertools").combinations(pools, 2):
        assert pools[left].isdisjoint(pools[right])


def test_builder_is_byte_deterministic():
    before = (ROWS.read_bytes(), RECEIPT.read_bytes())
    load_builder().main()
    after = (ROWS.read_bytes(), RECEIPT.read_bytes())
    assert before == after
