import ast
import json
from pathlib import Path


OPS = Path(__file__).resolve().parent
SCRIPT = OPS / "induction_selector_payload_factorial_rows_rung552.py"
ROOT = OPS.parent
ROWS = ROOT / "induction_selector_payload_factorial_rows_rung552.json"
RECEIPT = ROOT / "induction_selector_payload_factorial_rows_rung552_receipt.json"


def test_builder_parses_and_never_imports_model():
    tree = ast.parse(SCRIPT.read_text())
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert "torch" not in imports
    assert "transformers" not in imports
    assert "facade" not in SCRIPT.read_text()


def test_materialized_factorial_and_split_contract():
    document = json.loads(ROWS.read_text())
    receipt = json.loads(RECEIPT.read_text())
    assert document["group_count"] == 180
    assert document["row_count"] == 1800
    assert receipt["factorial_condition_count"] == 720
    assert receipt["split_group_counts"] == {"FIT": 72, "SELECT": 36, "FINAL_TEST": 36, "OOD": 36}
    assert all(receipt[key] is True for key in (
        "every_group_has_complete_factorial_and_controls",
        "every_group_belongs_to_one_split",
        "prompt_sequences_never_cross_groups",
        "variable_token_banks_disjoint_across_splits",
        "within_group_condition_reuse_declared",
    ))
    assert receipt["model_loaded"] is False and receipt["outcomes_opened"] == []


def test_each_group_has_four_conditions_and_ten_derived_rows():
    document = json.loads(ROWS.read_text())
    groups = {group["group_id"]: group for group in document["groups"]}
    counts = {group_id: 0 for group_id in groups}
    for row in document["rows"]:
        counts[row["group_id"]] += 1
        assert row["split"] == groups[row["group_id"]]["split"]
    assert all(set(group["factorial_conditions"]) == {"s0p0", "s1p0", "s0p1", "s1p1"}
               for group in groups.values())
    assert set(counts.values()) == {10}


def test_factorial_answer_rule_and_single_factor_edges():
    document = json.loads(ROWS.read_text())
    for group in document["groups"]:
        cells = group["factorial_conditions"]
        assert cells["s0p0"]["answer_id"] == cells["s1p1"]["answer_id"]
        assert cells["s1p0"]["answer_id"] == cells["s0p1"]["answer_id"]
        assert cells["s0p0"]["answer_id"] != cells["s1p0"]["answer_id"]
    for row in document["rows"]:
        if row["family_id"] in {"two_valid_sources_selector_swap", "payload_swap_match_preserved"}:
            assert row["answer_changes"] is True
        else:
            assert row["answer_changes"] is False
