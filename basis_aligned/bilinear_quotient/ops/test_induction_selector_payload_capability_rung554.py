import ast
from pathlib import Path


SCRIPT = Path(__file__).with_name("induction_selector_payload_capability_rung554.py")
TEXT = SCRIPT.read_text()


def test_script_parses_and_binds_frozen_authority():
    ast.parse(TEXT)
    assert "6a0a6d2c8a3891ae5d6f787527b35e71c17518548b3b1836042afe730b13c460" in TEXT
    assert "9fc0376fade6fb204686e164f293f8991caf7bc45c67eedd064f330dffd5d1ea" in TEXT
    assert "9de3b16299043b6cf96e0cf2c75eb686f2063082e34a51e594fee1b0c0c4f777" in TEXT


def test_budget_and_split_contract_are_static():
    assert 'SPLITS = ("FIT", "SELECT")' in TEXT
    assert "EXPECTED_SEQUENCES = 864" in TEXT
    assert "EXPECTED_FORWARDS = math.ceil(EXPECTED_SEQUENCES / BATCH)" in TEXT
    assert '"model_backwards": 0' in TEXT
    assert '"forbidden_splits_opened": []' in TEXT


def test_scoring_has_all_preregistered_arms():
    assert "pred_a_four_cell_capability" in TEXT
    assert "pred_b_relation_preserving_controls" in TEXT
    assert "pred_c_selected_match_necessity_and_selectivity" in TEXT
    assert "bootstrap95_lower_mean_selective_gap" in TEXT
    assert '"irrelevant_source_edit"' in TEXT
    assert '"copy_relation_preserved_nuisance_change"' in TEXT
