import ast
from pathlib import Path


SCRIPT = Path(__file__).with_name("pending_opener_three_value_confirmation_rung546.py")
TEXT = SCRIPT.read_text()


def test_script_parses_and_binds_fresh_authority():
    ast.parse(TEXT)
    assert "pending_opener_three_value_fresh_rows_rung545.json" in TEXT
    assert "07b64d2e48a6ca67685c81d3475a064daba612d6fe7ff233efd5b6c157b940a9" in TEXT
    assert 'SPLITS = ("FIT", "SELECT")' in TEXT


def test_budget_and_no_fit_contract_are_static():
    assert "EXPECTED_PAIRS = 540" in TEXT
    assert "EXPECTED_FORWARDS = math.ceil(EXPECTED_PAIRS / BATCH) * 3" in TEXT
    assert '"model_backwards": 0' in TEXT
    assert '"forbidden_splits_opened": []' in TEXT
    assert 'SITE = "attn13h8"' in TEXT
