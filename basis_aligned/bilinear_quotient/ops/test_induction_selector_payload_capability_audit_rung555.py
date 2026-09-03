import ast
from pathlib import Path


SCRIPT = Path(__file__).with_name("induction_selector_payload_capability_audit_rung555.py")
TEXT = SCRIPT.read_text()


def test_audit_parses_without_importing_r554():
    ast.parse(TEXT)
    assert "import induction_selector_payload_capability_rung554" not in TEXT
    assert "terminal_decision_recomputed" in TEXT
    assert "does not independently recompute bootstrap samples" in TEXT


def test_audit_binds_budget_splits_and_all_bars():
    assert 'result["model_forwards"] == 27' in TEXT
    assert 'result["unique_sequences"] == 864' in TEXT
    assert 'SPLITS = ("FIT", "SELECT")' in TEXT
    assert 'cell["correct_fraction"] >= .75' in TEXT
    assert 'cell["selected_match_break_positive_fraction"] >= .70' in TEXT
    assert 'cell["bootstrap95_lower_mean_selective_gap"] > 0' in TEXT
