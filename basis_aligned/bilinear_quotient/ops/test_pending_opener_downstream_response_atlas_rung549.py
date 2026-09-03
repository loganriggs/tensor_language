import ast
from pathlib import Path


SCRIPT = Path(__file__).with_name("pending_opener_downstream_response_atlas_rung549.py")
TEXT = SCRIPT.read_text()


def test_script_parses_and_binds_audited_authority():
    ast.parse(TEXT)
    assert "209b9bfcc20bff13bb37d822137003d6878506e66b0d9321ba0a0f7e9d8f2c5c" in TEXT
    assert "25acb35355f457163c1ed1183aeb55aea0c08a224992d688250ba5e272564875" in TEXT
    assert 'SPLITS = ("FIT", "SELECT")' in TEXT


def test_exact_candidate_order_and_budget_are_static():
    assert 'CANDIDATES = ("mlp13_write",)' in TEXT
    assert "for layer in range(14, 18)" in TEXT
    assert "EXPECTED_FORWARDS = math.ceil(EXPECTED_ROWS / BATCH) * 3" in TEXT
    assert '"model_backwards": 0' in TEXT
    assert '"forbidden_splits_opened": []' in TEXT


def test_selection_uses_fit_and_select_only_validates():
    tree = ast.parse(TEXT)
    source = ast.unparse(tree)
    assert 'metrics[site]["fit"]["eligible"]' in TEXT
    assert 'metrics[site]["fit"]["selection_score"]' in TEXT
    assert 'report = metrics[selected]["select"]' in TEXT
    assert "readout_alignment_diagnostic" in TEXT
    assert '"used_for_selection": False' in TEXT
    assert "rank" not in source.split("def score_candidate", 1)[1].split("def main", 1)[0]
