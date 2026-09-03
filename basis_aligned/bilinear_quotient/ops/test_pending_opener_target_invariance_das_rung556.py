import ast
from pathlib import Path


SCRIPT = Path(__file__).with_name("pending_opener_target_invariance_das_rung556.py")
TEXT = SCRIPT.read_text()


def test_script_parses_and_binds_fresh_authority():
    ast.parse(TEXT)
    assert "07b64d2e48a6ca67685c81d3475a064daba612d6fe7ff233efd5b6c157b940a9" in TEXT
    assert "209b9bfcc20bff13bb37d822137003d6878506e66b0d9321ba0a0f7e9d8f2c5c" in TEXT
    assert "706a4fda4788dace89de6a4ae5f41cf3bd7e56ff194d8956d8e977b7ced1dc44" in TEXT


def test_controls_are_inside_training_not_only_scoring():
    assert "fit_controls" in TEXT
    assert "target_batch + control_batch" in TEXT
    assert "control_loss.mean()" in TEXT
    assert "full_vocabulary_logit_rms_values" in TEXT


def test_budget_splits_and_null_are_static():
    assert 'SPLITS = ("FIT", "SELECT")' in TEXT
    assert "EXPECTED_GRAD_SUFFIX_EVALS = len(RANKS) * len(SEEDS) * STEPS" in TEXT
    assert "EXPECTED_SCORE_SUFFIX_EVALS = 675" in TEXT
    assert '"forbidden_splits_opened": []' in TEXT
    assert "record_linear_site_null_and_test_nonlinear_or_earlier_representation" in TEXT
