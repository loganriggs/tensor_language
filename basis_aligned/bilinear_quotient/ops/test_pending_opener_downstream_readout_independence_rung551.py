import ast
from pathlib import Path


SCRIPT = Path(__file__).with_name("pending_opener_downstream_readout_independence_rung551.py")
TEXT = SCRIPT.read_text()


def test_script_parses_and_has_no_model_forward():
    ast.parse(TEXT)
    assert "load_bilin18" not in TEXT
    assert '"model_forwards": 0' in TEXT
    assert 'map_location="cpu"' in TEXT


def test_guard_uses_full_readout_span_not_only_pairwise_cosine():
    assert "torch.linalg.svd(readouts" in TEXT
    assert "(templates @ basis).norm" in TEXT
    assert "READOUT_SPAN_FRACTION_MAX = 0.50" in TEXT
    assert 'result["pred_c_selected_candidate_validates"]' in TEXT


def test_guard_cannot_open_outcome_splits_or_change_selection():
    assert '"evaluated_splits": ["FIT"]' in TEXT
    assert '"forbidden_splits_opened": []' in TEXT
    assert "selected = result[\"selected_candidate\"]" in TEXT
    assert "fit_eligible_candidates" not in TEXT
