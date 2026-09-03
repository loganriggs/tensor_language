import ast
from pathlib import Path


SCRIPT = Path(__file__).with_name("pending_opener_downstream_response_atlas_rung550_audit.py")
TEXT = SCRIPT.read_text()


def test_audit_parses_and_never_loads_model_or_cuda():
    ast.parse(TEXT)
    assert "facade" not in TEXT
    assert 'map_location="cpu"' in TEXT
    assert 'device="cuda"' not in TEXT


def test_audit_rebuilds_fit_selection_and_select_verdict():
    assert "verify_rows" in TEXT and "recompute" in TEXT
    assert 'rebuilt[site]["fit"]["selection_score"]' in TEXT
    assert 'report = rebuilt[selected]["select"]' in TEXT
    assert '"selection_depends_only_on_fit": True' in TEXT
    assert 'result["forbidden_splits_opened"] != []' in TEXT


def test_readout_alignment_cannot_enter_selection():
    assert 'saved["readout_alignment_diagnostic"]["used_for_selection"] is not False' in TEXT
    selection_block = TEXT.split("eligible = [site", 1)[1].split("selected_pass =", 1)[0]
    assert "readout" not in selection_block
