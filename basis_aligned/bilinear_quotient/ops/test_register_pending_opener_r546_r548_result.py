import ast
from pathlib import Path


SCRIPT = Path(__file__).with_name("register_pending_opener_r546_r548_result.py")
TEXT = SCRIPT.read_text()


def test_registration_parses_and_requires_independent_audit():
    ast.parse(TEXT)
    assert 'audit["independent_summary_recomputation_exact"] is True' in TEXT
    assert 'audit["complete_row_identity_and_cell_counts"] is True' in TEXT
    assert 'result["model_forwards"] == 204' in TEXT


def test_registration_keeps_site_and_identification_distinct():
    assert '"status": "site_live"' in TEXT
    assert '"test_type": "capability"' in TEXT
    assert '"test_type": "full_swap_ceiling"' in TEXT
    assert "This does not yet identify a selective subspace" in TEXT
    assert "FINAL_TEST/OOD remain unopened" in TEXT
