import ast
from pathlib import Path


SCRIPT = Path(__file__).with_name("pending_opener_three_value_confirmation_rung548_audit.py")
TEXT = SCRIPT.read_text()


def test_audit_parses_and_is_cpu_only():
    tree = ast.parse(TEXT)
    assert not any(isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
                   for node in ast.walk(tree))
    assert "pending_opener_three_value_confirmation_rung546_results.json" in TEXT
    assert "pending_opener_three_value_fresh_rows_rung545.json" in TEXT


def test_audit_recomputes_every_gate_and_binds_authority():
    assert "EXPECTED_FORWARDS = 204" in TEXT
    assert 'result["model_backwards"] != 0' in TEXT
    assert 'result["forbidden_splits_opened"] != []' in TEXT
    assert "audit_capability" in TEXT and "audit_site" in TEXT
    assert "complete_row_identity_and_cell_counts" in TEXT
    assert "independent_summary_recomputation_exact" in TEXT
