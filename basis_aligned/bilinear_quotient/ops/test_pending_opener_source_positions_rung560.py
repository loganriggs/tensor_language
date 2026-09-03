import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("pending_opener_source_positions_rung560.py")
SPEC = importlib.util.spec_from_file_location("rung560_sources", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_last_semantic_opener_is_used():
    assert MODULE.source_position([10, 357, 20, 357, 30], 8) == 3
    assert MODULE.source_position([685, 10], 60) == 0
    assert MODULE.source_position([1, 366, 2], 1) == 1


def test_audit_is_model_free_and_covers_every_fit_select_row():
    result = subprocess.run(["python", str(SCRIPT)], check=True, capture_output=True, text=True)
    payload = json.loads(MODULE.OUT.read_text())
    assert payload["all_checks_pass"] is True
    assert payload["row_count"] == 540
    assert payload["unequal_length_distance_rows"] == 108
    assert payload["inconsistent_proposed_variable_endpoint_labels"] == 108
    assert payload["model_loaded"] is False and payload["model_forwards"] == 0
    assert payload["outcomes_opened"] == []
    assert "all_checks_pass" in result.stdout
