import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "increment_native_capability_rung565_audit.json"


def test_r565_recomputes_the_complete_opened_result_without_model_calls():
    audit = json.loads(AUDIT.read_text())
    assert audit["row_statistics_recomputed"] == 896
    assert audit["endpoint_cells_recomputed"] == 12
    assert audit["necessity_pairs_recomputed"] == 64
    assert audit["input_hashes_match"] and audit["call_and_split_ledger_match"]
    assert audit["terminal_null_reproduced"]
    assert audit["model_loaded"] is False and audit["model_forwards"] == audit["model_backwards"] == 0
