import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "make_circuit_experiment_index.py"
SPEC = importlib.util.spec_from_file_location("circuit_experiment_index", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_current_registry_has_no_exact_execution_duplicates():
    report = MODULE.audit(MODULE.load_events())
    assert report["duplicate_execution_groups"] == []


def test_landed_r546_is_not_open_and_r549_is_open():
    report = MODULE.audit(MODULE.load_events())
    open_ids = {row["event_id"] for row in report["open_preregistrations"]}
    assert "pending_opener_three_value_confirmation.r546.preregistered.v1" not in open_ids
    assert "pending_opener_downstream_response_atlas.r549.preregistered.v1" in open_ids


def test_generated_json_never_invents_events():
    MODULE.main()
    payload = json.loads(MODULE.OUT_JSON.read_text())
    assert payload["event_count"] == len(payload["events"])
    assert payload["execution_count"] == len({row["execution_key"] for row in payload["events"]})
    assert all(len(row["protocol_key"]) == 64 for row in payload["events"])
