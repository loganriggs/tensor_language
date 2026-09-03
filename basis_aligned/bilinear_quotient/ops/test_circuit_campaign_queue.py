import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "make_circuit_campaign_queue.py"
SPEC = importlib.util.spec_from_file_location("circuit_campaign_queue", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_every_v2_record_and_active_event_is_represented_once():
    payload = MODULE.build()
    registry = json.loads(MODULE.REGISTRY.read_text())["circuits"]
    expected_tags = {tag for tag, row in registry.items() if row.get("schema_version") == 2}
    assert {item["tag"] for item in payload["work_items"]} == expected_tags
    queue_events = [event for item in payload["work_items"] for event in item["active_event_ids"]]
    assert len(queue_events) == len(set(queue_events))


def test_legacy_candidates_are_not_counted_as_canonical_work_items():
    payload = MODULE.build()
    assert payload["summary"]["legacy_candidates"] == len(payload["legacy_candidates"])
    assert all(item["action"] == "candidate_only_do_not_count_as_counterfactual_circuit"
               for item in payload["legacy_candidates"])
    assert not ({item["tag"] for item in payload["work_items"]}
                & {item["tag"] for item in payload["legacy_candidates"]})


def test_generated_files_match_builder():
    MODULE.main()
    written = json.loads(MODULE.OUT_JSON.read_text())
    assert written == MODULE.build()
    assert "anti-duplication work view" in MODULE.OUT_MD.read_text()
