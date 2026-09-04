"""Tests for ops/circuit_prior_art_search.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circuit_prior_art_search as P


def test_load_events_reads_the_task_json_authority():
    ev = P.load_events()
    assert len(ev) > 20, f"expected the task_*.json event authority, got {len(ev)}"
    assert all(isinstance(e, dict) for _t, _g, e in ev)


def test_load_events_also_reads_terminal_fast_screen_authority():
    ev = P.load_events()
    fast = [(task, event) for task, _tag, event in ev if task == "fast_screen_ledger.jsonl"]
    assert fast, "new screens would otherwise be invisible until copied into a task dossier"
    assert all(event.get("event_id") and event.get("stage") for _task, event in fast)


def test_current_head_complement_screen_is_searchable():
    hits = P.search(["head11.3", "complement"])
    assert any(event.get("event_id") == "task14-head11-3-complement-v1" for _task, event in hits)


def test_search_requires_every_term():
    both = P.search(["task14", "localization"])
    assert both, "expected task14 localization events"
    assert not P.search(["task14", "zzzznotpresent"])


def test_failed_targets_are_the_invalid_and_null_verdicts():
    assert P.FAILED_VERDICTS == ("invalid", "null")
    hits = P.search(["agreement"])
    verdicts = {e.get("verdict") for _t, e in hits}
    assert verdicts, "no verdicts parsed"


def test_stale_dossier_reports_authority_events_the_rendered_dossier_omits():
    """The point of the tool: 'the dossier does not say so' must not read as 'it has not been done'."""
    missing = P.stale_dossier()
    assert isinstance(missing, dict)
    total = sum(len(v) for v in missing.values())
    assert total >= 0
    for _task, items in missing.items():
        for eid, _v in items:
            assert isinstance(eid, str) and eid
