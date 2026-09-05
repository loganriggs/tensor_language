#!/usr/bin/env python3
"""Pin screen_preflight against a known-bad and a known-good ledger state.

The tool's dangerous failure is a false CLEAR: it would waste exactly the screen it exists
to save. So both directions are exercised against ledgers built from the live runner's own
computed key -- no fixtures to drift.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_preflight as preflight

RUNNER = "run_circuit_fast_screen_task14_select_cross_noun"


def _key():
    key, complete, _, _ = preflight.check(RUNNER, Path("/nonexistent-ledger.jsonl"))
    assert complete, "this runner must yield the full 7-field key"
    return key


def test_empty_ledger_is_clear():
    """Known-good: nothing recorded, so nothing to collide with."""
    _, _, hits, _ = preflight.check(RUNNER, Path("/nonexistent-ledger.jsonl"))
    assert hits == []


def test_own_key_in_ledger_is_duplicate(tmp_path):
    """Known-bad: the runner's exact key already recorded -- must be refused."""
    ledger = tmp_path / "ledger.jsonl"
    entry = dict(_key(), request_id="prior-run-v1")
    ledger.write_text(json.dumps(entry) + "\n")
    _, complete, hits, _ = preflight.check(RUNNER, ledger)
    assert complete and len(hits) == 1 and hits[0][1]["request_id"] == "prior-run-v1"


def test_one_varied_field_clears_it(tmp_path):
    """Varying a single key field is what unblocks a refused run -- as the 05:02 fix did."""
    ledger = tmp_path / "ledger.jsonl"
    entry = dict(_key(), request_id="prior-run-v1")
    entry["prior_art_sha256"] = "0" * 64
    ledger.write_text(json.dumps(entry) + "\n")
    _, _, hits, _ = preflight.check(RUNNER, ledger)
    assert hits == []


def test_partial_check_never_claims_completeness(tmp_path):
    """A candidate without compile_plan must report a partial key, not a confident one."""
    key, complete = preflight.execution_key(
        __import__("run_circuit_fast_screen_p_vocab_match"))
    assert not complete
    assert set(preflight.PARTIAL_FIELDS) <= set(key)
