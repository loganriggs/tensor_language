"""Regression versions of the planted attacks that blocked R590 at 5fc3144eb.

These tests are CPU-only.  In particular, the outcome-read attack interposes on
both broad authority loaders and the forbidden path reads.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


OPS = Path(__file__).resolve().parent
ADAPTER_PATH = OPS / "execute_numbered_list_cached_value_downstream_use_rung590.py"

EXPECTED = {
    "producer": "c38654506f36fcf111f3a34f356893240548c3cfbf4eded58efb04d31fdb2e36",
    "owner_test": "49f6f7a998bfb69331c36391f5d3c16d9b702c1fd60b4da5b09c920f3832e5b0",
    "dryrun": "3ebada19f74906ba3e7cd1637fc1cd6cdff84936124dee01cb058875432d3b95",
    "adapter": "c525cad078935ef0552214fba13c16a5d56483c8e3048bbec4d6ab9ef3f17885",
    "adapter_test": "17d51c8e7df667ecf1cc146b1ac00e34f658e97759ee149ddb254f7d9317f07e",
    "note": "dae72b4aee35030f31ce42674d9535d6bff6c857b9beb8633a8ac809edaf031b",
    "handoff_v6": "d1fdedd90ffff29e6790042b9c9a6ad84278849c3f66707cb586317832fdad1c",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_adapter():
    name = "r590_immutable_closure_review_adapter"
    spec = importlib.util.spec_from_file_location(name, ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exact_candidate():
    adapter = load_adapter()
    snapshot = adapter.capture_frozen_bytes()
    loaded_names = [name for name, _, _ in adapter.EXECUTABLE_LOAD_ORDER]
    loaded_names.append("r590_managed_producer")
    previous = {name: sys.modules.get(name) for name in loaded_names}
    producer = adapter.load_frozen_producer(snapshot)
    try:
        yield adapter, snapshot, producer
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def test_exact_reviewed_packet_and_v6_binding(exact_candidate):
    adapter, snapshot, _ = exact_candidate
    assert digest(adapter.PRODUCER) == EXPECTED["producer"]
    assert digest(adapter.OWNER_TEST) == EXPECTED["owner_test"]
    assert digest(adapter.DRYRUN) == EXPECTED["dryrun"]
    assert digest(ADAPTER_PATH) == EXPECTED["adapter"]
    assert digest(adapter.ADAPTER_TEST) == EXPECTED["adapter_test"]
    assert digest(adapter.NOTE) == EXPECTED["note"]
    assert digest(adapter.HANDOFF_V6) == EXPECTED["handoff_v6"]
    assert adapter.FROZEN_HASHES[adapter.HANDOFF_V6] == EXPECTED["handoff_v6"]
    assert hashlib.sha256(snapshot[adapter.PRODUCER]).hexdigest() == EXPECTED["producer"]


def test_verified_import_creates_no_r590_outcome_namespace(exact_candidate):
    adapter, _, _ = exact_candidate
    assert all(not path.exists() for path in adapter.OUTCOME_NAMESPACES)


def test_recursive_dependency_uses_captured_bytes_after_path_swap(
    exact_candidate, monkeypatch, tmp_path,
):
    _, _, producer = exact_candidate
    auditor = producer.r588
    helper = tmp_path / "helper.py"

    helper.write_bytes(b"raise RuntimeError('mutable helper path executed')\n")
    captured = helper.read_bytes()
    monkeypatch.setattr(auditor, "R582_HELPER", helper)
    monkeypatch.setattr(auditor, "verify_preoutcome_authority", lambda: {})

    # This is the race v6 forbids: verification/capture has completed, then the
    # executable pathname changes before a recursive dependency opens it.
    helper.write_bytes(b"raise RuntimeError('swapped helper path executed')\n")
    loaded = auditor.load_r582_helper()
    assert hashlib.sha256(captured).hexdigest() != digest(helper)
    assert loaded is producer.r584.r582


def test_dryrun_transitive_closure_never_opens_prior_outcomes(
    exact_candidate, monkeypatch,
):
    _, _, producer = exact_candidate
    auditor = producer.r588
    forbidden_paths = {
        producer.r584.r582.R576_RESULT.resolve(),
        producer.r584.r582.R579_AUDIT.resolve(),
    }
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def reject_outcome_authority(*_args, **_kwargs):
        raise AssertionError("model-free dry run reached prior-outcome authority code")

    def guarded_read_bytes(path):
        if path.resolve() in forbidden_paths:
            raise AssertionError(f"model-free dry run read prior outcome bytes: {path}")
        return original_read_bytes(path)

    def guarded_read_text(path, *args, **kwargs):
        if path.resolve() in forbidden_paths:
            raise AssertionError(f"model-free dry run read prior outcome text: {path}")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(auditor, "verify_preoutcome_authority", reject_outcome_authority)
    monkeypatch.setattr(auditor, "load_authority", reject_outcome_authority)
    monkeypatch.setattr(producer.r584, "load_authority", reject_outcome_authority)
    monkeypatch.setattr(producer.r584.r582, "validate_authorities", reject_outcome_authority)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    producer.run_dryrun()


def test_provenance_names_the_executed_captured_producer(
    exact_candidate, monkeypatch, tmp_path,
):
    adapter, snapshot, producer = exact_candidate
    script = tmp_path / "producer.py"
    owner_test = tmp_path / "owner_test.py"
    script.write_bytes(snapshot[adapter.PRODUCER])
    owner_test.write_bytes(snapshot[adapter.OWNER_TEST])
    executed_digest = hashlib.sha256(snapshot[adapter.PRODUCER]).hexdigest()

    monkeypatch.setattr(producer, "SCRIPT", script)
    monkeypatch.setattr(producer, "TEST", owner_test)
    monkeypatch.setattr(producer, "validate_authorities", lambda: {})

    # The in-memory producer has already been compiled from snapshot bytes.
    # Changing the pathname must not change the implementation digest that a
    # future result or receipt claims to have executed.
    script.write_bytes(b"# different bytes at the old pathname\n")
    provenance = producer.source_hashes()
    assert provenance[str(adapter.PRODUCER)] == executed_digest
    assert str(script) not in provenance
    assert producer._role_sha256("implementation") == executed_digest
    assert producer.wrapper_science_callsite_census() == {
        "run_science.facade_load_bilin18": 1,
        "run_science.capture_split": 2,
        "run_science.evaluate_component": 4,
    }
