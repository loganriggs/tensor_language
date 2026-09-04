"""Independent planted attacks on exact R590 commit 5fc3144eb.

These tests are CPU-only.  In particular, the outcome-read attack interposes on
the first forbidden pathname before its bytes are opened.
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
    "producer": "5cc4544158312d7fa6224bf46c635acbb0d4a11fc2d620cedc2516d169f5966e",
    "owner_test": "49f6f7a998bfb69331c36391f5d3c16d9b702c1fd60b4da5b09c920f3832e5b0",
    "dryrun": "817f457ba1cc9737735182f495c54a3956be8c5dd6267bb5d8222f40e750d603",
    "adapter": "275f1c4d72f538283daba1b417be7e33e0c1749f0c1e21a2be1d0a6143f23f57",
    "adapter_test": "4c5bd25cdf06e21f823c9e09fdd57a7ca54d8700aa23a379a7913e2fc8c6b174",
    "note": "a6641a20a456d30895a9ba807c22ec74e7695fe5c84ce4300b909787c603afa7",
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


@pytest.mark.xfail(
    strict=True,
    reason="R588 reopens R582_HELPER by mutable pathname after snapshot loading",
)
def test_recursive_dependency_uses_captured_bytes_after_path_swap(
    exact_candidate, monkeypatch, tmp_path,
):
    _, _, producer = exact_candidate
    auditor = producer.r588
    helper = tmp_path / "helper.py"

    def helper_source(marker: str) -> bytes:
        return (
            f"SITES = {tuple(auditor.SITES)!r}\n"
            f"COMPONENT_ARMS = {tuple(auditor.COMPONENTS)!r}\n"
            f"NULL_ARMS = {tuple(auditor.NULLS)!r}\n"
            f"MARKER = {marker!r}\n"
        ).encode()

    helper.write_bytes(helper_source("captured"))
    captured = helper.read_bytes()
    monkeypatch.setattr(auditor, "R582_HELPER", helper)
    monkeypatch.setattr(auditor, "verify_preoutcome_authority", lambda: {})

    # This is the race v6 forbids: verification/capture has completed, then the
    # executable pathname changes before a recursive dependency opens it.
    helper.write_bytes(helper_source("swapped"))
    loaded = auditor.load_r582_helper()
    assert hashlib.sha256(captured).hexdigest() != digest(helper)
    assert loaded.MARKER == "captured"


@pytest.mark.xfail(
    strict=True,
    reason="the advertised R590 dry run transitively hashes R576/R579 outcome artifacts",
)
def test_dryrun_transitive_closure_never_opens_prior_outcomes(
    exact_candidate, monkeypatch,
):
    _, _, producer = exact_candidate
    auditor = producer.r588
    original_sha256 = auditor.sha256
    forbidden = {auditor.R576_RESULT.resolve(), auditor.R579_AUDIT.resolve()}

    def reject_outcome_read(path: Path) -> str:
        candidate = Path(path).resolve()
        if candidate in forbidden:
            raise AssertionError(f"model-free dry run reached outcome artifact: {candidate}")
        return original_sha256(path)

    monkeypatch.setattr(auditor, "sha256", reject_outcome_read)
    # The guard fires before either forbidden file is read.
    producer.run_dryrun()


@pytest.mark.xfail(
    strict=True,
    reason="result provenance re-hashes mutable producer/test paths after immutable loading",
)
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
    assert provenance[str(script)] == executed_digest
