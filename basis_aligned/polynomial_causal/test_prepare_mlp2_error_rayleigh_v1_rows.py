import copy
import json
from pathlib import Path

import pytest
import torch

import prepare_mlp2_error_rayleigh_v1_rows as rows


def records(count):
    return [
        {"document_id": f"doc-{index}", "dataset_document_index": 121_000 + index}
        for index in range(count)
    ]


def test_configured_base_freezes_roles_and_restores_parent_module():
    before = {name: getattr(rows.base, name) for name in rows._CONFIGURED_NAMES}
    with rows.configured_base() as configured:
        assert configured.START_DOCUMENT_INDEX == 121_000
        assert configured.DOCUMENTS_PER_ROLE == 32
        assert configured.TOTAL_DOCUMENTS == 64
        assert configured.ROLE_NAMES == ("DESIGN", "HELDOUT")
        assert configured.ROLE_AUTHORIZATIONS == rows.ROLE_AUTHORIZATIONS
        assert configured.SOURCE_PATHS == rows.SOURCE_PATHS
    assert all(getattr(rows.base, name) is value or getattr(rows.base, name) == value
               for name, value in before.items())


def test_split_rows_is_ordered_and_disjoint():
    tensor = torch.arange(64 * 257, dtype=torch.long).reshape(64, 257)
    split = rows.split_rows(tensor, records(64))
    assert set(split) == {"DESIGN", "HELDOUT"}
    assert torch.equal(split["DESIGN"][0], tensor[:32])
    assert torch.equal(split["HELDOUT"][0], tensor[32:])
    assert {x["document_id"] for x in split["DESIGN"][1]}.isdisjoint(
        x["document_id"] for x in split["HELDOUT"][1]
    )


def test_validate_selected_accepts_fresh_unique_documents():
    tensor = torch.arange(64 * 257, dtype=torch.long).reshape(64, 257)
    prior = (set(), set(), [], set())
    gates = rows.validate_selected(tensor, records(64), prior)
    assert all(gates.values())


def test_validate_selected_rejects_reused_document():
    tensor = torch.arange(64 * 257, dtype=torch.long).reshape(64, 257)
    reused = records(64)
    reused[-1] = copy.deepcopy(reused[0])
    with pytest.raises(RuntimeError, match="fresh MLP2 refit rows failed"):
        rows.validate_selected(tensor, reused, (set(), set(), [], set()))


def test_audit_validation_requires_exact_source_binding(tmp_path, monkeypatch):
    sources = {"a.py": "1" * 64}
    value = {
        "schema": "mlp2_error_rayleigh_v1_rows_independent_audit",
        "status": "GO", "outcome_access": False,
        "audited_source_commit": "c" * 40,
        "audited_source_hashes": sources, "tests_passed": 5,
        "reviewer": "independent-test",
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(rows, "source_hashes", lambda commit: sources)
    parsed, digest = rows.validate_independent_audit(sources, path)
    assert parsed == value and len(digest) == 64
    value["outcome_access"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="not an exact source-bound GO"):
        rows.validate_independent_audit(sources, path)


def test_freeze_restores_parent_configuration_on_success_and_failure(monkeypatch):
    before = {name: getattr(rows.base, name) for name in rows._CONFIGURED_NAMES}

    def successful():
        assert rows.base.RECEIPT == rows.RECEIPT
        assert rows.base.LOCK == rows.LOCK
        return {"status": "ok"}

    monkeypatch.setattr(rows.base, "freeze", successful)
    assert rows.freeze() == {"status": "ok"}
    assert all(getattr(rows.base, name) is value or getattr(rows.base, name) == value
               for name, value in before.items())

    def failing():
        assert rows.base.FAILURE == rows.FAILURE
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(rows.base, "freeze", failing)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        rows.freeze()
    assert all(getattr(rows.base, name) is value or getattr(rows.base, name) == value
               for name, value in before.items())


def test_configured_create_only_terminal_and_lock_guards(tmp_path):
    with rows.configured_base() as configured:
        receipt = tmp_path / "receipt.json"
        configured.write_json_create_only(receipt, {"status": "first"})
        with pytest.raises(FileExistsError):
            configured.write_json_create_only(receipt, {"status": "replacement"})
        assert json.loads(receipt.read_text()) == {"status": "first"}

        lock = tmp_path / "claim.lock"
        claim = configured.acquire_claim(lock)
        try:
            lock.unlink()
            lock.write_text("replacement\n")
            with pytest.raises(RuntimeError, match="replaced"):
                configured.require_claim(claim, lock)
        finally:
            configured.release_claim(claim, lock)


def test_configured_snapshot_rechecks_terminal_registry_and_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    receipt = tmp_path / "receipt.json"
    failure = tmp_path / "failure.json"
    lock = tmp_path / "claim.lock"
    cache.mkdir()
    with rows.configured_base() as configured:
        monkeypatch.setattr(configured, "CACHE", cache)
        monkeypatch.setattr(configured, "RECEIPT", receipt)
        monkeypatch.setattr(configured, "FAILURE", failure)
        monkeypatch.setattr(configured, "LOCK", lock)
        claim = configured.acquire_claim(lock)
        called = {"sources": 0, "registry": 0}
        monkeypatch.setattr(configured, "source_hashes", lambda _commit: called.__setitem__(
            "sources", called["sources"] + 1) or {})
        monkeypatch.setattr(configured.BASE, "discover_registry_files", lambda: called.__setitem__(
            "registry", called["registry"] + 1) or ())
        monkeypatch.setattr(configured.BASE, "load_registry_exclusions",
                            lambda _files: (set(), set(), set(), set()))
        parquet = tmp_path / "fineweb.parquet"
        parquet.write_bytes(b"pinned")
        monkeypatch.setattr(configured.BASE.BASE.local, "PINNED_SIZE", len(b"pinned"))
        monkeypatch.setattr(configured.BASE.BASE.local, "PINNED_SHA256",
                            configured.file_sha256(parquet))
        monkeypatch.setattr(configured, "validate_independent_audit", lambda _sources: ({}, "a"))
        snapshot = {
            "commit": "c" * 40, "sources": {}, "registry_files": (),
            "registry": (set(), set(), set(), set()), "parquet": parquet,
        }
        try:
            configured.verify_snapshot(snapshot)
            configured.require_claim(claim, lock)
            assert called["sources"] == 1 and called["registry"] == 2
        finally:
            configured.release_claim(claim, lock)


def test_configured_registry_discovery_remains_recursive_and_excludes_own_receipt(
    tmp_path, monkeypatch,
):
    basis = tmp_path / "basis_aligned"
    nested = basis / "nested" / "deeper"
    nested.mkdir(parents=True)
    canonical = basis / "canonical_receipt.json"
    prior = nested / "prior_rows_receipt.json"
    own = nested / rows.RECEIPT.name
    for path in (canonical, prior, own):
        path.write_text("{}")
    with rows.configured_base() as configured:
        monkeypatch.setattr(configured.BASE, "BASIS", basis)
        monkeypatch.setattr(configured.BASE.BASE, "CANONICAL_RECEIPT", canonical)
        monkeypatch.setattr(configured.BASE, "RECEIPT", own)
        assert configured.BASE.discover_registry_files() == tuple(sorted(
            (canonical.resolve(), prior.resolve())
        ))


def test_source_closure_includes_both_direct_test_modules():
    assert rows.TEST in rows.SOURCE_PATHS
    assert rows.METRICS_TEST in rows.SOURCE_PATHS
