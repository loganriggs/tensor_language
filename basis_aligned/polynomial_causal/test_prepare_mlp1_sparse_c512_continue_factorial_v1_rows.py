import copy
import json

import pytest
import torch

import prepare_mlp1_sparse_c512_continue_factorial_v1_rows as rows


def records(count: int):
    return [
        {
            "document_id": f"mlp1-factorial-doc-{index}",
            "dataset_document_index": 122_000 + index,
        }
        for index in range(count)
    ]


def test_configuration_matches_frozen_three_role_protocol_and_restores_parent():
    before = {name: getattr(rows.base, name) for name in rows._CONFIGURED_NAMES}
    with rows.configured_base() as configured:
        assert configured.START_DOCUMENT_INDEX == 122_000
        assert configured.DOCUMENTS_PER_ROLE == 96
        assert configured.TOTAL_DOCUMENTS == 288
        assert configured.ROLE_NAMES == ("FIT", "SELECT", "FINAL")
        assert configured.ROLE_AUTHORIZATIONS == rows.ROLE_AUTHORIZATIONS
        assert configured.SOURCE_PATHS == rows.SOURCE_PATHS
        assert configured.RECEIPT_SCHEMA == rows.RECEIPT_SCHEMA
        assert configured.FAILURE_SCHEMA == rows.FAILURE_SCHEMA
    assert all(
        getattr(rows.base, name) is value or getattr(rows.base, name) == value
        for name, value in before.items()
    )


def test_role_authorizations_are_mutually_firewalled():
    assert rows.ROLE_AUTHORIZATIONS == {
        "FIT": {
            "authorized_for_training": True,
            "authorized_for_selection": False,
            "authorized_for_final": False,
        },
        "SELECT": {
            "authorized_for_training": False,
            "authorized_for_selection": True,
            "authorized_for_final": False,
        },
        "FINAL": {
            "authorized_for_training": False,
            "authorized_for_selection": False,
            "authorized_for_final": True,
        },
    }


def test_split_rows_is_ordered_three_way_and_document_disjoint():
    tensor = torch.arange(288 * 257, dtype=torch.long).reshape(288, 257)
    split = rows.split_rows(tensor, records(288))
    assert tuple(split) == ("FIT", "SELECT", "FINAL")
    assert torch.equal(split["FIT"][0], tensor[:96])
    assert torch.equal(split["SELECT"][0], tensor[96:192])
    assert torch.equal(split["FINAL"][0], tensor[192:])
    identities = [
        {record["document_id"] for record in split[role][1]}
        for role in rows.ROLE_NAMES
    ]
    assert all(
        identities[left].isdisjoint(identities[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )


def test_validate_selected_accepts_registry_fresh_unique_documents():
    tensor = torch.arange(288 * 257, dtype=torch.long).reshape(288, 257)
    gates = rows.validate_selected(tensor, records(288), (set(), set(), [], set()))
    assert gates and all(gates.values())


def test_validate_selected_rejects_reused_document():
    tensor = torch.arange(288 * 257, dtype=torch.long).reshape(288, 257)
    reused = records(288)
    reused[-1] = copy.deepcopy(reused[0])
    with pytest.raises(RuntimeError, match="fresh MLP2 refit rows failed"):
        rows.validate_selected(tensor, reused, (set(), set(), [], set()))


def test_source_closure_contains_scientific_sources_and_generic_freezer_closure():
    assert len(rows.SOURCE_PATHS) == len(set(rows.SOURCE_PATHS))
    assert set(rows.DIRECT_SOURCES).issubset(rows.SOURCE_PATHS)
    assert set(rows.base.BASE.SOURCE_PATHS).issubset(rows.SOURCE_PATHS)
    assert {
        rows.PREREG,
        rows.RUNNER,
        rows.RUNNER_TEST,
        rows.PROGRAM,
        rows.PROGRAM_TEST,
        rows.FACADE,
        rows.FACADE_TEST,
        rows.JACCLUST_INIT,
        rows.JACCLUST_MODEL,
        rows.FREEZER,
        rows.TEST,
    }.issubset(rows.SOURCE_PATHS)
    assert set(rows.DIRECT_SOURCES) == {
        rows.PREREG, rows.RUNNER, rows.RUNNER_TEST, rows.PROGRAM, rows.PROGRAM_TEST,
        rows.FACADE, rows.FACADE_TEST, rows.JACCLUST_INIT, rows.JACCLUST_MODEL,
        rows.FREEZER, rows.TEST,
        rows.HERE / "prepare_mlp2_rank512_refit_v1_rows.py",
    }


def test_source_hashes_rejects_transitive_model_source_drift(monkeypatch):
    monkeypatch.setattr(rows.subprocess, "run", lambda *args, **kwargs: None)

    def committed_blob(command, cwd):
        relative = command[-1].split(":", 1)[1]
        return (rows.ROOT / relative).read_bytes()

    monkeypatch.setattr(rows.subprocess, "check_output", committed_blob)
    actual_hash = rows.file_sha256

    def drifted_hash(path):
        if path == rows.JACCLUST_MODEL:
            return "0" * 64
        return actual_hash(path)

    monkeypatch.setattr(rows, "file_sha256", drifted_hash)
    with pytest.raises(RuntimeError, match="uncommitted MLP1 factorial row-freezer source"):
        rows.source_hashes("c" * 40)


def test_audit_validation_requires_exact_source_bound_go(tmp_path, monkeypatch):
    sources = {"frozen.py": "1" * 64}
    value = {
        "schema": "mlp1_sparse_c512_continue_factorial_v1_rows_independent_audit",
        "status": "GO",
        "outcome_access": False,
        "audited_source_commit": "c" * 40,
        "audited_source_hashes": sources,
        "tests_passed": 11,
        "reviewer": "independent-test",
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(rows, "source_hashes", lambda commit: sources)
    parsed, digest = rows.validate_independent_audit(sources, path)
    assert parsed == value
    assert len(digest) == 64

    for field, replacement in (
        ("outcome_access", True),
        ("audited_source_hashes", {"frozen.py": "2" * 64}),
        ("status", "NO-GO"),
    ):
        mutated = copy.deepcopy(value)
        mutated[field] = replacement
        path.write_text(json.dumps(mutated))
        with pytest.raises(RuntimeError, match="not an exact source-bound GO"):
            rows.validate_independent_audit(sources, path)


def test_freeze_restores_parent_configuration_on_success_and_failure(monkeypatch):
    before = {name: getattr(rows.base, name) for name in rows._CONFIGURED_NAMES}

    def successful():
        assert rows.base.RECEIPT == rows.RECEIPT
        assert rows.base.LOCK == rows.LOCK
        assert rows.base.ROLE_NAMES == rows.ROLE_NAMES
        return {"status": "synthetic-success"}

    monkeypatch.setattr(rows.base, "freeze", successful)
    assert rows.freeze() == {"status": "synthetic-success"}
    assert all(
        getattr(rows.base, name) is value or getattr(rows.base, name) == value
        for name, value in before.items()
    )

    def failing():
        assert rows.base.FAILURE == rows.FAILURE
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(rows.base, "freeze", failing)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        rows.freeze()
    assert all(
        getattr(rows.base, name) is value or getattr(rows.base, name) == value
        for name, value in before.items()
    )


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


def test_configured_snapshot_rechecks_sources_registry_and_pinned_cache(
    tmp_path, monkeypatch,
):
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
        called = {"sources": 0, "registry": 0, "audit": 0}
        monkeypatch.setattr(
            configured,
            "source_hashes",
            lambda _commit: called.__setitem__("sources", called["sources"] + 1) or {},
        )
        monkeypatch.setattr(
            configured.BASE,
            "discover_registry_files",
            lambda: called.__setitem__("registry", called["registry"] + 1) or (),
        )
        monkeypatch.setattr(
            configured.BASE,
            "load_registry_exclusions",
            lambda _files: (set(), set(), [], set()),
        )
        monkeypatch.setattr(
            configured,
            "validate_independent_audit",
            lambda _sources: (
                called.__setitem__("audit", called["audit"] + 1) or {},
                "a" * 64,
            ),
        )
        parquet = tmp_path / "fineweb.parquet"
        parquet.write_bytes(b"pinned")
        monkeypatch.setattr(configured.BASE.BASE.local, "PINNED_SIZE", len(b"pinned"))
        monkeypatch.setattr(
            configured.BASE.BASE.local,
            "PINNED_SHA256",
            configured.file_sha256(parquet),
        )
        snapshot = {
            "commit": "c" * 40,
            "sources": {},
            "registry_files": (),
            "registry": (set(), set(), [], set()),
            "parquet": parquet,
        }
        try:
            configured.verify_snapshot(snapshot)
            configured.require_claim(claim, lock)
            assert called == {"sources": 1, "registry": 2, "audit": 1}
        finally:
            configured.release_claim(claim, lock)


def test_configured_registry_discovery_is_recursive_and_excludes_own_receipt(
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
        assert configured.BASE.discover_registry_files() == tuple(
            sorted((canonical.resolve(), prior.resolve()))
        )
