from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ordered_successor_masks_v1 import OrderedLexicon, SuccessorMasks
import prepare_ordered_successor_tensor_select_v3_rows as rows_v3


def _records(count: int) -> list[dict[str, object]]:
    return [{"document_id": f"doc-{index}"} for index in range(count)]


def _known_335_masks(rows: torch.Tensor, _lexicon: OrderedLexicon) -> SuccessorMasks:
    shape = (len(rows), 256)
    positive = torch.zeros(shape, dtype=torch.bool)
    wrong = torch.zeros(shape, dtype=torch.bool)
    none = torch.zeros(shape, dtype=torch.bool)
    positive[:199, 0] = True
    none[:199, 1] = True
    wrong[199:334, 0] = True
    positive[334, 65] = True
    wrong[334, :65] = True
    none[334, 66] = True
    eligible = positive | wrong | none
    pair_index = torch.full(shape, -1, dtype=torch.int16)
    pair_index[eligible] = 0
    zero = torch.zeros(shape, dtype=torch.bool)
    return SuccessorMasks(
        eligible, positive, zero.clone(), zero.clone(), wrong, none,
        zero.clone(), pair_index,
    )


def test_namespace_is_fresh_distinct_and_prospectively_no_go() -> None:
    assert rows_v3.N_SELECT == 384
    assert rows_v3.CACHE != rows_v3.v2.CACHE
    assert rows_v3.RECEIPT != rows_v3.v2.RECEIPT
    assert rows_v3.FAILURE != rows_v3.v2.FAILURE
    assert rows_v3.LOCK != rows_v3.v2.LOCK
    assert rows_v3.AUDIT != rows_v3.v2.AUDIT
    assert rows_v3.TERMINAL not in {
        rows_v3.RECEIPT, rows_v3.FAILURE, rows_v3.LOCK, rows_v3.AUDIT,
    }
    assert not rows_v3.CACHE.exists()
    assert not rows_v3.RECEIPT.exists()
    assert not rows_v3.FAILURE.exists()
    assert not rows_v3.TERMINAL.exists()
    assert rows_v3.AUDIT.is_file()
    with pytest.raises(RuntimeError, match="not an exact GO"):
        rows_v3.validate_independent_audit()


def test_source_closure_is_exact_unique_and_transitive() -> None:
    expected = tuple(dict.fromkeys((*rows_v3.OWN_SOURCES, *rows_v3.v2.SOURCE_PATHS)))
    assert rows_v3.SOURCE_PATHS == expected
    assert len(rows_v3.SOURCE_PATHS) == len(set(rows_v3.SOURCE_PATHS)) == 43
    assert set(rows_v3.v2.SOURCE_PATHS).issubset(rows_v3.SOURCE_PATHS)
    assert rows_v3.budget.V2_AUDIT in rows_v3.SOURCE_PATHS
    assert rows_v3.budget.V2_FAILURE in rows_v3.SOURCE_PATHS
    assert rows_v3.AMENDMENT in rows_v3.SOURCE_PATHS
    assert rows_v3.FREEZER in rows_v3.SOURCE_PATHS
    assert rows_v3.TEST in rows_v3.SOURCE_PATHS


def test_freezer_import_surface_is_model_free_in_fresh_process() -> None:
    forbidden = (
        "ordered_successor_tensor_discovery_v1", "circuit_campaign_runtime",
        "bilin18_observed_model_facade", "jacclust.tt_model",
        "successor_attention_backend", "ordered_successor_tensor_select_statistics_v1",
    )
    code = (
        "import json,sys; import prepare_ordered_successor_tensor_select_v3_rows; "
        f"print(json.dumps([name for name in {forbidden!r} if name in sys.modules]))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(rows_v3.HERE)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=rows_v3.ROOT, env=environment,
        check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == []


def test_exact_v2_parent_lineage_and_terminal_state_replay() -> None:
    snapshot = rows_v3._v2_lineage_snapshot()
    assert snapshot == {
        "lineage": rows_v3.budget.validate_v2_lineage(),
        "v2_cache_exists": False,
        "v2_receipt_exists": False,
        "v2_failure_exists": True,
    }


def test_v2_cache_or_receipt_reuse_fails_closed(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "old-cache"
    receipt = tmp_path / "old-receipt.json"
    monkeypatch.setattr(rows_v3.v2, "CACHE", cache)
    monkeypatch.setattr(rows_v3.v2, "RECEIPT", receipt)
    cache.mkdir()
    with pytest.raises(RuntimeError, match="spent successor v2 terminal namespace changed"):
        rows_v3._v2_lineage_snapshot()
    cache.rmdir()
    receipt.write_text("{}")
    with pytest.raises(RuntimeError, match="spent successor v2 terminal namespace changed"):
        rows_v3._v2_lineage_snapshot()


def test_exact_384_budget_replays_registered_335_stop() -> None:
    rows = torch.zeros(400, 257, dtype=torch.long)
    result = rows_v3.allocate_powered_select(
        rows, _records(400), OrderedLexicon("toy", ((1,), (2,))),
        mask_builder=_known_335_masks,
    )
    assert result.support_first_count == 335
    assert result.support_first_last_candidate == 334
    assert tuple(result.selected_rows.shape) == (384, 257)
    assert len(result.selected_records) == 384
    assert all(result.census[name]["passed"] for name in rows_v3.protocol.POWERED_CELLS)
    assert sum(value["positions"] for value in result.pair_occupancy.values()) == 600


def test_exact_independent_audit_schema_and_source_binding(tmp_path: Path, monkeypatch) -> None:
    commit = "a" * 40
    closure = {"source.py": "b" * 64}
    audit = tmp_path / "audit.json"
    payload = {
        "schema": "ordered_successor_tensor_select_v3_rows_independent_audit",
        "status": "GO",
        "outcome_access": False,
        "audited_source_commit": commit,
        "audited_source_hashes": closure,
        "tests_passed": 1,
        "reviewer": "independent-test",
    }
    audit.write_text(json.dumps(payload))
    monkeypatch.setattr(rows_v3, "source_closure", lambda value: closure if value == commit else {})
    assert rows_v3.validate_independent_audit(audit)[0] == payload
    payload["outcome_access"] = True
    audit.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="not an exact GO"):
        rows_v3.validate_independent_audit(audit)


def test_prior_registry_census_ignores_only_own_validated_manifest(
    tmp_path: Path, monkeypatch,
) -> None:
    cache = tmp_path / "v3-cache"
    prior = tmp_path / "prior_receipt.json"
    other = tmp_path / "other_manifest.json"
    prior.write_text("{}")
    other.write_text("{}")
    monkeypatch.setattr(rows_v3, "CACHE", cache)

    def census() -> tuple[Path, ...]:
        paths = [prior.resolve(), other.resolve()]
        own = cache / rows_v3.MANIFEST_NAME
        if own.is_file():
            paths.append(own.resolve())
        return tuple(sorted(paths))

    monkeypatch.setattr(rows_v3.v2.base, "discover_registry_files", census)
    before = rows_v3.discover_prior_registry_files()
    cache.mkdir()
    (cache / rows_v3.MANIFEST_NAME).write_text("{}")
    after = rows_v3.discover_prior_registry_files()
    assert before == after == tuple(sorted((prior.resolve(), other.resolve())))


def test_payload_and_manifest_semantic_replay(tmp_path: Path, monkeypatch) -> None:
    rows = torch.zeros(384, 257, dtype=torch.long)
    payload = tmp_path / "rows.pt"
    torch.save(rows, payload)
    entry = {
        "file_sha256": rows_v3.file_sha256(payload),
        "tensor_sha256": rows_v3.tensor_sha256(rows),
    }
    assert torch.equal(rows_v3._validate_payload(payload, entry), rows)
    entry["tensor_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="semantic replay failed"):
        rows_v3._validate_payload(payload, entry)

    manifest = tmp_path / "manifest.json"
    expected = {"schema": "test", "value": 1}
    manifest.write_text(json.dumps(expected))
    digest = rows_v3.file_sha256(manifest)
    rows_v3._validate_manifest(manifest, expected, digest)
    manifest.write_text(json.dumps({"schema": "changed"}))
    with pytest.raises(RuntimeError, match="semantic replay failed"):
        rows_v3._validate_manifest(manifest, expected, digest)


def test_guarded_terminal_writer_is_create_only_and_receipt_last(
    tmp_path: Path, monkeypatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    failure = tmp_path / "failure.json"
    terminal = tmp_path / "terminal.json"
    lock = tmp_path / "lock"
    monkeypatch.setattr(rows_v3, "RECEIPT", receipt)
    monkeypatch.setattr(rows_v3, "FAILURE", failure)
    monkeypatch.setattr(rows_v3, "TERMINAL", terminal)
    calls: list[str] = []
    claim = rows_v3.acquire_claim(lock)
    try:
        rows_v3._publish_terminal_json(
            {"schema": "terminal"}, kind="receipt", claim=claim,
            before_claim=lambda: calls.append("guard"),
        )
        assert calls == ["guard"]
        assert json.loads(receipt.read_text()) == {"schema": "terminal"}
        assert json.loads(terminal.read_text())["kind"] == "receipt"
        with pytest.raises(RuntimeError, match="terminal already exists"):
            rows_v3._publish_terminal_json(
                {"schema": "replacement"}, kind="failure", claim=claim,
                before_claim=lambda: None,
            )
    finally:
        rows_v3.release_claim(claim, lock)


def test_postlink_fsync_and_cleanup_errors_cannot_reverse_terminal(
    tmp_path: Path, monkeypatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    failure = tmp_path / "failure.json"
    terminal = tmp_path / "terminal.json"
    lock = tmp_path / "lock"
    monkeypatch.setattr(rows_v3, "RECEIPT", receipt)
    monkeypatch.setattr(rows_v3, "FAILURE", failure)
    monkeypatch.setattr(rows_v3, "TERMINAL", terminal)
    monkeypatch.setattr(
        rows_v3.v2.base, "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("late fsync")),
    )
    real_unlink = Path.unlink

    def cleanup_fails(path: Path, *args, **kwargs):
        if ".tmp." in path.name:
            raise OSError("late cleanup")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", cleanup_fails)
    claim = rows_v3.acquire_claim(lock)
    try:
        rows_v3._publish_terminal_json(
            {"schema": "terminal"}, kind="receipt", claim=claim,
            before_claim=lambda: None,
        )
        assert json.loads(receipt.read_text()) == {"schema": "terminal"}
        assert json.loads(terminal.read_text())["kind"] == "receipt"
        assert not failure.exists()
    finally:
        rows_v3.release_claim(claim, lock)


@pytest.mark.parametrize(
    ("first_kind", "rival_kind"), (("receipt", "failure"), ("failure", "receipt")),
)
def test_opposite_terminal_race_after_callback_has_exactly_one_winner(
    tmp_path: Path, monkeypatch, first_kind: str, rival_kind: str,
) -> None:
    """Immutable-audit reproduction: rival links after callback, before own link."""

    receipt = tmp_path / "receipt.json"
    failure = tmp_path / "failure.json"
    terminal = tmp_path / "terminal.json"
    lock = tmp_path / "lock"
    monkeypatch.setattr(rows_v3, "RECEIPT", receipt)
    monkeypatch.setattr(rows_v3, "FAILURE", failure)
    monkeypatch.setattr(rows_v3, "TERMINAL", terminal)
    claim = rows_v3.acquire_claim(lock)
    real_link = os.link
    injected = False

    def race_link(source, target, *args, **kwargs):
        nonlocal injected
        if Path(target) == terminal and not injected:
            injected = True
            rows_v3._publish_terminal_json(
                {"schema": rival_kind}, kind=rival_kind, claim=claim,
                before_claim=lambda: None,
            )
        return real_link(source, target, *args, **kwargs)

    try:
        monkeypatch.setattr(os, "link", race_link)
        with pytest.raises(FileExistsError):
            rows_v3._publish_terminal_json(
                {"schema": first_kind}, kind=first_kind, claim=claim,
                before_claim=lambda: None,
            )
        assert json.loads(terminal.read_text())["kind"] == rival_kind
        assert (receipt.exists(), failure.exists()) == (
            (True, False) if rival_kind == "receipt" else (False, True)
        )
    finally:
        rows_v3.release_claim(claim, lock)


def test_receipt_exists_suppresses_contradictory_failure(tmp_path: Path, monkeypatch) -> None:
    receipt = tmp_path / "receipt.json"
    failure = tmp_path / "failure.json"
    terminal = tmp_path / "terminal.json"
    monkeypatch.setattr(rows_v3, "RECEIPT", receipt)
    monkeypatch.setattr(rows_v3, "FAILURE", failure)
    monkeypatch.setattr(rows_v3, "TERMINAL", terminal)
    monkeypatch.setattr(rows_v3, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(rows_v3, "acquire_claim", lambda: object())
    monkeypatch.setattr(rows_v3, "release_claim", lambda _claim: None)

    def linked_then_raises(_claim):
        receipt.write_text('{"schema":"receipt"}')
        raise OSError("post-link warning escaped")

    monkeypatch.setattr(rows_v3, "freeze_locked", linked_then_raises)
    with pytest.raises(OSError, match="post-link warning escaped"):
        rows_v3.freeze()
    assert receipt.is_file()
    assert not failure.exists()
    assert not terminal.exists()


def test_failure_before_receipt_is_create_only_and_terminal(tmp_path: Path, monkeypatch) -> None:
    receipt = tmp_path / "receipt.json"
    failure = tmp_path / "failure.json"
    terminal = tmp_path / "terminal.json"
    cache = tmp_path / "cache"
    monkeypatch.setattr(rows_v3, "RECEIPT", receipt)
    monkeypatch.setattr(rows_v3, "FAILURE", failure)
    monkeypatch.setattr(rows_v3, "TERMINAL", terminal)
    monkeypatch.setattr(rows_v3, "CACHE", cache)
    monkeypatch.setattr(rows_v3, "acquire_claim", lambda: object())
    monkeypatch.setattr(rows_v3, "release_claim", lambda _claim: None)
    monkeypatch.setattr(rows_v3, "require_claim", lambda _claim: None)
    monkeypatch.setattr(
        rows_v3, "freeze_locked",
        lambda _claim: (_ for _ in ()).throw(RuntimeError("synthetic pre-link failure")),
    )
    with pytest.raises(RuntimeError, match="synthetic pre-link failure"):
        rows_v3.freeze()
    value = json.loads(failure.read_text())
    assert value == {
        "schema": "ordered_successor_tensor_select_v3_rows_failure",
        "status": "terminal_failure_no_receipt",
        "error_type": "RuntimeError",
        "error": "synthetic pre-link failure",
        "cache_exists": False,
        "outcome_access": False,
    }
    assert not receipt.exists()
    assert json.loads(terminal.read_text())["kind"] == "failure"


def test_inode_nonce_claim_rejects_replacement(tmp_path: Path) -> None:
    lock = tmp_path / "claim.lock"
    claim = rows_v3.acquire_claim(lock)
    try:
        lock.unlink()
        lock.write_text("replacement\n")
        with pytest.raises(RuntimeError, match="claim changed"):
            rows_v3.require_claim(claim, lock)
    finally:
        rows_v3.release_claim(claim, lock)


def test_claim_open_fstat_rejects_exact_stat_read_inode_swap(
    tmp_path: Path, monkeypatch,
) -> None:
    """Immutable-audit reproduction: replace path after stable open, before read."""

    lock = tmp_path / "claim.lock"
    claim = rows_v3.acquire_claim(lock)
    real_open = os.open
    swapped = False

    def open_then_swap(path, flags, *args, **kwargs):
        nonlocal swapped
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == lock and not swapped:
            swapped = True
            nonce = lock.read_bytes()
            lock.unlink()
            lock.write_bytes(nonce)
        return descriptor

    try:
        monkeypatch.setattr(os, "open", open_then_swap)
        with pytest.raises(RuntimeError, match="claim changed"):
            rows_v3.require_claim(claim, lock)
    finally:
        rows_v3.release_claim(claim, lock)
