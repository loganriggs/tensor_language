from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ordered_successor_masks_v1 import OrderedLexicon, SuccessorMasks
import ordered_successor_tensor_discovery_v1 as v1
import prepare_ordered_successor_tensor_select_v2_rows as rows_v2


def _candidate_records(count: int) -> list[dict[str, object]]:
    return [
        {
            "document_id": f"fresh-{index}",
            "dataset_document_index": rows_v2.START_DOCUMENT_INDEX + index,
            "source_document_ordinal": index,
            "row_index": index,
            "chunk_id": 0,
            "token_start": 0,
        }
        for index in range(count)
    ]


def _masks_for_rows(candidate_rows: torch.Tensor, _lexicon: OrderedLexicon) -> SuccessorMasks:
    shape = (candidate_rows.shape[0], candidate_rows.shape[1] - 1)
    positive = torch.zeros(shape, dtype=torch.bool)
    wrong = torch.zeros(shape, dtype=torch.bool)
    none = torch.zeros(shape, dtype=torch.bool)
    positive[:30, :7] = True
    wrong[30:60, :7] = True
    none[60:90, :7] = True
    eligible = positive | wrong | none
    pair_index = torch.full(shape, -1, dtype=torch.int16)
    pair_index[eligible] = 0
    zero = torch.zeros(shape, dtype=torch.bool)
    return SuccessorMasks(
        eligible, positive, zero.clone(), zero.clone(), wrong, none,
        zero.clone(), pair_index,
    )


def _underpowered(candidate_rows: torch.Tensor, lexicon: OrderedLexicon) -> SuccessorMasks:
    masks = _masks_for_rows(candidate_rows, lexicon)
    zero = torch.zeros_like(masks.positive_clean)
    eligible = masks.positive_clean.clone()
    pair_index = torch.full_like(masks.pair_index, -1)
    pair_index[eligible] = 0
    return SuccessorMasks(
        eligible, masks.positive_clean, zero.clone(), zero.clone(), zero.clone(),
        zero.clone(), zero.clone(), pair_index,
    )


def test_support_first_allocation_is_deterministic_and_powered() -> None:
    candidates = torch.arange(240 * 257, dtype=torch.long).view(240, 257)
    records = _candidate_records(240)
    lexicon = OrderedLexicon("toy", ((1,), (2,)))
    first = rows_v2.allocate_powered_select(
        candidates, records, lexicon, mask_builder=_masks_for_rows,
    )
    second = rows_v2.allocate_powered_select(
        candidates, records, lexicon, mask_builder=_masks_for_rows,
    )
    selected, selected_records, masks, census = first
    assert torch.equal(first[0], second[0])
    assert first[1] == second[1] and first[3] == second[3]
    assert torch.equal(selected, candidates[:192])
    assert [item["candidate_scan_ordinal"] for item in selected_records] == list(range(192))
    assert [item["source_document_ordinal"] for item in selected_records] == list(range(192))
    assert tuple(masks.eligible_target.shape) == (192, 256)
    for name in rows_v2.protocol.POWERED_CELLS:
        assert census[name] == {"positions": 210, "documents": 30, "passed": True}
    assert rows_v2.pair_occupancy(masks) == {
        "0->1": {"positions": 630, "documents": 90},
        **{
            f"{index}->{index + 1}": {"positions": 0, "documents": 0}
            for index in range(1, 9)
        },
    }


def test_underpowered_candidate_scan_fails_without_allocation() -> None:
    candidates = torch.zeros(192, 257, dtype=torch.long)
    with pytest.raises(RuntimeError, match="cannot power every successor cell"):
        rows_v2.allocate_powered_select(
            candidates, _candidate_records(192), OrderedLexicon("toy", ((1,), (2,))),
            mask_builder=_underpowered,
        )


def test_arm_registry_omits_only_nonpromotive_diagnostics() -> None:
    rows_v2.protocol.validate_registry()
    assert rows_v2.V2_ARM_NAMES == v1.ARM_NAMES[:-2]
    assert len(rows_v2.V2_ARM_NAMES) == 15
    assert v1.CURRENT_ONLY not in rows_v2.V2_ARM_NAMES
    assert v1.V1_ONLY not in rows_v2.V2_ARM_NAMES
    assert rows_v2.protocol.PROMOTIVE_ARMS == v1.PROMOTIVE_ARMS


def test_source_closure_is_unique_and_binds_amendment_registry_tests_and_base() -> None:
    expected = tuple(dict.fromkeys((
        *rows_v2.OWN_SOURCES,
        *(rows_v2.ROOT / relative for relative in rows_v2.protocol.SCORER_SOURCE_PATHS),
        *rows_v2.base.SOURCE_PATHS,
    )))
    assert rows_v2.SOURCE_PATHS == expected
    required = {
        rows_v2.AMENDMENT,
        rows_v2.HERE / "ordered_successor_digit_lexicon_v2.py",
        rows_v2.HERE / "ordered_successor_tensor_select_registry_v2.py",
        rows_v2.HERE / "test_ordered_successor_digit_lexicon_v2.py",
        rows_v2.HERE / "test_prepare_ordered_successor_tensor_select_v2_rows.py",
        rows_v2.ROOT / "jacclust/__init__.py",
        rows_v2.ROOT / "jacclust/tt_model.py",
    }
    assert required.issubset(rows_v2.SOURCE_PATHS)
    assert set(rows_v2.base.SOURCE_PATHS).issubset(rows_v2.SOURCE_PATHS)
    assert rows_v2.STATISTICS_SOURCES == tuple(
        rows_v2.ROOT / relative for relative in rows_v2.protocol.SCORER_SOURCE_PATHS
    )


def test_freezer_import_surface_is_model_free_in_fresh_process() -> None:
    names = (
        "ordered_successor_tensor_discovery_v1", "circuit_campaign_runtime",
        "bilin18_observed_model_facade", "jacclust.tt_model",
        "successor_attention_backend", "ordered_successor_tensor_select_statistics_v1",
    )
    code = (
        "import json,sys; import prepare_ordered_successor_tensor_select_v2_rows; "
        f"print(json.dumps({names!r})); "
        f"print(json.dumps([name for name in {names!r} if name in sys.modules]))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(rows_v2.HERE)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=rows_v2.ROOT, env=environment,
        check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == []


def test_prior_registry_membership_is_identical_before_and_after_self_install(
    tmp_path, monkeypatch,
) -> None:
    cache = tmp_path / ".rowcache_ordered_successor_tensor_select_v2"
    prior = tmp_path / "prior_receipt.json"
    other_manifest = tmp_path / "other_manifest.json"
    prior.write_text("{}")
    other_manifest.write_text("{}")
    monkeypatch.setattr(rows_v2, "CACHE", cache)

    def recursive_census():
        paths = [prior.resolve(), other_manifest.resolve()]
        own = cache / "select_manifest.json"
        if own.is_file():
            paths.append(own.resolve())
        return tuple(sorted(paths))

    monkeypatch.setattr(rows_v2.base, "discover_registry_files", recursive_census)
    before = rows_v2.discover_prior_registry_files()
    cache.mkdir()
    (cache / "select_manifest.json").write_text('{"schema":"self"}')
    after = rows_v2.discover_prior_registry_files()
    assert before == after == tuple(sorted((prior.resolve(), other_manifest.resolve())))


def test_exact_independent_audit_schema_and_source_binding(tmp_path, monkeypatch) -> None:
    commit = "a" * 40
    closure = {"source.py": "b" * 64}
    path = tmp_path / "audit.json"
    payload = {
        "schema": "ordered_successor_tensor_select_v2_rows_independent_audit",
        "status": "GO",
        "outcome_access": False,
        "audited_source_commit": commit,
        "audited_source_hashes": closure,
        "tests_passed": 1,
        "reviewer": "independent-test",
    }
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(rows_v2, "source_closure", lambda value: closure if value == commit else {})
    assert rows_v2.validate_independent_audit(path)[0] == payload
    payload["outcome_access"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="not an exact GO"):
        rows_v2.validate_independent_audit(path)


def test_guarded_writer_is_create_only_and_rival_terminal_wins(tmp_path, monkeypatch) -> None:
    receipt = tmp_path / "receipt.json"
    failure = tmp_path / "failure.json"
    monkeypatch.setattr(rows_v2, "RECEIPT", receipt)
    monkeypatch.setattr(rows_v2, "FAILURE", failure)
    rows_v2._write_json_create_only({"schema": "test"}, receipt, before_link=lambda: None)
    assert json.loads(receipt.read_text()) == {"schema": "test"}
    other = tmp_path / "other.json"

    def rival_guard() -> None:
        failure.write_text("{}")
        rows_v2._terminal_absent()

    with pytest.raises(RuntimeError, match="terminal already exists"):
        rows_v2._write_json_create_only({"schema": "other"}, other, before_link=rival_guard)
    assert not other.exists()


def test_postlink_fsync_and_temporary_cleanup_are_nonterminal_warnings(
    tmp_path, monkeypatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(
        rows_v2.base, "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("late fsync")),
    )
    real_unlink = Path.unlink

    def cleanup_fails(path, *args, **kwargs):
        if path.name.startswith(".receipt.json.tmp."):
            raise OSError("late cleanup")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", cleanup_fails)
    rows_v2._write_json_create_only({"schema": "terminal"}, receipt, before_link=lambda: None)
    assert json.loads(receipt.read_text()) == {"schema": "terminal"}


def test_installed_artifact_snapshot_binds_both_payload_and_manifest(
    tmp_path, monkeypatch,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(rows_v2, "CACHE", cache)
    payload = cache / "select_rows.pt"
    manifest = cache / "select_manifest.json"
    torch.save(torch.zeros(1, dtype=torch.long), payload)
    manifest.write_text("{}")
    entry = {"path": str(payload), "file_sha256": rows_v2.file_sha256(payload)}
    manifest_sha256 = rows_v2.file_sha256(manifest)
    assert rows_v2._artifact_snapshot(entry, manifest_sha256) == {
        "rows": entry["file_sha256"], "manifest": manifest_sha256,
    }
    manifest.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match="artifact hash changed"):
        rows_v2._artifact_snapshot(entry, manifest_sha256)


def test_inode_nonce_claim_rejects_replacement(tmp_path) -> None:
    lock = tmp_path / "claim.lock"
    claim = rows_v2.acquire_claim(lock)
    try:
        lock.unlink()
        lock.write_text("replacement\n")
        with pytest.raises(RuntimeError, match="claim changed"):
            rows_v2.require_claim(claim, lock)
    finally:
        rows_v2.release_claim(claim, lock)


def test_amendment_is_explicitly_prospective_and_no_execution_was_authorized() -> None:
    text = rows_v2.AMENDMENT.read_text()
    assert "V2 has exactly 15 arms" in text
    assert "No omitted diagnostic can promote or rescue an arm" in text
    assert "192 one-row-per-source-document" in text
    assert "must not be executed" in text
    assert "No freezer is run" in text
    assert not rows_v2.CACHE.exists()
    assert not rows_v2.RECEIPT.exists()
    assert not rows_v2.FAILURE.exists()
