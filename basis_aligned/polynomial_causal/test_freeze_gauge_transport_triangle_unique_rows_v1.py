from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch

import freeze_gauge_transport_triangle_unique_rows_v1 as freezer


def _records(source: str, documents: int, chunks: int = 2, *, start: int) -> list[dict]:
    output = []
    for document in range(documents):
        for chunk in range(chunks):
            output.append({
                "document_id": f"{source}-document-{document}",
                "dataset_document_index": start + document,
                "chunk_id": chunk,
                "token_start": chunk * freezer.TOKEN_LENGTH,
            })
    return output


def _synthetic_parent() -> dict:
    return {
        "document_provenance": {
            "schema_version": 1,
            "sets": {
                "n480_skip80": _records("early", 209, start=80) + [
                    {
                        "document_id": f"early-document-{index % 209}",
                        "dataset_document_index": 80 + index % 209,
                        "chunk_id": 2,
                        "token_start": 2 * freezer.TOKEN_LENGTH,
                    }
                    for index in range(62)
                ],
                "n192_skip7000": _records("middle", 79, start=7000) + [
                    {
                        "document_id": f"middle-document-{index % 79}",
                        "dataset_document_index": 7000 + index % 79,
                        "chunk_id": 2,
                        "token_start": 2 * freezer.TOKEN_LENGTH,
                    }
                    for index in range(34)
                ],
                "n192_skip11000": _records("late", 87, start=11000) + [
                    {
                        "document_id": f"late-extra-{index}",
                        "dataset_document_index": 11100 + index,
                        "chunk_id": 0,
                        "token_start": 0,
                    }
                    for index in range(18)
                ],
                "n96_skip1200": (
                    _records("bridge", 33, start=1200)
                    + _records("bridge", 15, start=1200)
                ),
            },
        },
    }


def test_synthetic_selection_is_one_row_per_document_and_role_disjoint():
    plan = freezer.build_selection_plan(_synthetic_parent())
    assert plan["role_sizes"] == freezer.ROLE_SIZES
    assert plan["source_contributions"] == freezer.EXPECTED_CONTRIBUTIONS
    documents = [
        record["document_id"]
        for records in plan["roles"].values()
        for record in records
    ]
    assert len(documents) == len(set(documents)) == 384
    assert all(record["chunk_id"] == 0 for records in plan["roles"].values() for record in records)


def test_current_parent_metadata_has_the_frozen_known_answer_without_tensor_load(monkeypatch):
    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("metadata selection must not deserialize row tensors")

    monkeypatch.setattr(torch, "load", forbidden_load)
    parent = freezer.load_parent_metadata()
    plan = freezer.build_selection_plan(parent)
    assert plan["source_contributions"] == {
        "basis": {"n480_skip80": 96},
        "fit": {"n480_skip80": 96},
        "evaluation": {
            "n192_skip11000": 105,
            "n192_skip7000": 79,
            "n96_skip1200": 8,
        },
    }
    assert plan["unique_document_count"] == 384
    assert max(record["dataset_document_index"] for record in plan["roles"]["fit"]) < min(
        record["dataset_document_index"] for record in plan["roles"]["evaluation"]
    )


def test_parent_receipt_hash_is_checked_before_and_after_read(monkeypatch, tmp_path: Path):
    parent = tmp_path / "parent.json"
    parent.write_text("{}")
    expected = freezer.file_sha256(parent)
    observed = iter((expected, "0" * 64))
    monkeypatch.setattr(freezer, "PARENT_RECEIPT", parent)
    monkeypatch.setattr(freezer, "PARENT_RECEIPT_SHA256", expected)
    monkeypatch.setattr(freezer, "file_sha256", lambda _path: next(observed))
    with pytest.raises(RuntimeError, match="parent receipt changed"):
        freezer.load_parent_metadata()


def test_metadata_shortfall_fails_instead_of_reusing_a_document():
    parent = _synthetic_parent()
    parent["document_provenance"]["sets"]["n96_skip1200"] = []
    with pytest.raises(RuntimeError, match="incomplete|exhausted"):
        freezer.build_selection_plan(parent)


def test_rows_payload_rejects_record_or_tensor_tampering(monkeypatch):
    monkeypatch.setattr(freezer, "EXPECTED_CONTRIBUTIONS", {
        "basis": {"basis_pool": 2},
        "fit": {"fit_pool": 2},
        "evaluation": {"eval_pool": 2},
    })
    monkeypatch.setattr(freezer, "ROLE_SIZES", {"basis": 2, "fit": 2, "evaluation": 2})
    plan_roles = {}
    caches = {}
    for role, source in (("basis", "basis_pool"), ("fit", "fit_pool"), ("evaluation", "eval_pool")):
        caches[source] = torch.arange(2 * freezer.TOKEN_LENGTH, dtype=torch.long).reshape(
            2, freezer.TOKEN_LENGTH
        )
        plan_roles[role] = [
            {
                "source_key": source,
                "source_row_index": index,
                "document_id": f"{role}-{index}",
                "dataset_document_index": 1000 * (1 + len(plan_roles)) + index,
                "chunk_id": 0,
                "token_start": 0,
                "role_index": index,
            }
            for index in range(2)
        ]
    plan_body = {
        "schema": "gauge_transport_triangle_unique_rows_v1_selection_plan",
        "role_sizes": freezer.ROLE_SIZES,
        "role_pool_order": {},
        "source_contributions": freezer.EXPECTED_CONTRIBUTIONS,
        "roles": plan_roles,
        "unique_document_count": 6,
    }
    authority = {
        "authority_sha256": "a" * 64,
        "selection_plan": {
            **plan_body,
            "selection_plan_sha256": freezer.canonical_sha256(plan_body),
        },
    }
    payload = freezer.build_rows_payload(authority, caches)
    freezer.validate_rows_payload(payload, authority)

    changed = copy.deepcopy(payload)
    changed["records"]["fit"][0]["document_id"] = "changed"
    with pytest.raises(RuntimeError, match="role changed"):
        freezer.validate_rows_payload(changed, authority)
    changed = copy.deepcopy(payload)
    changed["roles"]["evaluation"] = changed["roles"]["evaluation"][:, :-1]
    with pytest.raises(RuntimeError, match="role changed"):
        freezer.validate_rows_payload(changed, authority)


def test_json_publication_is_create_only(tmp_path: Path):
    path = tmp_path / "receipt.json"
    freezer.publish_json(path, {"status": "first"})
    assert json.loads(path.read_text()) == {"status": "first"}
    with pytest.raises(FileExistsError):
        freezer.publish_json(path, {"status": "second"})


def _rehash_authority(authority: dict) -> None:
    body = {key: authority[key] for key in authority if key != "authority_sha256"}
    authority["authority_sha256"] = freezer.canonical_sha256(body)


@pytest.mark.parametrize("field", ["cache_bindings", "permissions", "outputs", "schema"])
def test_authority_rejects_self_consistent_schema_or_contract_mutation(monkeypatch, field: str):
    source = {"commit": "synthetic", "paths": {}, "sha256": "s" * 64}
    parent = _synthetic_parent()
    authority = freezer.build_authority(source, parent)
    monkeypatch.setattr(freezer, "validate_source_closure", lambda _value: None)
    monkeypatch.setattr(freezer, "load_parent_metadata", lambda: parent)
    freezer.validate_authority(authority)

    changed = copy.deepcopy(authority)
    if field == "cache_bindings":
        changed[field]["n480_skip80"]["file_sha256"] = "0" * 64
    elif field == "permissions":
        changed[field]["triangle_runner_authorized_by_this_authority"] = True
    elif field == "outputs":
        changed[field]["receipt"] = "/tmp/wrong-receipt.json"
    else:
        changed["extra_self_hashed_field"] = "not frozen"
    _rehash_authority(changed)
    with pytest.raises(RuntimeError, match="schema changed|exact rebuilt authority"):
        freezer.validate_authority(changed)


def test_partial_atomic_publication_leaves_no_final_or_temporary(monkeypatch, tmp_path: Path):
    final = tmp_path / "artifact.json"

    def partial_then_fail(descriptor: int, payload: bytes) -> None:
        assert os.write(descriptor, payload[:3]) == 3
        raise OSError("synthetic short publication")

    monkeypatch.setattr(freezer, "_write_all", partial_then_fail)
    with pytest.raises(OSError, match="synthetic short publication"):
        freezer.publish_json(final, {"status": "never-visible"})
    assert not final.exists()
    assert list(tmp_path.iterdir()) == []


def test_owner_lock_rejects_inode_replacement_and_preserves_foreign_lock(
    monkeypatch, tmp_path: Path,
):
    lock = tmp_path / "owner.lock"
    monkeypatch.setattr(freezer, "LOCK", lock)
    owner = freezer.acquire_owner_lock()
    lock.unlink()
    lock.write_text('{"nonce":"foreign"}\n')
    with pytest.raises(RuntimeError, match="replaced"):
        freezer.assert_owner_lock(owner)
    with pytest.raises(RuntimeError, match="replaced"):
        freezer.release_owner_lock(owner)
    assert lock.read_text() == '{"nonce":"foreign"}\n'


def test_owner_lock_rejects_same_inode_content_mutation(monkeypatch, tmp_path: Path):
    lock = tmp_path / "owner.lock"
    monkeypatch.setattr(freezer, "LOCK", lock)
    owner = freezer.acquire_owner_lock()
    lock.write_text('{"nonce":"edited"}\n')
    with pytest.raises(RuntimeError, match="contents changed"):
        freezer.assert_owner_lock(owner)


def test_atomic_publication_rechecks_lock_before_final_link(monkeypatch, tmp_path: Path):
    lock = tmp_path / "owner.lock"
    final = tmp_path / "artifact.json"
    monkeypatch.setattr(freezer, "LOCK", lock)
    owner = freezer.acquire_owner_lock()
    original_write = freezer._write_all

    def write_then_replace_lock(descriptor: int, payload: bytes) -> None:
        original_write(descriptor, payload)
        lock.unlink()
        lock.write_text('{"nonce":"replacement"}\n')

    monkeypatch.setattr(freezer, "_write_all", write_then_replace_lock)
    with pytest.raises(RuntimeError, match="replaced"):
        freezer.publish_json(final, {"status": "must-not-link"}, owner=owner)
    assert not final.exists()
    assert lock.exists()
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_cache_loader_rejects_byte_and_tensor_mutation(tmp_path: Path):
    path = tmp_path / "cache.pt"
    tensor = torch.arange(12, dtype=torch.long).reshape(3, 4)
    torch.save(tensor, path)
    authority = {"cache_bindings": {"cache": {
        "path": str(path),
        "shape": [3, 4],
        "dtype": "torch.int64",
        "file_sha256": freezer.file_sha256(path),
        "tensor_sha256": freezer.tensor_sha256(tensor),
    }}}
    loaded = freezer.load_cache_tensors(authority)
    assert torch.equal(loaded["cache"], tensor)

    torch.save(tensor + 1, path)
    with pytest.raises(RuntimeError, match="cache tensor changed"):
        freezer.load_cache_tensors(authority)


def test_manifest_rejects_self_consistent_permission_mutation(monkeypatch, tmp_path: Path):
    rows = tmp_path / "rows.pt"
    rows.write_bytes(b"synthetic-row-bytes")
    monkeypatch.setattr(freezer, "ROWS", rows)
    monkeypatch.setattr(freezer, "ROLE_SIZES", {"basis": 1})
    monkeypatch.setattr(freezer, "EXPECTED_CONTRIBUTIONS", {"basis": {"cache": 1}})
    authority = {
        "authority_sha256": "a" * 64,
        "permissions": {"conditional_future_row_eligibility": "separate runner required"},
    }
    payload = {
        "selection_plan_sha256": "p" * 64,
        "roles": {"basis": torch.tensor([[1, 2]], dtype=torch.long)},
        "records": {"basis": [{"document_id": "one"}]},
    }
    manifest = freezer.build_manifest(payload, authority)
    freezer.validate_manifest(manifest, payload, authority)
    changed = copy.deepcopy(manifest)
    changed["triangle_runner_authorized_by_this_manifest"] = True
    changed_body = {key: changed[key] for key in changed if key != "manifest_sha256"}
    changed["manifest_sha256"] = freezer.canonical_sha256(changed_body)
    with pytest.raises(RuntimeError, match="exact rebuilt manifest"):
        freezer.validate_manifest(changed, payload, authority)


def test_strict_receipt_reload_rejects_self_consistent_mutation(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(freezer, "ROLE_SIZES", {"basis": 1})
    authority = {
        "authority_sha256": "a" * 64,
        "source_closure": {"sha256": "s" * 64},
    }
    authority_path = tmp_path / "authority.json"
    failure_path = tmp_path / "failure.json"
    authority_path.write_text(json.dumps(authority))
    monkeypatch.setattr(freezer, "AUTHORITY", authority_path)
    monkeypatch.setattr(freezer, "FAILURE", failure_path)
    payload = {
        "selection_plan_sha256": "p" * 64,
        "roles": {"basis": torch.tensor([[1, 2]], dtype=torch.long)},
    }
    manifest = {"manifest_sha256": "m" * 64}
    replay = {
        "authority_file_sha256": freezer.file_sha256(authority_path),
        "rows_file_sha256": "r" * 64,
        "manifest_file_sha256": "f" * 64,
        "cache_file_sha256s": {"cache": "c" * 64},
    }
    monkeypatch.setattr(
        freezer, "replay_terminal_state", lambda expected: (payload, manifest, replay),
    )
    receipt = freezer.build_receipt(authority, payload, manifest, replay)
    freezer.validate_receipt(receipt)
    assert receipt["failure_absent"] is True

    failure_path.write_text("{}")
    with pytest.raises(RuntimeError, match="receipt/failure exclusivity"):
        freezer.validate_receipt(receipt)
    failure_path.unlink()

    def replay_with_late_failure(_expected):
        failure_path.write_text("{}")
        return payload, manifest, replay

    monkeypatch.setattr(freezer, "replay_terminal_state", replay_with_late_failure)
    with pytest.raises(RuntimeError, match="appeared during receipt validation"):
        freezer.validate_receipt(receipt)
    failure_path.unlink()
    monkeypatch.setattr(
        freezer, "replay_terminal_state", lambda expected: (payload, manifest, replay),
    )

    changed = copy.deepcopy(receipt)
    changed["triangle_runner_authorized_by_this_receipt"] = True
    changed_body = {key: changed[key] for key in changed if key != "receipt_sha256"}
    changed["receipt_sha256"] = freezer.canonical_sha256(changed_body)
    with pytest.raises(RuntimeError, match="exact terminal replay"):
        freezer.validate_receipt(changed)


@pytest.mark.parametrize("failure_point", ["post_link_fsync", "post_link_validation"])
def test_receipt_linked_then_exception_never_publishes_failure(
    monkeypatch, tmp_path: Path, failure_point: str,
):
    paths = {
        "AUTHORITY": tmp_path / "authority.json",
        "ROWS": tmp_path / "rows.pt",
        "MANIFEST": tmp_path / "manifest.json",
        "RECEIPT": tmp_path / "receipt.json",
        "FAILURE": tmp_path / "failure.json",
        "LOCK": tmp_path / "owner.lock",
    }
    for name, path in paths.items():
        monkeypatch.setattr(freezer, name, path)
    monkeypatch.setattr(freezer, "ROLE_SIZES", {"basis": 1})
    authority = {
        "authority_sha256": "a" * 64,
        "source_closure": {"sha256": "s" * 64},
    }
    paths["AUTHORITY"].write_text(json.dumps(authority))
    payload = {
        "selection_plan_sha256": "p" * 64,
        "roles": {"basis": torch.tensor([[1, 2]], dtype=torch.long)},
    }
    manifest = {"manifest_sha256": "m" * 64}
    replay = {
        "authority_file_sha256": freezer.file_sha256(paths["AUTHORITY"]),
        "rows_file_sha256": "r" * 64,
        "manifest_file_sha256": "m" * 64,
        "cache_file_sha256s": {"cache": "c" * 64},
    }
    monkeypatch.setattr(freezer, "validate_authority", lambda _value: None)
    monkeypatch.setattr(freezer, "load_cache_tensors", lambda _authority: {})
    monkeypatch.setattr(freezer, "build_rows_payload", lambda _authority, _caches: payload)
    monkeypatch.setattr(freezer, "build_manifest", lambda _payload, _authority: manifest)
    monkeypatch.setattr(
        freezer, "replay_terminal_state", lambda _authority: (payload, manifest, replay),
    )

    if failure_point == "post_link_validation":
        monkeypatch.setattr(
            freezer, "validate_receipt",
            lambda _receipt: (_ for _ in ()).throw(RuntimeError("synthetic validation failure")),
        )
    else:
        original_fsync = freezer._fsync_directory
        raised = False

        def fail_first_fsync_after_receipt_link(path: Path) -> None:
            nonlocal raised
            if paths["RECEIPT"].exists() and not raised:
                raised = True
                raise OSError("synthetic post-link fsync failure")
            original_fsync(path)

        monkeypatch.setattr(freezer, "_fsync_directory", fail_first_fsync_after_receipt_link)

    with pytest.raises((OSError, RuntimeError), match="synthetic"):
        freezer.materialize()
    assert paths["ROWS"].exists()
    assert paths["MANIFEST"].exists()
    assert paths["RECEIPT"].exists()
    assert json.loads(paths["RECEIPT"].read_text())["failure_absent"] is True
    assert not paths["FAILURE"].exists()
    assert not paths["LOCK"].exists()
