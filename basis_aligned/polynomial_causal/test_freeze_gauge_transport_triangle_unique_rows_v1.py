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


def test_strict_receipt_reload_rejects_self_consistent_mutation(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(freezer, "ROLE_SIZES", {"basis": 1})
    authority = {
        "authority_sha256": "a" * 64,
        "source_closure": {"sha256": "s" * 64},
    }
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority))
    monkeypatch.setattr(freezer, "AUTHORITY", authority_path)
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

    changed = copy.deepcopy(receipt)
    changed["triangle_runner_authorized_by_this_receipt"] = True
    changed_body = {key: changed[key] for key in changed if key != "receipt_sha256"}
    changed["receipt_sha256"] = freezer.canonical_sha256(changed_body)
    with pytest.raises(RuntimeError, match="exact terminal replay"):
        freezer.validate_receipt(changed)
