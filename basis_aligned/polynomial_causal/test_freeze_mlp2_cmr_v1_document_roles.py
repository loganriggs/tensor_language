from __future__ import annotations

import hashlib
import json

import pytest

import freeze_mlp2_cmr_v1_document_roles as roles


def test_ordering_digest_and_role_assignment_are_exact_and_disjoint() -> None:
    assert roles.ordering_digest(17) == hashlib.sha256(
        b"bilin18_mlp2_cmr_v1\0" + b"17"
    ).digest()
    assigned = roles.assign_roles(1000, {1, 5, 9})
    assert tuple(assigned) == roles.ROLES
    flattened = [index for role in roles.ROLES for index in assigned[role]]
    assert len(flattened) == len(set(flattened)) == 768
    assert not {1, 5, 9}.intersection(flattened)


def test_assignment_rejects_malformed_or_insufficient_population() -> None:
    with pytest.raises(ValueError, match="outside"):
        roles.assign_roles(1000, {1000})
    with pytest.raises(ValueError, match="nonnegative"):
        roles.ordering_digest(True)
    with pytest.raises(RuntimeError, match="not enough"):
        roles.assign_roles(768, {0})


def test_registry_snapshot_extracts_all_supported_identity_forms(tmp_path, monkeypatch) -> None:
    prior = tmp_path / "prior_receipt.json"
    prior.write_text(json.dumps({
        "source_document_index": 2,
        "ordered_document_indices": [3, 4],
        "document_id": "rev:data/file.parquet:5"
    }))
    ordered = tmp_path / "ordered_receipt.json"
    ordered.write_text("{}")
    monkeypatch.setattr(roles, "ROOT", tmp_path)
    monkeypatch.setattr(roles, "HERE", tmp_path)
    monkeypatch.setattr(roles, "ORDERED_RECEIPT", ordered)
    monkeypatch.setattr(roles, "MANIFEST", tmp_path / "own_manifest.json")
    monkeypatch.setattr(roles, "RECEIPT", tmp_path / "own_receipt.json")
    protocol = {"fineweb_source": {
        "revision": "rev", "relative_path": "data/file.parquet", "parquet_rows": 100
    }}
    exclusions, hashes = roles.registry_snapshot(protocol)
    assert exclusions == {2, 3, 4, 5}
    assert set(hashes) == {str(prior.resolve()), str(ordered.resolve())}


def test_create_only_writer_never_overwrites(tmp_path) -> None:
    path = tmp_path / "result.json"
    roles.write_create_only(path, {"value": 1})
    with pytest.raises(FileExistsError):
        roles.write_create_only(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}
