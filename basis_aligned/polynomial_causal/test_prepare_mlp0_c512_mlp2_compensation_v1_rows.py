import json
import os
from pathlib import Path

import pytest
import torch

import prepare_mlp0_c512_mlp2_compensation_v1_rows as rows


def test_registry_census_extracts_documents_indices_and_row_tensors(tmp_path, monkeypatch):
    tensor_path = tmp_path / "fineweb_rows.pt"
    tensor = torch.arange(2 * 40, dtype=torch.long).reshape(2, 40)
    torch.save(tensor, tensor_path)
    reference_path = tmp_path / "eval_tokens.pt"
    torch.save(torch.arange(40, dtype=torch.long).reshape(1, 40) + 100, reference_path)
    receipt = tmp_path / "role_rows_receipt.json"
    receipt.write_text(json.dumps({
        "entries": {"eval": {"cache_path": str(tensor_path)}},
        "document_provenance": {"sets": {"eval": [
            {"document_id": "doc-a", "dataset_document_index": 17}
        ]}},
    }))
    monkeypatch.setattr(rows, "REFERENCE_ROWS", reference_path)
    prior, registry_hashes, tensor_hashes = rows.load_registry_exclusions((receipt,))
    assert prior[0] == {"doc-a"} and prior[1] == {17}
    assert tuple(tensor[0].tolist()) in prior[2]
    assert tuple(tensor[0, :32].tolist()) in prior[3]
    assert str(receipt.resolve()) in registry_hashes
    assert {str(tensor_path.resolve()), str(reference_path.resolve())} == set(tensor_hashes)


def test_registry_census_fails_on_missing_registered_row_tensor(tmp_path):
    receipt = tmp_path / "role_manifest.json"
    receipt.write_text(json.dumps({"row_path": str(tmp_path / "missing_fineweb_rows.pt")}))
    with pytest.raises(RuntimeError, match="row-like tensor is missing"):
        rows.load_registry_exclusions((receipt,))


def test_long_row_tensor_filter_rejects_float_programs():
    assert list(rows.long_row_tensors(torch.zeros(3, 40))) == []
    assert len(list(rows.long_row_tensors({"rows": torch.zeros(3, 40, dtype=torch.long)}))) == 1


def test_registry_discovery_is_recursive(tmp_path, monkeypatch):
    registry_root = tmp_path / "registry"
    nested = registry_root / ".rowcache_shadow" / "fineweb_local_shadow_v1_receipt.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}")
    canonical = tmp_path / "canonical_receipt.json"
    canonical.write_text("{}")
    monkeypatch.setattr(rows, "BQ", registry_root)
    monkeypatch.setattr(rows, "RECEIPT", registry_root / "current_rows_receipt.json")
    monkeypatch.setattr(rows.BASE, "CANONICAL_RECEIPT", canonical)
    discovered = rows.discover_prior_registry_files()
    assert nested in discovered
    assert canonical in discovered


def test_declared_file_and_tensor_digests_are_enforced(tmp_path, monkeypatch):
    tensor_path = tmp_path / "fineweb_rows.pt"
    tensor = torch.arange(2 * 300, dtype=torch.long).reshape(2, 300)
    torch.save(tensor, tensor_path)
    receipt = tmp_path / "rows_receipt.json"
    receipt.write_text(json.dumps({"entry": {
        "cache_path": str(tensor_path),
        "file_sha256": rows.file_sha256(tensor_path),
        "tensor_full_raw_sha256": rows.tensor_sha256(tensor),
        "tensor_prefix257_raw_sha256": rows.tensor_sha256(tensor[:, :257]),
    }}))
    monkeypatch.setattr(rows, "REFERENCE_ROWS", tensor_path)
    rows.load_registry_exclusions((receipt,))

    payload = json.loads(receipt.read_text())
    payload["entry"]["file_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="declared file_sha256 mismatch"):
        rows.load_registry_exclusions((receipt,))

    payload["entry"]["file_sha256"] = rows.file_sha256(tensor_path)
    payload["entry"]["tensor_full_raw_sha256"] = "1" * 64
    receipt.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="declared tensor_full_raw_sha256 mismatch"):
        rows.load_registry_exclusions((receipt,))


def test_invalid_declared_digest_fails_closed(tmp_path):
    tensor_path = tmp_path / "fineweb_rows.pt"
    torch.save(torch.arange(40, dtype=torch.long).reshape(1, 40), tensor_path)
    with pytest.raises(RuntimeError, match="invalid declared SHA-256"):
        rows.referenced_row_specs({
            "cache_path": str(tensor_path), "tensor_raw_sha256": "not-a-digest",
        })


@pytest.mark.parametrize("schema", ["flat_corpus", "nested_corpus", "final_cache"])
def test_real_manifest_file_digest_schemas_are_enforced(tmp_path, schema):
    tensor_path = tmp_path / "curated_rows.pt"
    torch.save(torch.arange(40, dtype=torch.long).reshape(1, 40), tensor_path)
    digest = rows.file_sha256(tensor_path)
    if schema == "flat_corpus":
        payload = {"corpus_path": str(tensor_path), "corpus_sha256": digest}
        digest_key = "corpus_sha256"
    elif schema == "nested_corpus":
        payload = {"corpus": {"path": str(tensor_path), "sha256": digest}}
        digest_key = "sha256"
    else:
        payload = {
            "final_cache_path": str(tensor_path), "final_cache_sha256": digest,
        }
        digest_key = "final_cache_sha256"
    specifications = rows.referenced_row_specs(payload)
    rows.load_verified_row_tensor(tensor_path, specifications[tensor_path.resolve()])

    if schema == "nested_corpus":
        payload["corpus"][digest_key] = "0" * 64
    else:
        payload[digest_key] = "0" * 64
    specifications = rows.referenced_row_specs(payload)
    with pytest.raises(RuntimeError, match=f"declared {digest_key} mismatch"):
        rows.load_verified_row_tensor(tensor_path, specifications[tensor_path.resolve()])


def test_schema_paired_file_digest_cannot_be_half_declared(tmp_path):
    tensor_path = tmp_path / "curated_rows.pt"
    torch.save(torch.arange(40, dtype=torch.long).reshape(1, 40), tensor_path)
    with pytest.raises(RuntimeError, match="incomplete declared row-file pair"):
        rows.referenced_row_specs({"corpus_path": str(tensor_path)})


def test_tensor_change_during_load_fails_closed(tmp_path, monkeypatch):
    tensor_path = tmp_path / "fineweb_rows.pt"
    torch.save(torch.arange(40, dtype=torch.long).reshape(1, 40), tensor_path)
    real_load = torch.load

    def load_then_mutate(path, *args, **kwargs):
        payload = real_load(path, *args, **kwargs)
        with Path(path).open("ab") as handle:
            handle.write(b"changed-after-load")
        return payload

    monkeypatch.setattr(rows.torch, "load", load_then_mutate)
    with pytest.raises(RuntimeError, match="changed while loading"):
        rows.load_verified_row_tensor(tensor_path, [])


def test_create_only_receipt_never_overwrites_existing_or_late_competitor(tmp_path, monkeypatch):
    receipt = tmp_path / "authority.json"
    receipt.write_text("existing-authority")
    with pytest.raises(FileExistsError):
        rows.write_json_create_only({"authorized": True}, receipt)
    assert receipt.read_text() == "existing-authority"

    receipt.unlink()
    real_link = os.link

    def competitor_wins(source, destination):
        Path(destination).write_text("competitor-authority")
        return real_link(source, destination)

    monkeypatch.setattr(rows.os, "link", competitor_wins)
    with pytest.raises(FileExistsError):
        rows.write_json_create_only({"authorized": True}, receipt)
    assert receipt.read_text() == "competitor-authority"


def test_lock_and_cache_namespace_are_create_only(tmp_path):
    lock = tmp_path / "freeze.lock"
    descriptor = rows.acquire_lock(lock)
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            rows.acquire_lock(lock)
    finally:
        rows.release_lock(descriptor, lock)

    staging = tmp_path / "staging"
    staging.mkdir()
    staged = staging / "eval.pt"
    staged.write_bytes(b"new")
    cache = tmp_path / "cache"
    cache.mkdir()
    sentinel = cache / "eval.pt"
    sentinel.write_bytes(b"competitor")
    with pytest.raises(FileExistsError):
        rows.install_cache_create_only(staging, cache)
    assert sentinel.read_bytes() == b"competitor"
    assert staged.read_bytes() == b"new"


def test_snapshot_revalidation_detects_registry_membership_drift(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("source")
    registry_a = tmp_path / "a_receipt.json"
    registry_b = tmp_path / "b_receipt.json"
    registry_a.write_text("{}")
    registry_b.write_text("{}")
    monkeypatch.setattr(rows, "ROOT", tmp_path)
    monkeypatch.setattr(rows, "git", lambda *args: "commit-a")
    monkeypatch.setattr(rows, "require_committed_source", lambda path, commit: None)
    monkeypatch.setattr(rows, "discover_prior_registry_files", lambda: (registry_a, registry_b))
    with pytest.raises(RuntimeError, match="registry membership changed"):
        rows.verify_frozen_snapshot(
            source_commit="commit-a",
            source_closure=(source,),
            implementation_hashes={"source.py": rows.file_sha256(source)},
            registry_files=(registry_a,),
            registry_hashes={str(registry_a.resolve()): rows.file_sha256(registry_a)},
            prior_tensor_hashes={},
            prior=(set(), set(), set(), set()),
            source=source,
        )


def test_snapshot_revalidation_detects_source_drift(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("changed")
    monkeypatch.setattr(rows, "ROOT", tmp_path)
    monkeypatch.setattr(rows, "git", lambda *args: "commit-a")
    with pytest.raises(RuntimeError, match="implementation source changed"):
        rows.verify_frozen_snapshot(
            source_commit="commit-a",
            source_closure=(source,),
            implementation_hashes={"source.py": "0" * 64},
            registry_files=(),
            registry_hashes={},
            prior_tensor_hashes={},
            prior=(set(), set(), set(), set()),
            source=source,
        )


@pytest.mark.parametrize(
    ("changed_registry", "changed_tensors", "message"),
    [
        (True, False, "registry contents changed"),
        (False, True, "row tensor contents changed"),
    ],
)
def test_snapshot_revalidation_detects_content_drift(
    tmp_path, monkeypatch, changed_registry, changed_tensors, message,
):
    source = tmp_path / "source.py"
    source.write_text("source")
    registry = tmp_path / "rows_receipt.json"
    registry.write_text("{}")
    registry_key = str(registry.resolve())
    prior = (set(), set(), set(), set())
    expected_registry = {registry_key: "a" * 64}
    expected_tensors = {"/rows.pt": "b" * 64}
    observed_registry = {registry_key: ("c" if changed_registry else "a") * 64}
    observed_tensors = {"/rows.pt": ("d" if changed_tensors else "b") * 64}
    monkeypatch.setattr(rows, "ROOT", tmp_path)
    monkeypatch.setattr(rows, "git", lambda *args: "commit-a")
    monkeypatch.setattr(rows, "require_committed_source", lambda path, commit: None)
    monkeypatch.setattr(rows, "discover_prior_registry_files", lambda: (registry,))
    monkeypatch.setattr(
        rows, "load_registry_exclusions",
        lambda files: (prior, observed_registry, observed_tensors),
    )
    with pytest.raises(RuntimeError, match=message):
        rows.verify_frozen_snapshot(
            source_commit="commit-a",
            source_closure=(source,),
            implementation_hashes={"source.py": rows.file_sha256(source)},
            registry_files=(registry,),
            registry_hashes=expected_registry,
            prior_tensor_hashes=expected_tensors,
            prior=prior,
            source=source,
        )


def test_freeze_releases_namespace_lock_on_failure(tmp_path, monkeypatch):
    lock = tmp_path / "freeze.lock"
    monkeypatch.setattr(rows, "LOCK", lock)
    monkeypatch.setattr(
        rows, "freeze_locked",
        lambda: (_ for _ in ()).throw(RuntimeError("deliberate failure")),
    )
    with pytest.raises(RuntimeError, match="deliberate failure"):
        rows.freeze()
    assert not lock.exists()
