from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import prepare_mlp1_empirical_moment_tensor_v1_rows as freezer


def minimal_protocol() -> dict[str, object]:
    return {
        "fineweb_source": {
            "revision": "rev",
            "relative_path": "data/file.parquet",
            "parquet_rows": 100,
        }
    }


def test_protocol_binds_parents_and_exact_registered_selection() -> None:
    value = freezer.load_protocol()
    assert value["experiment_id"] == freezer.role_manifest.EXPERIMENT_ID
    assert value["selection"]["role_order"] == ["FIT", "VALIDATION", "REPLICATION"]
    assert value["selection"]["documents_per_role"] == 2_084
    assert value["selection"]["fit_prefix_rows"] == {
        "FIT100": 100_000, "FIT200": 200_000, "FIT400": 400_000,
    }
    assert value["publication"]["rule"].startswith("manifest_create_only_then_receipt_last")


def test_recursive_registry_discovery_includes_worktree_and_excludes_own_outputs(
    tmp_path, monkeypatch,
) -> None:
    nested = tmp_path / "nested" / "deeper"
    nested.mkdir(parents=True)
    prior = nested / "old_rows_receipt.json"
    prior.write_text("{}")
    unrelated = nested / "scientific_results.json"
    unrelated.write_text("{}")
    own_manifest = tmp_path / "mlp1_empirical_moment_tensor_v1_row_roles_manifest.json"
    own_manifest.write_text("{}")
    canonical = tmp_path / "fineweb_authority.json"
    canonical.write_text("{}")
    monkeypatch.setattr(freezer, "ORDERED_FINEWEB_RECEIPT", canonical)
    monkeypatch.setattr(
        freezer, "EXCLUDED_OUTPUTS", frozenset((own_manifest.resolve(),)),
    )
    found = freezer.discover_registry_files(tmp_path)
    assert prior.resolve() in found and canonical.resolve() in found
    assert unrelated.resolve() not in found and own_manifest.resolve() not in found


def test_registry_exclusions_parse_indices_lists_and_canonical_document_ids(tmp_path) -> None:
    first = tmp_path / "one_receipt.json"
    first.write_text(json.dumps({
        "document_provenance": {"sets": {"x": [
            {"dataset_document_index": 7},
            {"document_id": "rev:data/file.parquet:9"},
        ]}},
        "ordered_document_indices": [11, 13],
    }))
    second = tmp_path / "two_manifest.json"
    second.write_text(json.dumps({"source_document_index": 17}))
    indices, hashes = freezer.load_registry_exclusions(
        (first.resolve(), second.resolve()), minimal_protocol(),
    )
    assert indices == frozenset({7, 9, 11, 13, 17})
    assert set(hashes) == {str(first.resolve()), str(second.resolve())}


@pytest.mark.parametrize("payload,match", [
    ({"dataset_document_index": True}, "literal nonnegative"),
    ({"ordered_document_indices": "not-a-list"}, "not an index list"),
    ({"document_id": 5}, "not a string"),
    ({"document_id": "rev:data/file.parquet:09"}, "malformed index"),
    ({"dataset_document_index": 100}, "exceeds pinned parquet"),
])
def test_registry_exclusions_fail_closed_on_malformed_identity(
    tmp_path, payload, match,
) -> None:
    path = tmp_path / "bad_receipt.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match=match):
        freezer.load_registry_exclusions((path,), minimal_protocol())


def test_registry_json_toctou_is_rejected(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rows_receipt.json"
    path.write_text("{}")
    calls = iter(("0" * 64, "1" * 64))
    monkeypatch.setattr(freezer, "file_sha256", lambda ignored: next(calls))
    with pytest.raises(RuntimeError, match="changed while reading"):
        freezer.load_registry_exclusions((path,), minimal_protocol())


def test_atomic_json_publication_is_create_only(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    freezer.write_json_create_only({"value": 1}, path)
    assert json.loads(path.read_text()) == {"value": 1}
    with pytest.raises(FileExistsError):
        freezer.write_json_create_only({"value": 2}, path)
    assert json.loads(path.read_text()) == {"value": 1}


def test_lock_claim_detects_replacement_and_release_preserves_foreign_lock(
    tmp_path, monkeypatch,
) -> None:
    lock = tmp_path / "roles.lock"
    claim = freezer.acquire_lock(lock)
    freezer.require_run_claim(claim)
    lock.unlink()
    lock.write_text(json.dumps({"pid": os.getpid(), "nonce": claim.nonce}))
    with pytest.raises(RuntimeError, match="ownership changed"):
        freezer.require_run_claim(claim)
    freezer.release_lock(claim)
    assert lock.exists()


def _patch_transaction(tmp_path, monkeypatch, *, fail_final=False):
    manifest = tmp_path / "role_manifest.json"
    receipt = tmp_path / "role_receipt.json"
    lock = tmp_path / "role.lock"
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}")
    monkeypatch.setattr(freezer, "ROOT", tmp_path)
    monkeypatch.setattr(freezer, "PROTOCOL", protocol_path)
    monkeypatch.setattr(freezer, "MANIFEST", manifest)
    monkeypatch.setattr(freezer, "RECEIPT", receipt)
    monkeypatch.setattr(freezer, "LOCK", lock)
    monkeypatch.setattr(freezer, "load_protocol", lambda: minimal_protocol())
    monkeypatch.setattr(freezer, "source_identity", lambda: ("c" * 40, {"source": "a" * 64}))
    monkeypatch.setattr(freezer, "validate_parquet_identity", lambda protocol: {
        "path": "/p", "size": 1, "sha256": "b" * 64,
        "parquet_rows": 7_000, "parquet_row_groups": 1,
    })
    registry_file = tmp_path / "prior_receipt.json"
    registry_file.write_text("{}")
    monkeypatch.setattr(freezer, "discover_registry_files", lambda: (registry_file,))
    monkeypatch.setattr(
        freezer, "load_registry_exclusions",
        lambda files, protocol: (frozenset({1}), {str(registry_file): "d" * 64}),
    )
    monkeypatch.setattr(freezer, "file_sha256", lambda path: "e" * 64)
    monkeypatch.setattr(freezer.role_manifest, "build_role_manifest", lambda **kwargs: {
        "schema_version": 1, "authority": "none", "roles": {},
    })
    calls = []
    def verify(**kwargs):
        calls.append(kwargs["require_manifest"])
        freezer.require_run_claim(kwargs["claim"])
        if fail_final and kwargs["require_manifest"]:
            raise RuntimeError("injected final guard failure")
    monkeypatch.setattr(freezer, "verify_snapshot", verify)
    return manifest, receipt, calls


def test_manifest_precedes_last_written_receipt(tmp_path, monkeypatch) -> None:
    manifest, receipt, calls = _patch_transaction(tmp_path, monkeypatch)
    result = freezer.freeze()
    assert calls == [False, True]
    assert manifest.is_file() and receipt.is_file()
    assert result["manifest_sha256"] == "e" * 64
    assert result["authority"] == "document_role_identity_only"
    assert result["authorized_for_document_role_identity"] is True
    assert all(result[key] is False for key in (
        "authorized_for_tokenization", "authorized_for_activation_capture",
        "authorized_for_model_forward", "authorized_for_fit_or_validation",
        "authorized_for_scientific_outcomes",
    ))


def test_final_guard_failure_preserves_manifest_without_receipt(tmp_path, monkeypatch) -> None:
    manifest, receipt, calls = _patch_transaction(tmp_path, monkeypatch, fail_final=True)
    with pytest.raises(RuntimeError, match="injected final guard failure"):
        freezer.freeze()
    assert calls == [False, True]
    assert manifest.is_file() and not receipt.exists()
    assert not (tmp_path / "role.lock").exists()


def test_freezer_source_has_no_token_tensor_checkpoint_or_model_capability() -> None:
    source = Path(freezer.__file__).read_text()
    assert "import torch" not in source
    assert "import tiktoken" not in source
    assert "transformers" not in source
    assert "iter_batches" not in source
    assert "read_table" not in source
    assert "torch.load" not in source
