import importlib.util
import json
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).with_name("prepare_fineweb_oracle_rows.py")
SPEC = importlib.util.spec_from_file_location("prepare_fineweb_oracle_rows", PATH)
PREP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREP)


def test_receipt_content_addresses_every_required_row_set(monkeypatch, tmp_path):
    monkeypatch.setattr(PREP.rowcache, "CACHE", str(tmp_path / "cache"))
    rows = {}
    for n, skip in PREP.SPECS:
        tensor = torch.arange(n * PREP.rowcache.T_LEN, dtype=torch.long).view(
            n, PREP.rowcache.T_LEN
        ) + skip
        rows[(n, skip)] = tensor
        path = Path(PREP.rowcache._path(n, skip))
        path.parent.mkdir(parents=True, exist_ok=True)
        PREP.rowcache._save_atomic(tensor, str(path))
    receipt = PREP.build_receipt(rows)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))

    loaded_receipt, loaded = PREP.validate_receipt(path)
    assert loaded_receipt == receipt
    for spec in PREP.SPECS:
        assert torch.equal(loaded[spec], rows[spec])

    target = PREP.SPECS[-1]
    tampered = rows[target].clone()
    tampered[0, 0] += 1
    PREP.rowcache._save_atomic(tampered, PREP.rowcache._path(*target))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        PREP.validate_receipt(path)


def test_receipt_requires_real_stream_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(PREP.rowcache, "CACHE", str(tmp_path / "cache"))
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"real_stream_bit_identity_gate": {"passed": False}}))
    with pytest.raises(RuntimeError, match="identity gate"):
        PREP.validate_receipt(path)


def test_ordered_manifest_receipt_is_a_second_accepted_identity_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(PREP.rowcache, "CACHE", str(tmp_path / "cache"))
    source = tmp_path / "source.parquet"
    source.write_bytes(b"pinned")
    monkeypatch.setattr(PREP, "PINNED_LOCAL_FILE_SIZE", source.stat().st_size)
    monkeypatch.setattr(PREP, "PINNED_LOCAL_FILE_SHA256", PREP.file_sha256(source))
    rows = {}
    entries = {}
    provenance = {}
    for n, skip in PREP.SPECS:
        tensor = torch.arange(n * PREP.rowcache.T_LEN, dtype=torch.long).view(
            n, PREP.rowcache.T_LEN
        ) + skip
        rows[(n, skip)] = tensor
        cache_path = Path(PREP.rowcache._path(n, skip))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        PREP.rowcache._save_atomic(tensor, str(cache_path))
        key = PREP.spec_key(n, skip)
        entries[key] = {
            "n": n, "skip": skip, "tensor_raw_sha256": PREP.tensor_sha256(tensor),
            "cache_path": str(cache_path),
        }
        provenance[key] = [{"dataset_document_index": skip} for _ in range(n)]
    receipt = {
        "receipt_kind": PREP.ORDERED_MANIFEST_RECEIPT_KIND,
        "authority": "pinned_local_ordered_manifest",
        "authorized_for_scored_experiments": True,
        "ordered_manifest_local_parquet_identity_gate": {
            "passed": True,
            "revision": PREP.PINNED_REVISION,
            "config": "default",
            "first_relative_path": PREP.PINNED_FIRST_FILE,
            "source_local_path": str(source),
            "source_size": source.stat().st_size,
            "source_sha256": PREP.file_sha256(source),
            "ordered_file_count": 2,
            "ordered_manifest_sha256": "a" * 64,
        },
        "rowcache_source_sha256": PREP.hashlib.sha256(
            (PREP.BQ / "rowcache.py").read_bytes()
        ).hexdigest(),
        "entries": entries,
        "document_provenance": {"schema_version": 1, "sets": provenance},
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    _, loaded = PREP.validate_receipt(path)
    assert all(torch.equal(rows[spec], loaded[spec]) for spec in PREP.SPECS)
