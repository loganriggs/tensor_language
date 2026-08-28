from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import prepare_mlp1_global_gate_rows as rows


def test_prospective_constants_are_one_row_per_new_document() -> None:
    assert rows.START_DOCUMENT_INDEX == 50_000
    assert rows.N_SOURCE_DOCUMENTS == 32
    assert rows.WAVE_DOCUMENTS == 16
    assert rows.MAX_CHUNKS_PER_DOCUMENT == 1
    assert rows.TOKEN_LENGTH == 513


def test_local_summary_validates_the_registered_16_by_16_waves() -> None:
    records = [
        {"document_id": f"doc-{index}", "wave": "A" if index < 16 else "B"}
        for index in range(32)
    ]
    summary = rows.summarize_records(records)
    assert summary["n_source_documents"] == summary["n_chunks"] == 32
    assert summary["waves"]["A"]["n_source_documents"] == 16
    assert summary["waves"]["B"]["n_source_documents"] == 16
    with pytest.raises(RuntimeError, match="balanced-document"):
        rows.summarize_records(records[:-1])


def test_registry_discovery_includes_old_receipts_and_excludes_new_namespace(
    tmp_path, monkeypatch,
) -> None:
    old = tmp_path / "old_rows_receipt.json"
    old.write_text("{}")
    current = tmp_path / "mlp1_global_gate_v1_rows_receipt.json"
    current.write_text("{}")
    canonical = tmp_path / "canonical_receipt.json"
    canonical.write_text("{}")
    monkeypatch.setattr(rows, "BQ", tmp_path)
    monkeypatch.setattr(rows, "RECEIPT", current)
    monkeypatch.setattr(rows.registry.BASE, "CANONICAL_RECEIPT", canonical)
    discovered = rows.discover_prior_registry_files()
    assert old in discovered and canonical in discovered and current not in discovered


def test_load_frozen_rows_enforces_document_uniqueness_and_bytes(tmp_path, monkeypatch) -> None:
    tensor = torch.arange(32 * 513, dtype=torch.long).reshape(32, 513)
    cache = tmp_path / "rows.pt"
    torch.save(tensor, cache)
    receipt = tmp_path / "receipt.json"
    records = [{"document_id": f"doc-{index}"} for index in range(32)]
    receipt.write_text(json.dumps({
        "status": "frozen_before_any_global_gate_model_forward",
        "authorized_for_scored_experiments": True,
        "authorized_for_training": False,
        "disjointness_gates": {"documents": True, "rows": True},
        "entries": {"all": {
            "cache_path": str(cache),
            "tensor_raw_sha256": rows.registry.tensor_sha256(tensor),
            "file_sha256": rows.registry.file_sha256(cache),
        }},
        "document_provenance": {"sets": {"all": records}},
    }))
    monkeypatch.setattr(rows, "RECEIPT", receipt)
    loaded_receipt, loaded = rows.load_frozen_rows()
    assert loaded_receipt["status"] == "frozen_before_any_global_gate_model_forward"
    assert torch.equal(loaded, tensor)

    records[-1] = records[0]
    payload = json.loads(receipt.read_text())
    payload["document_provenance"]["sets"]["all"] = records
    receipt.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="rows changed"):
        rows.load_frozen_rows()


def test_freeze_releases_lock_on_failure(tmp_path, monkeypatch) -> None:
    lock = tmp_path / "freeze.lock"
    monkeypatch.setattr(rows, "LOCK", lock)
    monkeypatch.setattr(
        rows, "freeze_locked",
        lambda: (_ for _ in ()).throw(RuntimeError("deliberate failure")),
    )
    with pytest.raises(RuntimeError, match="deliberate failure"):
        rows.freeze()
    assert not lock.exists()
