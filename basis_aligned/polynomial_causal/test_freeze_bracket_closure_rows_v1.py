from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import freeze_bracket_closure_rows_v1 as freezer
from test_bracket_closure_rows_v1 import REGISTRY, _pool


def _candidate(path: Path):
    rows, records = _pool()
    registry = {
        "families": [{"name": family.name, "opener_ids": list(family.opener_ids),
                      "closer_ids": list(family.closer_ids)} for family in REGISTRY.families],
        "quote_control_ids": list(REGISTRY.quote_control_ids),
        "punctuation_control_ids": list(REGISTRY.punctuation_control_ids),
    }
    source_identity = {
        "tokenizer_name": "toy", "tokenizer_sha256": "1" * 64,
        "prose_source": "toy-prose", "prose_revision": "rev",
        "prose_blob_sha256": "2" * 64, "prose_license": "test",
        "code_repository": "toy-code", "code_commit": "3" * 40,
        "code_license": "test", "builder_source_commit": "4" * 40,
        "builder_source_hashes": {"builder.py": "5" * 64},
    }
    payload = {
        "schema": "bracket_closure_rows_v1_candidates", "rows": rows,
        "records": [{
            "document_id": record.document_id,
            "source_document_index": record.source_document_index,
            "source_file": record.source_file,
            "source_revision": record.source_revision,
            "source_blob_sha256": record.source_blob_sha256,
            "domain": record.domain.value, "license_id": record.license_id,
            "normalized_python_sha256": record.normalized_python_sha256,
        } for record in records],
        "delimiter_registry": registry, "source_identity": source_identity,
    }
    torch.save(payload, path)
    source_hash = hashlib.sha256(json.dumps(
        source_identity, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return source_hash


def _write(path: Path, value) -> str:
    path.write_text(json.dumps(value, sort_keys=True))
    return freezer.file_sha256(path)


def test_authority_and_audit_are_external_exact_and_outcome_blind(monkeypatch):
    sources = {name: "b" * 64 for name in freezer.SOURCE_CLOSURE}
    monkeypatch.setattr(freezer, "source_closure", lambda commit: sources)
    authority = {
        "schema": "bracket_closure_rows_v1_authority", "source_commit": "a" * 40,
        "source_hashes": sources, "candidate_path": "/tmp/candidate.pt",
        "candidate_sha256": "c" * 64, "candidate_source_identity_sha256": "d" * 64,
        "delimiter_registry_sha256": "e" * 64,
        "historical_registries": [{"path": "/tmp/history.json", "sha256": "f" * 64}],
        "allocation_seed": "fixed", "cache_path": "/tmp/cache",
        "receipt_path": "/tmp/receipt", "failure_path": "/tmp/failure",
        "lock_path": "/tmp/lock", "outcome_access": False,
    }
    assert freezer.validate_authority(authority) == authority
    audit = {
        "schema": "bracket_closure_rows_v1_independent_audit", "status": "GO",
        "outcome_access": False, "authority_sha256": "1" * 64,
        "audited_source_commit": "a" * 40, "audited_source_hashes": sources,
        "tests_passed": 1, "reviewer": "independent",
    }
    freezer.validate_independent_audit(
        audit, authority_sha256="1" * 64, authority=authority,
    )
    audit["outcome_access"] = True
    with pytest.raises(RuntimeError, match="source-bound GO"):
        freezer.validate_independent_audit(
            audit, authority_sha256="1" * 64, authority=authority,
        )


def test_receipt_is_not_linked_when_final_guard_fails(tmp_path):
    target = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError, match="injected"):
        freezer._publish_json_last(
            {"schema": "known"}, target,
            lambda: (_ for _ in ()).throw(RuntimeError("injected final guard")),
        )
    assert not target.exists()
    assert not tuple(tmp_path.glob(".*.tmp.*"))


def test_lock_inode_nonce_swap_is_detected(tmp_path):
    lock = tmp_path / "lock"
    claim = freezer._claim(lock)
    lock.unlink(); lock.write_text(claim[2])
    with pytest.raises(RuntimeError, match="ownership changed"):
        freezer._require_claim(lock, claim)


def test_complete_model_free_transaction_publishes_role_files_then_receipt(
    tmp_path, monkeypatch,
):
    candidate = tmp_path / "candidates.pt"
    source_identity_sha = _candidate(candidate)
    history = tmp_path / "history.json"
    history_sha = _write(history, {
        "records": [{"document_id": "old-doc", "row_sha256": "9" * 64,
                     "prefix32_sha256": "8" * 64, "source_file": "old.py",
                     "normalized_python_sha256": "7" * 64}],
    })
    sources = {name: "2" * 64 for name in freezer.SOURCE_CLOSURE}
    monkeypatch.setattr(freezer, "source_closure", lambda commit: sources)
    authority_path, audit_path = tmp_path / "authority.json", tmp_path / "audit.json"
    authority = {
        "schema": "bracket_closure_rows_v1_authority", "source_commit": "1" * 40,
        "source_hashes": sources, "candidate_path": str(candidate.resolve()),
        "candidate_sha256": freezer.file_sha256(candidate),
        "candidate_source_identity_sha256": source_identity_sha,
        "delimiter_registry_sha256": __import__("bracket_closure_rows_v1").registry_sha256(REGISTRY),
        "historical_registries": [{"path": str(history.resolve()), "sha256": history_sha}],
        "allocation_seed": "fixed-v1", "cache_path": str((tmp_path / "cache").resolve()),
        "receipt_path": str((tmp_path / "receipt.json").resolve()),
        "failure_path": str((tmp_path / "failure.json").resolve()),
        "lock_path": str((tmp_path / "run.lock").resolve()), "outcome_access": False,
    }
    authority_sha = _write(authority_path, authority)
    audit = {
        "schema": "bracket_closure_rows_v1_independent_audit", "status": "GO",
        "outcome_access": False, "authority_sha256": authority_sha,
        "audited_source_commit": authority["source_commit"],
        "audited_source_hashes": sources, "tests_passed": 7, "reviewer": "reviewer",
    }
    _write(audit_path, audit)
    result = freezer.freeze(authority_path, audit_path)
    assert result["status"] == "frozen_before_any_model_forward_receipt_last"
    assert result["outcome_access"] is False
    assert (tmp_path / "receipt.json").is_file()
    assert not (tmp_path / "failure.json").exists()
    assert sorted(path.name for path in (tmp_path / "cache").iterdir()) == [
        "fit.pt", "ood.pt", "select.pt",
    ]
    forged = torch.load(tmp_path / "cache/fit.pt", map_location="cpu", weights_only=True)
    forged["masks"]["compatible_closer"] = torch.zeros_like(
        forged["masks"]["compatible_closer"],
    )
    forged_path = tmp_path / "forged.pt"
    torch.save(forged, forged_path)
    with pytest.raises(RuntimeError, match="do not replay"):
        freezer._payload_summary(
            forged_path, freezer.file_sha256(forged_path), REGISTRY,
        )


def test_module_source_has_no_model_or_self_audit_mint() -> None:
    source = Path(freezer.__file__).read_text()
    assert "tt_model" not in source and "load_model" not in source
    assert "write_independent_audit" not in source
    assert "outcome_access\": False" in source
    assert freezer.SOURCE_CLOSURE[-3:] == (
        "basis_aligned/polynomial_causal/test_bracket_closure_masks_v1.py",
        "basis_aligned/polynomial_causal/test_bracket_closure_rows_v1.py",
        "basis_aligned/polynomial_causal/test_freeze_bracket_closure_rows_v1.py",
    )
