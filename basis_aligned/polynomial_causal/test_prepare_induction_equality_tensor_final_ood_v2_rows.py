import json
import os
from pathlib import Path

import pytest

import prepare_induction_equality_tensor_final_ood_v2_rows as subject


def test_registry_exclusion_is_metadata_only_and_never_loads_tensors(tmp_path, monkeypatch):
    old_final = tmp_path / "final_natural.pt"
    old_ood = tmp_path / "ood_code.pt"
    registry = tmp_path / "receipt.json"
    registry.write_text(json.dumps({
        "entries": {"final": {"path": str(old_final)}, "ood": {"path": str(old_ood)}},
        "document_provenance": {"sets": {
            "final": [{"document_id": "doc-old", "dataset_document_index": 4}],
            "ood": [{"path": "pkg/old.py", "blob_sha256": "a" * 64,
                     "normalized_python_sha256": "b" * 64}],
        }},
    }))
    monkeypatch.setattr(subject.torch, "load", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("metadata census attempted tensor deserialization")
    ))
    prior, hashes = subject.metadata_registry_snapshot((registry,))
    assert prior["documents"] == {"doc-old"}
    assert prior["indices"] == {4}
    assert prior["code_paths"] == {"pkg/old.py"}
    assert prior["normalized"] == {"b" * 64}
    assert prior["code_sources_missing_normalized"] == set()
    assert {Path(value).name for value in prior["forbidden_v1_role_references"]} == {
        "final_natural.pt", "ood_code.pt",
    }
    assert hashes[str(registry.resolve())] == subject.file_sha256(registry)


def test_new_namespace_never_names_old_role_cache_as_an_input():
    assert subject.CACHE.name == ".rowcache_induction_equality_tensor_final_ood_v2"
    assert set(subject.ROLES) == {"label_fit", "final_natural", "ood_code"}
    source_text = Path(subject.__file__).read_text()
    assert "torch.load(path" in source_text  # only validates newly staged/installed payloads
    assert ".rowcache_terminal_copy_induction_v2/final_natural.pt" not in source_text
    assert ".rowcache_terminal_copy_induction_v2/ood_code.pt" not in source_text


def test_audit_schema_is_exact_and_outcome_blind(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "AUDIT", tmp_path / "audit.json")
    commit, sources = "a" * 40, {"source.py": "b" * 64}
    payload = {
        "schema": "induction_equality_tensor_final_ood_v2_rows_independent_audit",
        "status": "GO", "outcome_access": False, "audited_source_commit": commit,
        "audited_source_hashes": sources, "tests_passed": 3, "reviewer": "independent",
    }
    subject.AUDIT.write_text(json.dumps(payload))
    assert subject.validate_audit(commit, sources) == payload
    payload["outcome_access"] = True
    subject.AUDIT.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="exact source-bound GO"):
        subject.validate_audit(commit, sources)


def test_audited_source_binding_uses_frozen_audit_commit_not_moving_head(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "AUDIT", tmp_path / "audit.json")
    commit, sources = "c" * 40, {"source.py": "d" * 64}
    payload = {"schema": "induction_equality_tensor_final_ood_v2_rows_independent_audit", "status": "GO", "outcome_access": False, "audited_source_commit": commit, "audited_source_hashes": sources, "tests_passed": 1, "reviewer": "reviewer"}
    subject.AUDIT.write_text(json.dumps(payload))
    monkeypatch.setattr(subject, "source_closure", lambda selected: sources if selected == commit else (_ for _ in ()).throw(AssertionError("moving HEAD selected")))
    assert subject.audited_source_binding() == (commit, sources, payload)


def test_preserved_v1_no_go_is_a_direct_source_and_exact_parent():
    assert subject.V1_AUDIT in subject.SOURCE_PATHS
    assert subject.file_sha256(subject.V1_AUDIT) == subject.V1_AUDIT_SHA256
    audit = json.loads(subject.V1_AUDIT.read_text())
    assert audit["approved"] is False and audit["outcome_access"] is False


def test_missing_normalized_hash_is_recovered_from_historical_bound_blob(
    tmp_path, monkeypatch,
):
    blob = b"x = 123  # formatting and literal are normalized\n"
    blob_sha = subject.hashlib.sha256(blob).hexdigest()
    registry = tmp_path / "receipt.json"
    registry.write_text(json.dumps({
        "source_commit": "a" * 40,
        "files": [{"path": "old/name.py", "blob_sha256": blob_sha}],
    }))
    prior, _ = subject.metadata_registry_snapshot((registry,))
    assert prior["normalized"] == set()
    assert prior["code_sources_missing_normalized"] == {
        ("a" * 40, "old/name.py", blob_sha),
    }

    class Completed:
        returncode = 0
        stdout = blob

    monkeypatch.setattr(subject.subprocess, "run", lambda *_a, **_k: Completed())
    assert subject.recover_prior_normalized_hashes(
        prior["code_sources_missing_normalized"],
    ) == {subject.base.normalized_python_sha256(blob)}


def test_historical_blob_recovery_fails_on_hash_mismatch(monkeypatch):
    class Completed:
        returncode = 0
        stdout = b"different bytes"

    monkeypatch.setattr(subject.subprocess, "run", lambda *_a, **_k: Completed())
    with pytest.raises(RuntimeError, match="cannot be recovered exactly"):
        subject.recover_prior_normalized_hashes({
            ("a" * 40, "old/name.py", "b" * 64),
        })


def test_support_census_binds_off_target_and_all_and_fails_closed():
    masks = {
        name: subject.torch.zeros(32, 256, dtype=subject.torch.bool)
        for name in ("positive", "matched_negative", "off_target", "all")
    }
    masks["positive"][:, 64:72] = True
    masks["matched_negative"][:, 72:80] = True
    masks["off_target"][:, 80:256] = True
    masks["all"][:, 64:256] = True
    census = subject.scored_support_census(masks)
    assert census["positive"] == {"tokens": 256, "documents": 32}
    assert census["off_target"] == {"tokens": 5632, "documents": 32}
    assert census["all"] == {"tokens": 6144, "documents": 32}
    masks["off_target"].zero_()
    with pytest.raises(RuntimeError, match="support is below"):
        subject.scored_support_census(masks)


def test_linked_receipt_survives_directory_fsync_error_without_failure_eligibility(
    tmp_path, monkeypatch,
):
    receipt, failure = tmp_path / "receipt.json", tmp_path / "failure.json"
    monkeypatch.setattr(subject, "RECEIPT", receipt)
    monkeypatch.setattr(subject, "FAILURE", failure)
    real_fsync = os.fsync
    calls = {"n": 0}

    def fail_second_fsync(descriptor):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("injected directory fsync failure after link")
        return real_fsync(descriptor)

    monkeypatch.setattr(subject.os, "fsync", fail_second_fsync)
    with pytest.raises(OSError, match="after link"):
        subject.write_receipt_create_only(
            {"schema": "known", "value": 3}, receipt,
            pre_link_check=lambda: None,
        )
    assert json.loads(receipt.read_text()) == {"schema": "known", "value": 3}
    assert not subject.failure_is_still_publishable()
    assert not failure.exists()
