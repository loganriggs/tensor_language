import json
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
