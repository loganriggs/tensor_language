from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import newline_l12h6_canary_rows_v1 as contract
import newline_l12h6_token_registry_v1 as token_registry
import prepare_newline_l12h6_canary_v1_rows as subject


def _fake_encode(text: str) -> list[int]:
    return [198 if character == "\n" else 1_000 + ord(character) for character in text]


def test_domain_classifier_and_enumerator_are_role_licensed_and_disjoint(monkeypatch) -> None:
    quotas = {role: {domain: 1 for domain in contract.DOMAIN_ORDER} for role in contract.ROLE_ORDER}
    monkeypatch.setattr(contract, "ROLE_DOMAIN_QUOTAS", quotas)
    monkeypatch.setattr(subject, "EXTRA_CANDIDATES_PER_CELL", 0)
    natural = []
    for index in range(300):
        natural.append((index, f"prose-{index}", f"prose-{index}\n" + "x" * 600))
        natural.append((10_000 + index, f"list-{index}", (
            f"- item {index}\n- second\n- third\n- fourth\n" + "x" * 600
        )))
    code = [
        (index, f"pkg/module_{index}.py", f"value_{index} = 1\n".encode() + b"x" * 600)
        for index in range(300)
    ]
    import tiktoken
    registry = token_registry.build_registry(tiktoken.get_encoding("gpt2"))
    rows, records = subject.enumerate_from_sources(
        natural, code, _fake_encode, registry, contract.HistoricalExclusions.empty(),
        code_revision="a" * 40,
    )
    assert tuple(sorted((record.role_license, record.domain.value) for record in records)) == tuple(
        sorted((role, domain) for role in contract.ROLE_ORDER for domain in contract.DOMAIN_ORDER)
    )
    assert all(record.normalized_python_sha256 for record in records if record.domain.value == "code")
    roles = contract.allocate_roles(
        rows, records, subject.frozen_mask_spec(registry), contract.HistoricalExclusions.empty(),
        seed=subject.ALLOCATION_SEED,
    )
    contract.validate_role_disjointness(roles)
    assert subject.is_list_table("- a\n- b\n- c\n- d")
    assert not subject.is_list_table("one paragraph\nwith a continuation")


def test_recursive_registry_replay_is_metadata_only(monkeypatch, tmp_path) -> None:
    registry = tmp_path / "prior.json"
    registry.write_text(json.dumps({"nested": [{
        "document_id": "doc", "source_file": "file", "source_blob_sha256": "a" * 64,
        "normalized_python_sha256": "b" * 64, "row_sha256": "c" * 64,
        "prefix32_sha256": "d" * 64,
    }]}))
    monkeypatch.setattr(subject, "discover_registry_files", lambda: (registry.resolve(),))
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("registry replay deserialized a tensor")
    ))
    snapshot = subject.registry_snapshot()
    excluded = subject.historical_exclusions(snapshot)
    assert excluded.document_ids == frozenset({"doc"})
    assert excluded.source_files == frozenset({"file"})
    assert excluded.source_blobs == frozenset({"a" * 64})
    assert excluded.normalized_python_sha256s == frozenset({"b" * 64})
    assert excluded.row_sha256s == frozenset({"c" * 64})
    assert excluded.prefix_sha256s == frozenset({"d" * 64})
    registry.write_text("{}")
    with pytest.raises(RuntimeError, match="snapshot changed"):
        subject.historical_exclusions(snapshot)


def _synthetic_pool(registry):
    rows, records = [], []
    counter = 0
    for role in contract.ROLE_ORDER:
        for domain in contract.NewlineDomain:
            for _ in range(contract.ROLE_DOMAIN_QUOTAS[role][domain.value]):
                row = torch.full((257,), 100, dtype=torch.long)
                row[0] = 1_000 + counter; row[81] = row[101] = 198
                rows.append(row)
                records.append(contract.CandidateRecord(
                    f"doc-{counter}", counter, f"source-{counter}", "revision",
                    f"{counter + 1:064x}", domain, "source-license", role,
                    f"{role.lower()}:{domain.value}:partition",
                    f"{counter + 20_000:064x}" if domain is contract.NewlineDomain.CODE else None,
                ))
                counter += 1
    return torch.stack(rows).contiguous(), tuple(records)


def _configure_transaction(monkeypatch, tmp_path):
    authority_path = tmp_path / "authority.json"; audit_path = tmp_path / "audit.json"
    cache = tmp_path / "cache"; manifest = cache / "manifest.json"
    receipt = tmp_path / "receipt.json"; failure = tmp_path / "failure.json"
    lock = tmp_path / "lock"
    role_files = {role: cache / f"{role.lower()}.pt" for role in contract.ROLE_ORDER}
    for name, value in {
        "AUTHORITY": authority_path, "AUDIT": audit_path, "CACHE": cache,
        "MANIFEST": manifest, "RECEIPT": receipt, "FAILURE": failure, "LOCK": lock,
        "ROLE_FILES": role_files,
    }.items():
        monkeypatch.setattr(subject, name, value)
    prior = tmp_path / "prior.json"; prior.write_text("{}")
    snapshot = {str(prior.resolve()): subject.file_sha256(prior)}
    sources = {"source.py": "a" * 64}; identity = {"identity": "frozen"}
    monkeypatch.setattr(subject, "source_closure", lambda _commit: dict(sources))
    monkeypatch.setattr(subject, "registry_snapshot", lambda: dict(snapshot))
    monkeypatch.setattr(subject, "source_identity", lambda _commit: dict(identity))
    authority = {
        "schema": "newline_l12h6_canary_v1_rows_authority",
        "status": "frozen_before_candidate_enumeration_or_row_access",
        "outcome_access": False, "source_commit": "b" * 40,
        "source_hashes": sources, "audit_path": str(audit_path),
        "registry_snapshot": snapshot, "source_identity": identity,
        "allocation_seed": subject.ALLOCATION_SEED, "outputs": subject._expected_outputs(),
    }
    authority_path.write_text(json.dumps(authority, sort_keys=True))
    authority_sha = subject.file_sha256(authority_path)
    audit = {
        "schema": "newline_l12h6_canary_v1_rows_independent_audit", "status": "GO",
        "outcome_access": False, "authority_sha256": authority_sha,
        "audited_source_commit": authority["source_commit"],
        "audited_source_hashes": sources, "tests_passed": 1, "reviewer": "independent",
    }
    audit_path.write_text(json.dumps(audit, sort_keys=True))
    import tiktoken
    registry = token_registry.build_registry(tiktoken.get_encoding("gpt2"))
    pool = _synthetic_pool(registry)
    monkeypatch.setattr(subject, "production_enumeration", lambda *_args: pool)
    return authority_path, audit_path, cache, receipt, failure, lock


def test_mocked_transaction_is_receipt_last_and_keeps_roles_separate(monkeypatch, tmp_path) -> None:
    authority, audit, cache, receipt, failure, lock = _configure_transaction(monkeypatch, tmp_path)
    result = subject.freeze(authority, audit)
    assert result["status"].endswith("receipt_last")
    assert receipt.is_file() and not failure.exists() and not lock.exists()
    assert tuple(sorted(path.name for path in cache.iterdir())) == tuple(sorted((
        "canary_select.pt", "final.pt", "ood.pt", "manifest.json",
    )))
    for role, path in subject.ROLE_FILES.items():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["role"] == role and payload["authorization"] == subject.ROLE_AUTHORIZATIONS[role]
    assert json.loads(receipt.read_text()) == result


def test_audit_failure_precedes_candidate_or_protected_source_access(monkeypatch, tmp_path) -> None:
    authority, audit, _cache, _receipt, _failure, _lock = _configure_transaction(monkeypatch, tmp_path)
    value = json.loads(audit.read_text()); value["status"] = "NO-GO"; audit.write_text(json.dumps(value))
    monkeypatch.setattr(subject, "source_identity", lambda *_args: (_ for _ in ()).throw(
        AssertionError("source opened before audit")
    ))
    monkeypatch.setattr(subject, "production_enumeration", lambda *_args: (_ for _ in ()).throw(
        AssertionError("candidate enumeration opened before audit")
    ))
    with pytest.raises(RuntimeError, match="not an exact"):
        subject.freeze(authority, audit)


def test_failure_binds_partial_state_and_no_success_receipt(monkeypatch, tmp_path) -> None:
    authority, audit, cache, receipt, failure, _lock = _configure_transaction(monkeypatch, tmp_path)
    monkeypatch.setattr(subject, "production_enumeration", lambda *_args: (_ for _ in ()).throw(
        RuntimeError("injected enumeration failure")
    ))
    with pytest.raises(RuntimeError, match="injected"):
        subject.freeze(authority, audit)
    terminal = json.loads(failure.read_text())
    assert terminal["status"] == "terminal_failure_without_success_receipt"
    assert terminal["artifacts"]["receipt"]["exists"] is False
    assert not receipt.exists() and not cache.exists()


def test_late_rival_failure_blocks_success_receipt(monkeypatch, tmp_path) -> None:
    authority, audit, _cache, receipt, failure, _lock = _configure_transaction(monkeypatch, tmp_path)
    original = subject.write_create_only
    def inject(path, data, *, before_link):
        if path == receipt:
            failure.write_text("{}")
        return original(path, data, before_link=before_link)
    monkeypatch.setattr(subject, "write_create_only", inject)
    with pytest.raises(RuntimeError, match="terminal/artifact state"):
        subject.freeze(authority, audit)
    assert not receipt.exists() and failure.exists()


def test_lock_replacement_and_adjacent_guard_are_fail_closed(tmp_path) -> None:
    lock = tmp_path / "lock"; claim = subject.acquire_claim(lock)
    try:
        lock.unlink(); lock.write_text(claim.nonce + "\n")
        with pytest.raises(RuntimeError, match="ownership changed"):
            subject.require_claim(claim, lock)
    finally:
        subject.release_claim(claim, lock)
    target = tmp_path / "target.json"; events = []
    subject.write_create_only(target, b"{}", before_link=lambda: events.append(target.exists()))
    assert events == [False] and target.read_bytes() == b"{}"
    with pytest.raises(FileExistsError):
        subject.write_create_only(target, b"changed", before_link=lambda: None)


def test_source_closure_is_exact_once_and_contains_transitive_tests() -> None:
    assert len(subject.SOURCE_PATHS) == len(set(subject.SOURCE_PATHS))
    assert {
        "basis_aligned/polynomial_causal/NEWLINE_FIXED_CREW_V1_PREREGISTRATION.md",
        "basis_aligned/polynomial_causal/NEWLINE_L12H6_CANARY_V1_EXECUTION_AMENDMENT.md",
        "basis_aligned/polynomial_causal/prepare_newline_l12h6_canary_v1_rows.py",
        "basis_aligned/polynomial_causal/test_prepare_newline_l12h6_canary_v1_rows.py",
        "basis_aligned/polynomial_causal/newline_l12h6_canary_rows_v1.py",
        "basis_aligned/polynomial_causal/newline_l12h6_token_registry_v1.py",
        "basis_aligned/polynomial_causal/local_fineweb_harvest.py",
        "jacclust/__init__.py", "jacclust/tt_model.py",
    }.issubset(subject.SOURCE_PATHS)

