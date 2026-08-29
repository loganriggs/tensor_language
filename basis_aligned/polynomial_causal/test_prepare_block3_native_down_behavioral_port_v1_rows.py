import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).with_name("prepare_block3_native_down_behavioral_port_v1_rows.py")
SPEC = importlib.util.spec_from_file_location("block3_native_down_fresh_rows", PATH)
ROWS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ROWS)


def _texts(n=12):
    for index in range(n):
        yield f"doc-{index}", "x" * (40 + index)


def _encode(text):
    base = len(text) * 100
    return list(range(base, base + 96))


def test_harvest_is_ordered_one_row_per_document_and_skips_prior_identities():
    prior = ({"doc-2"}, {3}, set(), set())
    rows, records = ROWS.harvest_fresh_documents(
        _texts(), _encode, prior,
        start_document_index=2, n_source_documents=3, token_length=32,
    )
    assert tuple(rows.shape) == (3, 32)
    assert [record["document_id"] for record in records] == ["doc-4", "doc-5", "doc-6"]
    assert [record["dataset_document_index"] for record in records] == [4, 5, 6]
    assert [record["row_index"] for record in records] == [0, 1, 2]


def test_harvest_skips_prior_257_projection_and_prefix_then_uses_later_chunk():
    tokens = list(range(96))
    texts = [("doc-a", "a")]
    encode = lambda _text: tokens
    prior_row = tuple(tokens[:32]) + tuple(range(100, 140))
    prior = (set(), set(), {prior_row}, set())
    rows, records = ROWS.harvest_fresh_documents(
        texts, encode, prior, start_document_index=0,
        n_source_documents=1, token_length=32,
    )
    assert [record["document_id"] for record in records] == ["doc-a"]
    assert records[0]["token_start"] == 32
    assert torch.equal(rows[0], torch.tensor(tokens[32:64]))


def test_harvest_rejects_short_source_and_malformed_inputs():
    with pytest.raises(RuntimeError, match="eligible documents"):
        ROWS.harvest_fresh_documents(
            _texts(2), _encode, (set(), set(), set(), set()),
            start_document_index=1, n_source_documents=2, token_length=32,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        ROWS.harvest_fresh_documents([], _encode, (set(), set(), set(), set()),
                                     start_document_index=-1)


def test_disjointness_checks_every_registered_identity_class(monkeypatch):
    monkeypatch.setattr(ROWS, "N_SOURCE_DOCUMENTS", 2)
    monkeypatch.setattr(ROWS, "TOKEN_LENGTH", 8)
    rows = torch.arange(16, dtype=torch.long).view(2, 8)
    records = [
        {"document_id": "a", "dataset_document_index": 10,
         "source_document_ordinal": 0, "row_index": 0},
        {"document_id": "b", "dataset_document_index": 11,
         "source_document_ordinal": 1, "row_index": 1},
    ]
    assert all(ROWS.validate_disjointness(
        rows, records, (set(), set(), set(), set())
    ).values())
    with pytest.raises(RuntimeError, match="source_documents"):
        ROWS.validate_disjointness(rows, records, ({"a"}, set(), set(), set()))
    with pytest.raises(RuntimeError, match="dataset_indices"):
        ROWS.validate_disjointness(rows, records, (set(), {10}, set(), set()))
    with pytest.raises(RuntimeError, match="full257_rows"):
        ROWS.validate_disjointness(
            rows, records, (set(), set(), {tuple(rows[0].tolist())}, set())
        )
    with pytest.raises(RuntimeError, match="prefix32"):
        ROWS.validate_disjointness(
            rows, records, (set(), set(), set(), {tuple(rows[0].tolist())})
        )


def test_summary_freezes_identity_row_map(monkeypatch):
    monkeypatch.setattr(ROWS, "N_SOURCE_DOCUMENTS", 2)
    monkeypatch.setattr(ROWS, "TOKEN_LENGTH", 8)
    rows = torch.zeros(2, 8, dtype=torch.long)
    records = [
        {"document_id": "a", "dataset_document_index": 60_001,
         "source_document_ordinal": 0, "row_index": 0},
        {"document_id": "b", "dataset_document_index": 60_003,
         "source_document_ordinal": 1, "row_index": 1},
    ]
    summary = ROWS.summarize(rows, records)
    assert summary["row_to_document_identity"] == [0, 1]
    with pytest.raises(RuntimeError, match="canonical"):
        ROWS.summarize(rows, [records[1], records[0]])


def test_registry_discovery_is_recursive_and_excludes_own_receipt(tmp_path, monkeypatch):
    basis = tmp_path / "basis_aligned"
    nested = basis / "polynomial_causal" / "nested"
    nested.mkdir(parents=True)
    canonical = basis / "canonical_receipt.json"
    canonical.write_text("{}")
    prior = nested / "prior_authority.json"
    prior.write_text("{}")
    own = nested / "block3_native_down_behavioral_port_v1_rows_receipt.json"
    own.write_text("{}")
    monkeypatch.setattr(ROWS, "BASIS", basis)
    monkeypatch.setattr(ROWS.BASE, "CANONICAL_RECEIPT", canonical)
    monkeypatch.setattr(ROWS, "RECEIPT", own)
    assert ROWS.discover_registry_files() == tuple(sorted((canonical.resolve(), prior.resolve())))


def test_registry_json_is_hash_bound_to_the_bytes_actually_parsed(tmp_path, monkeypatch):
    registry = tmp_path / "prior_receipt.json"
    registry.write_text("{}")
    reference = tmp_path / "eval_tokens.pt"
    torch.save(torch.arange(64, dtype=torch.long).view(1, 64), reference)
    monkeypatch.setattr(ROWS, "REFERENCE_ROWS", reference)
    real_hash = ROWS.file_sha256
    calls = {str(registry): 0}

    def drifting_hash(path):
        if path == registry:
            calls[str(registry)] += 1
            if calls[str(registry)] == 2:
                return "0" * 64
        return real_hash(path)

    monkeypatch.setattr(ROWS, "file_sha256", drifting_hash)
    with pytest.raises(RuntimeError, match="changed while reading"):
        ROWS.load_registry_exclusions((registry,))


def test_create_only_writer_never_replaces(tmp_path):
    path = tmp_path / "receipt.json"
    ROWS.write_json_create_only({"a": 1}, path)
    with pytest.raises(FileExistsError):
        ROWS.write_json_create_only({"a": 2}, path)
    assert path.read_text() == '{\n  "a": 1\n}\n'


def test_create_only_writer_rechecks_guard_immediately_before_link(tmp_path):
    path = tmp_path / "receipt.json"
    protected = tmp_path / "protected.json"
    protected.write_text("original")

    def guard():
        if protected.read_text() != "original":
            raise RuntimeError("protected registry drift")

    ROWS.write_json_create_only({"a": 1}, path, pre_link_check=guard)
    second = tmp_path / "second.json"
    protected.write_text("changed")
    with pytest.raises(RuntimeError, match="registry drift"):
        ROWS.write_json_create_only({"a": 2}, second, pre_link_check=guard)
    assert not second.exists()


def test_lock_claim_rejects_replacement(tmp_path):
    path = tmp_path / "claim.lock"
    claim = ROWS.acquire_claim(path)
    try:
        path.unlink()
        path.write_text("attacker\n")
        with pytest.raises(RuntimeError, match="replaced"):
            ROWS.require_claim(claim, path)
    finally:
        ROWS.release_claim(claim, path)


def test_freeze_refuses_spent_namespace_before_any_source_or_row_load(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    receipt = tmp_path / "receipt.json"
    lock = tmp_path / "claim.lock"
    monkeypatch.setattr(ROWS, "CACHE", cache)
    monkeypatch.setattr(ROWS, "RECEIPT", receipt)
    monkeypatch.setattr(ROWS, "LOCK", lock)
    claim = ROWS.acquire_claim(lock)
    try:
        with pytest.raises(RuntimeError, match="overwrite"):
            ROWS.freeze_locked(claim)
    finally:
        ROWS.release_claim(claim, lock)


def test_source_is_outcome_blind_and_closes_transitive_registry_helpers():
    source = PATH.read_text()
    assert "transformers" not in source
    assert ".cuda(" not in source
    assert "model(" not in source
    assert "scientific_outcomes_read\": False" in source
    assert ROWS.REGISTRY_PATH in ROWS.SOURCE_PATHS
    assert ROWS.BASE_PATH in ROWS.SOURCE_PATHS
    assert ROWS.LOCAL_HARVESTER in ROWS.SOURCE_PATHS
    assert ROWS.TEST in ROWS.SOURCE_PATHS
    assert ROWS.PREREGISTRATION in ROWS.SOURCE_PATHS


def test_exact_failed_unmaterialized_lineage_is_waived_twice_and_only_twice():
    registry = (ROWS.FAILED_ROW_AUTHORITY, ROWS.TERMINAL_COPY_V2_RECEIPT)
    _prior, hashes, tensor_hashes, waivers, nonrows = ROWS.load_registry_exclusions(registry)
    assert set(hashes) == {str(path) for path in registry}
    assert {item["registry_json"] for item in waivers} == {str(path) for path in registry}
    assert all(item["omitted_missing_row_path"] == str(ROWS.FAILED_ROW) for item in waivers)
    assert nonrows == [{
        "kind": "exact_nonrow_frequency_vector",
        "path": str(ROWS.TERMINAL_COPY_FIT_FREQUENCIES),
        "file_sha256": ROWS.TERMINAL_COPY_FIT_FREQUENCIES_SHA256,
        "keys": ["query", "target"],
        "shape": [50_257],
        "reason": "contains no document rows; filename-only row heuristic overmatch",
    }]
    assert tensor_hashes[str(ROWS.TERMINAL_COPY_FIT_FREQUENCIES)] == (
        ROWS.TERMINAL_COPY_FIT_FREQUENCIES_SHA256
    )


def test_embedded_waiver_proofs_are_validated_but_not_reparsed_as_rows():
    prior = ROWS.BQ / "mlp2_rank512_refit_v1_rows_receipt.json"
    registry = (ROWS.FAILED_ROW_AUTHORITY, ROWS.TERMINAL_COPY_V2_RECEIPT, prior)
    _prior, hashes, _tensor_hashes, waivers, _nonrows = ROWS.load_registry_exclusions(registry)
    assert str(prior) in hashes
    assert {item["registry_json"] for item in waivers} == {
        str(ROWS.FAILED_ROW_AUTHORITY), str(ROWS.TERMINAL_COPY_V2_RECEIPT),
    }


def test_independent_audit_binds_exact_source_bytes(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("x = 1\n")
    digest = ROWS.file_sha256(source)
    commit = "a" * 40
    audit = tmp_path / "audit.json"
    audit.write_text(__import__("json").dumps({
        "schema": "block3_native_down_behavioral_port_v1_rows_independent_audit",
        "status": "GO",
        "outcome_access": False,
        "audited_source_commit": commit,
        "audited_source_hashes": {source.name: digest},
        "tests_passed": 17,
        "reviewer": "independent-test",
    }))
    monkeypatch.setattr(ROWS, "SOURCE_PATHS", (source,))
    monkeypatch.setattr(ROWS, "ROOT", tmp_path)
    monkeypatch.setattr(ROWS, "_committed_blob", lambda _path, _commit: source.read_bytes())
    monkeypatch.setattr(ROWS.subprocess, "run", lambda *args, **kwargs: None)
    payload, observed = ROWS.validate_independent_audit(audit)
    assert observed == ROWS.file_sha256(audit)
    assert payload["tests_passed"] == 17
    source.write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="changed after"):
        ROWS.validate_independent_audit(audit)


def test_canonical_constants_are_exactly_frozen():
    assert ROWS.START_DOCUMENT_INDEX == 60_000
    assert ROWS.N_SOURCE_DOCUMENTS == 192
    assert ROWS.TOKEN_LENGTH == 257
    assert ROWS.MAX_CHUNKS_PER_DOCUMENT == 1
    assert ROWS.RECEIPT.name == "block3_native_down_behavioral_port_v1_rows_receipt.json"
