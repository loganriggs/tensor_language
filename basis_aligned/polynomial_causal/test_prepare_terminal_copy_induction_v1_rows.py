from __future__ import annotations

import hashlib

import pytest
import torch

import prepare_terminal_copy_induction_v1_rows as rows


def _natural_combined():
    total = len(rows.NATURAL_ROLES) * rows.N_PER_ROLE
    tensor = torch.arange(total * rows.contract.ROW_WIDTH, dtype=torch.long).reshape(
        total, rows.contract.ROW_WIDTH,
    )
    records = [{"document_id": f"doc-{index}"} for index in range(total)]
    return tensor, records


def test_natural_split_is_role_and_document_disjoint():
    tensor, records = _natural_combined()
    role_rows, role_records = rows.split_natural_rows(tensor, records)
    assert set(role_rows) == set(rows.NATURAL_ROLES)
    assert all(value.shape == (rows.N_PER_ROLE, rows.contract.ROW_WIDTH) for value in role_rows.values())
    documents = [item["document_id"] for role in rows.NATURAL_ROLES for item in role_records[role]]
    assert len(documents) == len(set(documents)) == 3 * rows.N_PER_ROLE
    assert role_records["selection_natural"][0]["role_row_index"] == 0


def test_natural_split_rejects_repeated_document():
    tensor, records = _natural_combined()
    records[-1]["document_id"] = records[0]["document_id"]
    with pytest.raises(RuntimeError, match="repeat"):
        rows.split_natural_rows(tensor, records)


def test_code_allocator_is_path_row_prefix_fresh_and_deterministic():
    blobs = []
    for index in range(6):
        # A simple encoder below turns every byte character into one supported ID.
        blobs.append((f"p{index}.py", bytes([65 + index]) * 700))
    encode = lambda text: [ord(character) for character in text]
    prior = (set(), set(), set(), set())
    first, first_records = rows.allocate_code_rows(blobs, encode, prior, {"p0.py"}, n_rows=4)
    second, second_records = rows.allocate_code_rows(blobs, encode, prior, {"p0.py"}, n_rows=4)
    assert torch.equal(first, second)
    assert first_records == second_records
    assert len({record["path"] for record in first_records}) == 4
    assert "p0.py" not in {record["path"] for record in first_records}
    blocked_prefix = {tuple(first[0, : rows.natural.PREFIX_LENGTH].tolist())}
    replaced, _ = rows.allocate_code_rows(blobs, encode, (set(), set(), set(), blocked_prefix), {"p0.py"}, n_rows=4)
    assert not torch.equal(first[0], replaced[0])


def test_code_allocator_fails_when_file_support_is_insufficient():
    with pytest.raises(RuntimeError, match="1/2"):
        rows.allocate_code_rows(
            [("a.py", b"a" * 300)], lambda text: [1] * len(text),
            (set(), set(), set(), set()), set(), n_rows=2,
        )


def test_code_register_excludes_experiments_generated_tests_and_normalized_duplicates():
    assert not rows.code_path_is_eligible("basis_aligned/polynomial_causal/runner.py")
    assert not rows.code_path_is_eligible("runs/example.py")
    assert not rows.code_path_is_eligible("pkg/generated/module.py")
    assert not rows.code_path_is_eligible("pkg/test_module.py")
    assert rows.code_path_is_eligible("jacclust/model.py")
    first = b"x = 1  # comment\nprint(x)\n"
    second = b"x=999\n\nprint( x )\n"
    assert rows.normalized_python_sha256(first) == rows.normalized_python_sha256(second)
    blobs = [("a.py", first * 80), ("b.py", second * 80), ("c.py", b"y=2\n" * 150)]
    selected, records = rows.allocate_code_rows(
        blobs, lambda text: [ord(char) % 127 for char in text],
        (set(), set(), set(), set()), set(), n_rows=2,
    )
    assert selected.shape == (2, rows.contract.ROW_WIDTH)
    assert {record["path"] for record in records} == {"a.py", "c.py"}
    assert len({record["normalized_python_sha256"] for record in records}) == 2


def test_prior_code_paths_recurses_generic_registries_and_normalizes_workspace_paths(tmp_path):
    manifest = tmp_path / "registry.json"
    inside = rows.ROOT / "jacclust" / "model.py"
    manifest.write_text(__import__("json").dumps({
        "nested": [{"source_path": str(inside)}, {"file_path": "pkg/module.py"}],
        "ignore": {"path": "not-code.txt"},
    }))
    paths, hashes = rows.prior_code_paths((manifest,))
    assert paths == {"jacclust/model.py", "pkg/module.py"}
    assert hashes[str(manifest.resolve())] == rows.file_sha256(manifest)


def _role_sources():
    generator = torch.Generator().manual_seed(5)
    return {
        role: torch.randint(0, 2000, (rows.N_PER_ROLE, rows.contract.ROW_WIDTH), generator=generator)
        for role in rows.ALL_ROLES
    }


def test_synthetic_crossovers_preserve_multisets_reverse_association_and_use_disjoint_banks():
    sources = _role_sources()
    synthetic, banks = rows.build_synthetic_roles(sources)
    used = []
    for role in rows.ALL_ROLES:
        query_to_y = synthetic[role]["query_to_y"]
        query_to_z = synthetic[role]["query_to_z"]
        assert query_to_y.shape == query_to_z.shape == (
            rows.SYNTHETIC_PAIRS_PER_ROLE, rows.contract.ROW_WIDTH,
        )
        assert torch.equal(
            torch.sort(query_to_y, 1).values, torch.sort(query_to_z, 1).values,
        )
        flat_bank = {token for sequence in banks[role] for token in sequence}
        assert len(flat_bank) == rows.SYNTHETIC_PAIRS_PER_ROLE * 4
        assert all(flat_bank.isdisjoint(other) for other in used)
        used.append(flat_bank)
        for index, bank in enumerate(banks[role]):
            first, reciprocal, query = rows.SYNTHETIC_POSITION_TEMPLATES[
                index % len(rows.SYNTHETIC_POSITION_TEMPLATES)
            ]
            q, r, y, z = bank
            assert query_to_y[index, first:first + 2].tolist() == [q, y]
            assert query_to_z[index, first:first + 2].tolist() == [q, z]
            assert query_to_y[index, reciprocal:reciprocal + 2].tolist() == [r, z]
            assert query_to_z[index, reciprocal:reciprocal + 2].tolist() == [r, y]
            assert query_to_y[index, query:query + 2].tolist() == [q, y]
            assert query_to_z[index, query:query + 2].tolist() == [q, y]


def test_fit_frequencies_bind_distinct_query_and_target_domains():
    fit = torch.zeros(rows.N_PER_ROLE, rows.contract.ROW_WIDTH, dtype=torch.long)
    fit[:, -1] = 7
    frequencies = rows.fit_token_frequencies(fit)
    assert frequencies.query.shape == frequencies.target.shape == (50257,)
    assert int(frequencies.query[0]) == rows.N_PER_ROLE * rows.contract.MODEL_WIDTH
    assert int(frequencies.query[7]) == 0
    assert int(frequencies.target[0]) == rows.N_PER_ROLE * (rows.contract.MODEL_WIDTH - 1)
    assert int(frequencies.target[7]) == rows.N_PER_ROLE


def test_copy_cell_serialization_preserves_support_diagnostics():
    tensor, records = _natural_combined()
    role_rows, role_records = rows.split_natural_rows(tensor, records)
    frequencies = rows.fit_token_frequencies(role_rows["fit_natural"])
    cells = rows.contract.build_copy_cells(
        role_rows["fit_natural"], frequencies,
        tuple(record["document_id"] for record in role_records["fit_natural"]),
    )
    payload = rows.serialize_copy_cells(cells)
    assert set(payload) == {
        "all_positive", "positive", "matched_negative", "off_target",
        "pair_indices", "unmatched_positive_count", "negative_candidate_count",
        "eligible_stratum_count", "excluded_low_document_stratum_count",
    }
    assert payload["positive"].shape == (rows.N_PER_ROLE, rows.contract.MODEL_WIDTH)


def test_support_census_fails_closed_below_frozen_document_or_position_minima():
    shape = (rows.N_PER_ROLE, rows.contract.MODEL_WIDTH)
    all_positive = torch.zeros(shape, dtype=torch.bool)
    positive = torch.zeros(shape, dtype=torch.bool)
    matched_negative = torch.zeros(shape, dtype=torch.bool)
    positive_coordinates = [
        (document, position)
        for document in range(23) for position in (64, 65)
    ] + [(0, 66), (1, 66)]
    negative_coordinates = [
        (document, position)
        for document in range(24) for position in (80, 81)
    ]
    for document, position in positive_coordinates:
        positive[document, position] = True
    for document, position in negative_coordinates:
        matched_negative[document, position] = True
    all_positive |= positive
    valid = torch.zeros(shape, dtype=torch.bool)
    valid[:, rows.contract.SCORE_START:rows.contract.SCORE_STOP] = True
    cells = rows.contract.CopyCells(
        all_positive=all_positive,
        positive=positive,
        matched_negative=matched_negative,
        off_target=valid & ~all_positive & ~matched_negative,
        pair_indices=tuple(
            (*positive_coordinates[index], *negative_coordinates[index])
            for index in range(48)
        ),
        unmatched_positive_count=0,
        negative_candidate_count=48,
        eligible_stratum_count=1,
        excluded_low_document_stratum_count=0,
    )
    with pytest.raises(RuntimeError, match="pre-model support gate failed"):
        rows.support_census({
            "selection_natural": cells, "final_natural": cells, rows.OOD_ROLE: cells,
        })


def test_summary_requires_all_roles_and_closes_every_gate():
    role_rows = _role_sources()
    synthetic, banks = rows.build_synthetic_roles(role_rows)
    records = {
        role: ([{"document_id": f"{role}-{i}"} for i in range(rows.N_PER_ROLE)]
               if role in rows.NATURAL_ROLES
               else [{"path": f"p{i}.py"} for i in range(rows.N_PER_ROLE)])
        for role in rows.ALL_ROLES
    }
    summary = rows.summarize_roles(role_rows, records, synthetic, banks)
    assert all(summary["gates"].values())


def test_audit_hash_constants_are_full_sha256_when_authority_is_created():
    # The real audit is deliberately absent until an independent reviewer approves
    # the committed source; this test prevents a placeholder from being mistaken for it.
    assert not rows.AUDIT.exists() or len(hashlib.sha256(rows.AUDIT.read_bytes()).hexdigest()) == 64
    assert rows.AUDIT not in rows.SOURCE_PATHS
    assert rows.CONTRACT_TEST in rows.SOURCE_PATHS


def test_terminal_json_post_link_fsync_error_is_success_not_contradictory_failure(
    tmp_path, monkeypatch,
):
    lock = tmp_path / "owner.lock"
    claim = rows.natural.acquire_claim(lock)
    try:
        monkeypatch.setattr(
            rows, "_fsync_directory",
            lambda path: (_ for _ in ()).throw(OSError("synthetic post-link fsync")),
        )
        target = tmp_path / "receipt.json"
        rows._publish_json_terminal({"status": "complete"}, target, claim, lock_path=lock)
        assert __import__("json").loads(target.read_bytes()) == {"status": "complete"}
        with pytest.raises(FileExistsError):
            rows._publish_json_terminal({"status": "again"}, target, claim, lock_path=lock)
    finally:
        rows.natural.release_claim(claim, lock)


def test_cached_payload_validator_binds_every_tensor_and_semantic_field(tmp_path):
    role = rows.ALL_ROLES[0]
    source = torch.zeros(rows.N_PER_ROLE, rows.contract.ROW_WIDTH, dtype=torch.long)
    role_sources = {name: source.clone() for name in rows.ALL_ROLES}
    synthetic, banks = rows.build_synthetic_roles(role_sources, pairs_per_role=2)
    cells_shape = (rows.N_PER_ROLE, rows.contract.MODEL_WIDTH)
    positive = torch.zeros(cells_shape, dtype=torch.bool)
    negative = torch.zeros(cells_shape, dtype=torch.bool)
    all_positive = positive.clone()
    valid = torch.zeros(cells_shape, dtype=torch.bool)
    valid[:, rows.contract.SCORE_START:rows.contract.SCORE_STOP] = True
    payload = {
        "rows": source,
        "records": [{"document_id": f"d{i}"} for i in range(rows.N_PER_ROLE)],
        "synthetic": synthetic[role],
        "synthetic_token_banks": banks[role],
        "synthetic_position_templates": rows.SYNTHETIC_POSITION_TEMPLATES,
        "copy_cells": {
            "all_positive": all_positive, "positive": positive,
            "matched_negative": negative, "off_target": valid,
            "pair_indices": (), "unmatched_positive_count": 0,
            "negative_candidate_count": 0, "eligible_stratum_count": 0,
            "excluded_low_document_stratum_count": 0,
        },
    }
    path = tmp_path / "role.pt"
    rows._save_staged(payload, path)
    entry = {
        "file_sha256": rows.file_sha256(path),
        "rows_tensor_sha256": rows.tensor_sha256(source),
        "query_to_y_tensor_sha256": rows.tensor_sha256(synthetic[role]["query_to_y"]),
        "query_to_z_tensor_sha256": rows.tensor_sha256(synthetic[role]["query_to_z"]),
        "copy_positive_mask_sha256": rows.tensor_sha256(positive),
        "copy_matched_negative_mask_sha256": rows.tensor_sha256(negative),
    }
    loaded = rows.validate_cached_payload(path, payload, entry, role)
    assert rows._semantic_equal(loaded, payload)
    bad = dict(payload)
    bad["records"] = list(payload["records"])
    bad["records"][0] = {"document_id": "changed"}
    with pytest.raises(RuntimeError, match="exact replay"):
        rows.validate_cached_payload(path, bad, entry, role)


def test_receipt_json_normalization_makes_nested_templates_reload_exactly():
    value = {
        "selection": {"synthetic_position_templates": rows.SYNTHETIC_POSITION_TEMPLATES},
        "roles": rows.ALL_ROLES,
    }
    normalized = rows.json_normalize(value)
    assert normalized == __import__("json").loads(__import__("json").dumps(normalized))
    assert normalized["selection"]["synthetic_position_templates"][0] == [8, 32, 80]
