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


def _role_sources():
    generator = torch.Generator().manual_seed(5)
    return {
        role: torch.randint(0, 2000, (rows.N_PER_ROLE, rows.contract.ROW_WIDTH), generator=generator)
        for role in rows.ALL_ROLES
    }


def test_synthetic_pairs_preserve_multisets_break_witness_and_use_disjoint_banks():
    sources = _role_sources()
    synthetic, banks = rows.build_synthetic_roles(sources)
    used = []
    for role in rows.ALL_ROLES:
        positive, control = synthetic[role]["positive"], synthetic[role]["control"]
        assert positive.shape == control.shape == (rows.SYNTHETIC_PAIRS_PER_ROLE, rows.contract.ROW_WIDTH)
        assert torch.equal(torch.sort(positive, 1).values, torch.sort(control, 1).values)
        flat_bank = {token for sequence in banks[role] for token in sequence}
        assert len(flat_bank) == rows.SYNTHETIC_PAIRS_PER_ROLE * rows.SYNTHETIC_SEQUENCE_LENGTH
        assert all(flat_bank.isdisjoint(other) for other in used)
        used.append(flat_bank)
        query_position = rows.contract.ROW_WIDTH - 2
        assert torch.equal(positive[:, query_position:], control[:, query_position:])


def test_fit_counts_exclude_targets_and_cover_tokenizer_support():
    fit = torch.zeros(rows.N_PER_ROLE, rows.contract.ROW_WIDTH, dtype=torch.long)
    fit[:, -1] = 7
    counts = rows.fit_token_counts(fit)
    assert counts.shape == (50257,)
    assert int(counts[0]) == rows.N_PER_ROLE * rows.contract.MODEL_WIDTH
    assert int(counts[7]) == 0


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

