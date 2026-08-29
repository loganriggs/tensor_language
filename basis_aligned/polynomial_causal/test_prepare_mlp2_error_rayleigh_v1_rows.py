import copy
import json

import pytest
import torch

import prepare_mlp2_error_rayleigh_v1_rows as rows


def records(count):
    return [
        {"document_id": f"doc-{index}", "dataset_document_index": 121_000 + index}
        for index in range(count)
    ]


def test_configure_freezes_two_document_disjoint_roles(monkeypatch):
    monkeypatch.setattr(rows.base, "START_DOCUMENT_INDEX", -1)
    rows.configure()
    assert rows.base.START_DOCUMENT_INDEX == 121_000
    assert rows.base.DOCUMENTS_PER_ROLE == 32
    assert rows.base.TOTAL_DOCUMENTS == 64
    assert rows.base.ROLE_NAMES == ("DESIGN", "HELDOUT")
    assert rows.base.ROLE_AUTHORIZATIONS == rows.ROLE_AUTHORIZATIONS
    assert rows.base.SOURCE_PATHS == rows.SOURCE_PATHS


def test_split_rows_is_ordered_and_disjoint():
    rows.configure()
    tensor = torch.arange(64 * 257, dtype=torch.long).reshape(64, 257)
    split = rows.base.split_rows(tensor, records(64))
    assert set(split) == {"DESIGN", "HELDOUT"}
    assert torch.equal(split["DESIGN"][0], tensor[:32])
    assert torch.equal(split["HELDOUT"][0], tensor[32:])
    assert {x["document_id"] for x in split["DESIGN"][1]}.isdisjoint(
        x["document_id"] for x in split["HELDOUT"][1]
    )


def test_validate_selected_accepts_fresh_unique_documents():
    rows.configure()
    tensor = torch.arange(64 * 257, dtype=torch.long).reshape(64, 257)
    prior = (set(), set(), [], set())
    gates = rows.base.validate_selected(tensor, records(64), prior)
    assert all(gates.values())


def test_validate_selected_rejects_reused_document():
    rows.configure()
    tensor = torch.arange(64 * 257, dtype=torch.long).reshape(64, 257)
    reused = records(64)
    reused[-1] = copy.deepcopy(reused[0])
    with pytest.raises(RuntimeError, match="fresh MLP2 refit rows failed"):
        rows.base.validate_selected(tensor, reused, (set(), set(), [], set()))


def test_audit_validation_requires_exact_source_binding(tmp_path, monkeypatch):
    sources = {"a.py": "1" * 64}
    value = {
        "schema": "mlp2_error_rayleigh_v1_rows_independent_audit",
        "status": "GO", "outcome_access": False,
        "audited_source_commit": "c" * 40,
        "audited_source_hashes": sources, "tests_passed": 5,
        "reviewer": "independent-test",
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(rows, "source_hashes", lambda commit: sources)
    parsed, digest = rows.validate_independent_audit(sources, path)
    assert parsed == value and len(digest) == 64
    value["outcome_access"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="not an exact source-bound GO"):
        rows.validate_independent_audit(sources, path)
