from __future__ import annotations

import hashlib

import pytest
import torch

import bracket_closure_rows_v1 as rows_module
from bracket_closure_masks_v1 import (
    BracketDomain, DelimiterFamily, DelimiterRegistry,
)


REGISTRY = DelimiterRegistry(
    (DelimiterFamily("round", (10,), (11,)),
     DelimiterFamily("square", (20,), (21,))),
    quote_control_ids=(30,), punctuation_control_ids=(40,),
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _pool():
    values, records = [], []
    serial = 0
    for domain in BracketDomain:
        for cell in rows_module.CELL_ORDER:
            for _ in range(rows_module.ROWS_PER_CELL * 3):
                row = torch.full((257,), 99, dtype=torch.long)
                row[0] = 1000 + serial
                if cell == "compatible_closer":
                    row[60], row[65] = 10, 11
                elif cell == "incompatible_closer":
                    row[60], row[65] = 20, 11
                elif cell == "no_opener":
                    row[65] = 11
                elif cell == "quote_control":
                    row[65] = 30
                else:
                    row[65] = 40
                document = f"{domain.value}-doc-{serial}"
                source_file = (
                    f"src/code_{serial}.py" if domain is BracketDomain.CODE
                    else "fineweb/pinned.parquet"
                )
                records.append(rows_module.CandidateRecord(
                    document, serial, source_file, "revision-1", _sha(source_file),
                    domain, "allowed-license",
                    _sha(f"normalized-{serial}") if domain is BracketDomain.CODE else None,
                ))
                values.append(row); serial += 1
    return torch.stack(values).contiguous(), tuple(records)


def test_exact_stratified_roles_are_document_code_file_and_row_disjoint() -> None:
    candidates, records = _pool()
    roles = rows_module.allocate_roles(
        candidates, records, REGISTRY, rows_module.PriorExclusions.empty(), seed="fixed-v1",
    )
    assert tuple(role.role for role in roles) == tuple(rows_module.RowRole)
    assert all(role.rows.shape == (320, 257) for role in roles)
    documents = [{record.document_id for record in role.records} for role in roles]
    assert all(not documents[i] & documents[j] for i in range(3) for j in range(i + 1, 3))
    for role in roles:
        for domain in BracketDomain:
            census = role.support[domain.value]
            assert all(census[cell]["documents"] == 32 for cell in rows_module.CELL_ORDER)
            assert census["all"]["documents"] == 160


def test_historical_metadata_excludes_without_tensor_deserialization() -> None:
    candidates, records = _pool()
    first = records[0]
    prior = rows_module.historical_exclusions(({
        "record": {
            "document_id": first.document_id,
            "source_file": first.source_file,
            "source_blob_sha256": first.source_blob_sha256,
            "normalized_python_sha256": first.normalized_python_sha256,
            "row_sha256": rows_module.row_sha256(candidates[0]),
            "prefix32_sha256": rows_module.prefix32_sha256(candidates[0]),
            "cache_path": "/protected/old_rows.pt",
        },
    },))
    assert first.document_id in prior.documents
    assert rows_module.row_sha256(candidates[0]) in prior.row_sha256
    with pytest.raises(RuntimeError, match="lacks"):
        rows_module.allocate_roles(candidates, records, REGISTRY, prior, seed="fixed-v1")


def test_code_requires_normalized_hash_and_registry_is_exact() -> None:
    with pytest.raises(ValueError, match="normalized Python"):
        rows_module.CandidateRecord(
            "doc", 0, "x.py", "rev", "a" * 64,
            BracketDomain.CODE, "license", None,
        )
    assert len(rows_module.registry_sha256(REGISTRY)) == 64


def test_support_census_fails_closed_when_a_cell_is_removed() -> None:
    candidates, records = _pool()
    roles = rows_module.allocate_roles(
        candidates, records, REGISTRY, rows_module.PriorExclusions.empty(), seed="fixed-v1",
    )
    mask = roles[0].masks
    mask.compatible.zero_()
    with pytest.raises((ValueError, RuntimeError)):
        rows_module.support_census(mask, BracketDomain.PROSE)
