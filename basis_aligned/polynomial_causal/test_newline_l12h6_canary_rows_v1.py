from __future__ import annotations

import dataclasses

import pytest
import torch

from circuit_newline_fixed_crew_v1 import NewlineMaskSpec
import newline_l12h6_canary_rows_v1 as subject


def _spec() -> NewlineMaskSpec:
    return NewlineMaskSpec(
        newline_token_ids=(10,), punctuation_token_ids=(11,),
        capitalized_token_ids=(12,), quote_bracket_token_ids=(13,),
        first_prediction=64, jitter_offsets=(2, -2, 3, -3, 4, -4), random_seed=17,
    )


def _pool(extra_per_domain: int = 4):
    rows, records = [], []
    counter = 0
    for role in subject.ROLE_ORDER:
        for domain in subject.NewlineDomain:
            for local in range(subject.ROLE_DOMAIN_QUOTAS[role][domain.value] + extra_per_domain):
                row = torch.arange(257, dtype=torch.long) + 1_000 + counter
                row[81] = row[101] = 10
                rows.append(row)
                records.append(subject.CandidateRecord(
                    document_id=f"doc-{counter}", source_document_index=counter,
                    source_file=f"source-{counter}", source_revision="rev",
                    source_blob_sha256=f"{counter + 1:064x}", domain=domain,
                    license_id="permissive", role_license=role,
                    structural_partition=f"{role.lower()}-{domain.value}-{local % 3}",
                    normalized_python_sha256=(f"{counter + 10_000:064x}"
                                              if domain is subject.NewlineDomain.CODE else None),
                ))
                counter += 1
    return torch.stack(rows).contiguous(), tuple(records)


def test_exact_deterministic_fresh_roles_and_support() -> None:
    rows, records = _pool()
    left = subject.allocate_roles(
        rows, records, _spec(), subject.HistoricalExclusions.empty(), seed="newline-v1",
    )
    right = subject.allocate_roles(
        rows, records, _spec(), subject.HistoricalExclusions.empty(), seed="newline-v1",
    )
    assert tuple(role.role for role in left) == subject.ROLE_ORDER
    assert [subject.role_summary(role) for role in left] == [
        subject.role_summary(role) for role in right
    ]
    for role in left:
        assert role.support["domains"] == subject.ROLE_DOMAIN_QUOTAS[role.role]
        assert role.support["target_documents"] >= subject.MIN_TARGET_DOCUMENTS
        assert role.support["cells"]["newline_target"] >= subject.MIN_TARGET_POSITIONS
        assert torch.equal(role.masks.newline_target.sum(1), torch.full(
            (role.rows.shape[0],), 2, dtype=torch.long,
        ))
    subject.validate_role_disjointness(left)


def test_historical_exclusions_are_applied_before_sha_allocation() -> None:
    rows, records = _pool(extra_per_domain=8)
    excluded = subject.HistoricalExclusions(
        frozenset({records[0].document_id}), frozenset(), frozenset(), frozenset(),
        frozenset({subject.tensor_sha256(rows[1].contiguous())}),
        frozenset({subject.tensor_sha256(rows[2, :32].contiguous())}),
    )
    roles = subject.allocate_roles(rows, records, _spec(), excluded, seed="newline-v1")
    selected_documents = {record.document_id for role in roles for record in role.records}
    assert not selected_documents.intersection({records[index].document_id for index in (0, 1, 2)})


def test_candidate_and_role_identity_collisions_fail_closed() -> None:
    rows, records = _pool()
    bad = list(records); bad[1] = dataclasses.replace(bad[1], document_id=bad[0].document_id)
    with pytest.raises(ValueError, match="repeats"):
        subject.allocate_roles(
            rows, tuple(bad), _spec(), subject.HistoricalExclusions.empty(), seed="newline-v1",
        )
    roles = list(subject.allocate_roles(
        rows, records, _spec(), subject.HistoricalExclusions.empty(), seed="newline-v1",
    ))
    copied = list(roles[1].records); copied[0] = roles[0].records[0]
    roles[1] = dataclasses.replace(roles[1], records=tuple(copied))
    with pytest.raises(RuntimeError, match="overlap"):
        subject.validate_role_disjointness(tuple(roles))


def test_underpowered_domain_and_missing_newline_fail_closed() -> None:
    rows, records = _pool(extra_per_domain=0)
    last_ood_list = max(index for index, record in enumerate(records) if (
        record.role_license == "OOD" and record.domain is subject.NewlineDomain.LIST
    ))
    keep = [index for index in range(len(records)) if index != last_ood_list]
    with pytest.raises(RuntimeError, match="has .* <"):
        subject.allocate_roles(
            rows[keep].contiguous(), tuple(records[index] for index in keep), _spec(),
            subject.HistoricalExclusions.empty(), seed="newline-v1",
        )
    rows, records = _pool(extra_per_domain=0)
    rows[:, 81] = rows[:, 82]; rows[:, 101] = rows[:, 102]
    with pytest.raises(RuntimeError, match="has 0 <"):
        subject.allocate_roles(
            rows.contiguous(), records, _spec(), subject.HistoricalExclusions.empty(),
            seed="newline-v1",
        )
