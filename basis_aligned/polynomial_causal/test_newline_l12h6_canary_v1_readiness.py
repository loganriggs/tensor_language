from __future__ import annotations

from pathlib import Path
import dataclasses

import pytest
import tiktoken
import torch

from circuit_newline_fixed_crew_v1 import NewlineMaskSpec
import newline_l12h6_canary_rows_v1 as rows_contract
import newline_l12h6_canary_v1_readiness as subject
import newline_l12h6_token_registry_v1 as token_registry


def _materialized():
    registry = token_registry.build_registry(tiktoken.get_encoding("gpt2"))
    spec = NewlineMaskSpec(
        newline_token_ids=registry["newline"],
        punctuation_token_ids=registry["punctuation"],
        capitalized_token_ids=registry["capitalized"],
        quote_bracket_token_ids=registry["quote_bracket"],
        first_prediction=64,
        jitter_offsets=(2, -2, 3, -3, 4, -4, 8, -8, 16, -16, 32, -32),
        random_seed=2_026_083_000,
    )
    rows, records = [], []
    counter = 0
    for role in rows_contract.ROLE_ORDER:
        for domain in rows_contract.NewlineDomain:
            for local in range(rows_contract.ROLE_DOMAIN_QUOTAS[role][domain.value] + 4):
                row = torch.arange(257, dtype=torch.long) + 1_000 + counter
                row[81] = row[101] = 198
                rows.append(row)
                records.append(rows_contract.CandidateRecord(
                    f"doc-{counter}", counter, f"source-{counter}", "revision",
                    f"{counter + 1:064x}", domain, "permissive", role,
                    f"{role.lower()}-{domain.value}-heldout-{local % 3}",
                    f"{counter + 10_000:064x}" if domain is rows_contract.NewlineDomain.CODE else None,
                ))
                counter += 1
    roles = rows_contract.allocate_roles(
        torch.stack(rows).contiguous(), tuple(records), spec,
        rows_contract.HistoricalExclusions.empty(), seed="newline-canary-v1",
    )
    return registry, spec, roles


def test_readiness_is_exact_and_permanently_nonauthorizing() -> None:
    registry, spec, roles = _materialized()
    readiness = subject.build_readiness(
        roles, registry, spec, allocation_seed="newline-canary-v1",
    )
    assert not readiness.row_publication_authorized
    assert not readiness.model_forward_authorized
    assert tuple(role for role, _summary in readiness.role_summaries) == rows_contract.ROLE_ORDER
    with pytest.raises(RuntimeError, match="launch-NO-GO"):
        subject.require_launch_ready(readiness)


def test_mask_registry_or_spec_drift_is_rejected() -> None:
    registry, spec, roles = _materialized()
    wrong = NewlineMaskSpec(
        spec.newline_token_ids, spec.punctuation_token_ids,
        spec.capitalized_token_ids, spec.quote_bracket_token_ids,
        first_prediction=63, jitter_offsets=spec.jitter_offsets,
        random_seed=spec.random_seed,
    )
    with pytest.raises(RuntimeError, match="mask spec"):
        subject.build_readiness(
            roles, registry, wrong, allocation_seed="newline-canary-v1",
        )
    tampered = list(roles)
    bad_masks = dataclasses.replace(
        roles[0].masks, newline_target=roles[0].masks.position_jitter.clone(),
        position_jitter=roles[0].masks.newline_target.clone(),
    )
    tampered[0] = dataclasses.replace(roles[0], masks=bad_masks)
    with pytest.raises(RuntimeError, match="do not replay"):
        subject.build_readiness(
            tuple(tampered), registry, spec, allocation_seed="newline-canary-v1",
        )


def test_source_closure_is_exact_once_and_import_complete() -> None:
    assert len(subject.SOURCE_PATHS) == len(set(subject.SOURCE_PATHS))
    assert {
        "basis_aligned/polynomial_causal/NEWLINE_FIXED_CREW_V1_PREREGISTRATION.md",
        "basis_aligned/polynomial_causal/NEWLINE_L12H6_CANARY_V1_EXECUTION_AMENDMENT.md",
        "basis_aligned/polynomial_causal/circuit_newline_fixed_crew_v1.py",
        "basis_aligned/polynomial_causal/circuit_campaign_runtime.py",
        "basis_aligned/polynomial_causal/circuit_campaign_statistics.py",
        "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
        "jacclust/__init__.py", "jacclust/tt_model.py",
    }.issubset(subject.SOURCE_PATHS)
    assert all((subject.ROOT / relative).is_file() for relative in subject.SOURCE_PATHS)
    source = Path(subject.__file__).read_text()
    assert "torch.load" not in source and "load_bilin18" not in source
