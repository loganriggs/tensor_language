from __future__ import annotations

import dataclasses

import pytest
import torch

import bracket_closure_execution_v1 as subject
import bracket_closure_canary_v1 as canary
from bracket_closure_canary_v1 import ARM_NAMES
from bracket_closure_masks_v1 import BracketMasks
from bracket_closure_tensor_v1 import PRODUCTION_STORED_VALUES


def _authority():
    return subject.ExecutionAuthority(
        "a" * 40, (("source.py", "b" * 64),), "c" * 64,
        (("fit", "d" * 64), ("select", "e" * 64), ("ood", "f" * 64)),
        (("fit", "4" * 64), ("select", "5" * 64), ("ood", "6" * 64)),
        (("fit", "7" * 64), ("select", "8" * 64), ("ood", "9" * 64)),
        ("round", "square"),
        "1" * 64, "2" * 64, "3" * 64,
        tuple(subject.ProgramAuthority(
            arm, format(index + 10, "064x"), PRODUCTION_STORED_VALUES, 0, 0, True,
        ) for index, arm in enumerate(ARM_NAMES[1:])),
        False, None, None,
    )


def test_authority_cannot_enable_forward_or_smuggle_a_ruling() -> None:
    authority = _authority()
    with pytest.raises(RuntimeError, match="lacks source-bound"):
        subject.require_launch_ready(authority)
    ready = dataclasses.replace(
        authority, authorized_for_forward=True,
        inference_ruling_sha256="7" * 64, independent_audit_sha256="8" * 64,
    )
    subject.require_launch_ready(ready)
    with pytest.raises(ValueError, match="disabled authority"):
        dataclasses.replace(authority, inference_ruling_sha256="4" * 64)


def test_run_one_batch_fails_before_model_or_tokens_are_touched() -> None:
    class Poison:
        def __getattribute__(self, _name):
            raise AssertionError("NO-GO path touched model/backend")
    with pytest.raises(RuntimeError, match="lacks source-bound"):
        subject.run_one_batch(Poison(), Poison(), "native", Poison(), _authority())


def test_document_means_use_signed_arm_minus_native_and_common_counts() -> None:
    documents, cells, arms = 4, len(subject.CELL_ORDER), len(ARM_NAMES)
    counts = torch.full((documents, cells), 2, dtype=torch.int64)
    ce = torch.zeros(arms, documents, cells, dtype=torch.float64)
    ce[2] = 1.0
    stats = subject.RoleSufficientStatistics(
        "select", tuple(f"doc-{i}" for i in range(documents)), subject.CELL_ORDER, counts, ce,
        torch.zeros_like(ce), torch.zeros_like(ce), 0.0,
    )
    stats.validate()
    actual = subject.document_means(stats, ARM_NAMES[2])
    torch.testing.assert_close(actual, torch.full((documents, cells), 0.5, dtype=torch.float64))
    counts[0, 0] = 0
    stats2 = dataclasses.replace(stats, counts=counts)
    assert torch.isnan(subject.document_means(stats2, ARM_NAMES[2])[0, 0])


def test_registered_20k_shared_document_bootstrap_known_answer() -> None:
    values = torch.arange(60, dtype=torch.float64).reshape(20, 3)
    point, replicates, critical = subject.bootstrap_means(values)
    torch.testing.assert_close(point, values.mean(0), rtol=0, atol=0)
    assert replicates.shape == (20_000, 3)
    assert critical > 0 and torch.isfinite(replicates).all()
    point2, replicates2, critical2 = subject.bootstrap_means(values)
    assert torch.equal(point, point2) and torch.equal(replicates, replicates2)
    assert critical == critical2


def test_program_authority_price_and_order_fail_closed() -> None:
    with pytest.raises(ValueError, match="price/call"):
        subject.ProgramAuthority(ARM_NAMES[1], "a" * 64, 1, 0, 0, True)
    with pytest.raises(ValueError, match="programs"):
        dataclasses.replace(_authority(), programs=_authority().programs[::-1])


def test_derangement_realization_is_exact_authority_bound_control() -> None:
    permutation = torch.roll(torch.arange(128), -1).contiguous()
    authority = dataclasses.replace(
        _authority(), derangement_sha256=canary.tensor_sha256(permutation),
    )
    subject.validate_derangement_realization(permutation, authority)
    with pytest.raises(RuntimeError, match="differs from authority"):
        subject.validate_derangement_realization(torch.roll(permutation, -1), authority)
    with pytest.raises(ValueError, match=r"CPU int64\[128\]"):
        subject.validate_derangement_realization(permutation.to(torch.int32), authority)


def _masks(documents: int = 8) -> BracketMasks:
    shape = (documents, 256)
    cells = [torch.zeros(shape, dtype=torch.bool) for _ in range(5)]
    for document in range(documents):
        cells[document % 5][document, 64 + document] = True
    family = torch.full(shape, -1, dtype=torch.int16)
    family[cells[0]] = torch.arange(documents, dtype=torch.int16)[
        torch.nonzero(cells[0], as_tuple=True)[0]
    ] % 2
    family[cells[1] | cells[2]] = 0
    depth = torch.zeros(shape, dtype=torch.int16)
    distance = torch.zeros(shape, dtype=torch.int16)
    depth[cells[0] | cells[1]] = 1; distance[cells[0] | cells[1]] = 1
    domain = torch.empty(shape, dtype=torch.int8)
    domain[:documents // 2] = 0; domain[documents // 2:] = 1
    return BracketMasks(*cells, family, depth, distance, domain)


def test_coordinate_registry_preserves_domain_and_family_score_only_masks() -> None:
    coordinates = subject.score_coordinate_masks(_masks(10), ("round", "square"))
    assert tuple(coordinates) == (
        *(f"prose:{cell}" for cell in subject.CELL_ORDER),
        "prose:family:round:compatible_closer",
        "prose:family:square:compatible_closer",
        *(f"code:{cell}" for cell in subject.CELL_ORDER),
        "code:family:round:compatible_closer",
        "code:family:square:compatible_closer",
    )
    assert all(value.dtype == torch.bool and value.device.type == "cpu"
               for value in coordinates.values())


def _known_score_stats(role: str, *, weak_ood: bool = False) -> subject.RoleSufficientStatistics:
    families = ("round", "square")
    names = (
        *(f"prose:{cell}" for cell in subject.CELL_ORDER),
        "prose:family:round:compatible_closer", "prose:family:square:compatible_closer",
        *(f"code:{cell}" for cell in subject.CELL_ORDER),
        "code:family:round:compatible_closer", "code:family:square:compatible_closer",
    )
    documents = 40
    counts = torch.ones(documents, len(names), dtype=torch.int64)
    ce = torch.ones(len(ARM_NAMES), documents, len(names), dtype=torch.float64)
    for domain in ("prose", "code"):
        compatible = names.index(f"{domain}:compatible_closer")
        ce[2, :, compatible] += 0.2
        ce[3, :, compatible] += 0.15
        if weak_ood:
            ce[1, :, compatible] += 0.15
        for cell in subject.CELL_ORDER[1:5]:
            ce[2, :, names.index(f"{domain}:{cell}")] += 0.01
        ce[2, :, names.index(f"{domain}:all")] += 0.005
        for family in families:
            ce[2, :, names.index(f"{domain}:family:{family}:compatible_closer")] += 0.2
    zeros = torch.zeros_like(ce)
    result = subject.RoleSufficientStatistics(
        role, tuple(f"{role}-doc-{i}" for i in range(documents)), names,
        counts, ce, zeros, zeros, 0.0,
    )
    result.validate(); return result


def test_score_recomputes_nonlinear_gates_and_conjoins_ood_retention() -> None:
    integrity = subject.ExecutionIntegrity(*(True for _ in range(7)))
    passed = subject.score_roles(
        _known_score_stats("select"), _known_score_stats("ood"),
        integrity, ("round", "square"),
    )
    assert passed.promoted and all(dict(passed.decisions).values())
    assert "simplification" in passed.to_payload()["claim_boundary"]
    failed = subject.score_roles(
        _known_score_stats("select"), _known_score_stats("ood", weak_ood=True),
        integrity, ("round", "square"),
    )
    assert not failed.promoted and not dict(failed.decisions)["ood"]


def test_score_rejects_nonpositive_bootstrap_denominator_and_false_integrity() -> None:
    select, ood = _known_score_stats("select"), _known_score_stats("ood")
    false_integrity = dataclasses.replace(
        subject.ExecutionIntegrity(*(True for _ in range(7))), finite_outputs=False,
    )
    result = subject.score_roles(select, ood, false_integrity, ("round", "square"))
    assert not result.promoted and not dict(result.decisions)["common_integrity"]
    ce = ood.ce_sums.clone()
    column = ood.coordinate_names.index("prose:compatible_closer")
    ce[2, :, column] = ce[0, :, column]
    with pytest.raises(RuntimeError, match="nonpositive deletion"):
        subject.score_roles(
            select, dataclasses.replace(ood, ce_sums=ce),
            subject.ExecutionIntegrity(*(True for _ in range(7))), ("round", "square"),
        )
