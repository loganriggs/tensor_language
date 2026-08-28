from __future__ import annotations

import pytest
import torch

import finite_horizon_tangent_response_bank as response_bank


def plan() -> response_bank.TangentResponsePlan:
    documents = ("doc-a", "doc-a", "doc-b", "doc-c")
    splits = response_bank.allocate_whole_document_splits(documents)
    return response_bank.TangentResponsePlan(
        experiment_id="synthetic-tangent-bank-v1",
        row_artifact_sha256="a" * 64,
        row_ids=("r0", "r1", "r2", "r3"),
        document_ids=documents,
        splits=splits,
        scored_positions=(64, 65, 66, 67),
        input_dims=((0, 2), (1, 2), (2, 2)),
        target_site=3,
        probes_per_row=3,
        direction_seed=11,
        probe_seed=12,
        position_seed=13,
    )


def responses(offset: float = 0.0) -> dict[int, torch.Tensor]:
    base = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    return {site: base * (site + 1.0) + offset for site in (0, 1, 2)}


def complete_transaction() -> response_bank.TangentResponseBankTransaction:
    transaction = response_bank.TangentResponseBankTransaction(plan())
    for index, row_id in enumerate(plan().row_ids):
        transaction.add_row(row_id, responses(float(index)))
    return transaction


def test_whole_document_allocator_never_leaks_and_balances_rows() -> None:
    documents = ("large",) * 4 + ("medium",) * 2 + ("small-a", "small-b")
    splits = response_bank.allocate_whole_document_splits(documents)
    assignment = {}
    for document, split in zip(documents, splits, strict=True):
        assert assignment.setdefault(document, split) == split
    assert abs(splits.count("primary") - splits.count("replication")) <= 2


def test_seal_builds_complete_document_disjoint_split_operators() -> None:
    transaction = complete_transaction()
    bank = transaction.seal()
    assert transaction.closed and transaction.aliases_revoked
    assert bank.receipt["every_direction_evaluated_on_every_row"] is True
    assert bank.receipt["whole_document_splits"] is True
    for split, blocks in bank.split_blocks.items():
        rows = bank.receipt["splits"][split]["rows"]
        assert set(blocks) == {(3, 0), (3, 1), (3, 2)}
        assert all(block.shape == (rows * 3, 2) for block in blocks.values())
    analyses = response_bank.analyze_bank(bank, (1, 2, 3))
    assert set(analyses) == {"primary", "replication"}
    assert all(set(value) == {"1", "2", "3"} for value in analyses.values())


def test_input_alias_mutation_cannot_change_sealed_bank() -> None:
    p = plan()
    transaction = response_bank.TangentResponseBankTransaction(p)
    first = responses()
    transaction.add_row("r0", first)
    first[0].fill_(999.0)
    for index, row_id in enumerate(p.row_ids[1:], start=1):
        transaction.add_row(row_id, responses(float(index)))
    bank = transaction.seal()
    split = p.splits[0]
    assert not bool((bank.split_blocks[split][(3, 0)] == 999.0).any())


def test_incomplete_seal_emits_nothing_and_spends_transaction() -> None:
    transaction = response_bank.TangentResponseBankTransaction(plan())
    transaction.add_row("r0", responses())
    with pytest.raises(RuntimeError, match="incomplete"):
        transaction.seal()
    assert transaction.aliases_revoked
    with pytest.raises(RuntimeError, match="spent"):
        transaction.seal()


def test_duplicate_unregistered_and_missing_source_rows_fail_closed() -> None:
    transaction = response_bank.TangentResponseBankTransaction(plan())
    transaction.add_row("r0", responses())
    with pytest.raises(ValueError, match="duplicate"):
        transaction.add_row("r0", responses())
    with pytest.raises(ValueError, match="not registered"):
        transaction.add_row("unknown", responses())
    with pytest.raises(ValueError, match="every and only"):
        transaction.add_row("r1", {0: responses()[0]})


@pytest.mark.parametrize("bad", [
    torch.ones(3, 2, dtype=torch.float32),
    torch.ones(3, 2, dtype=torch.float64, requires_grad=True),
    torch.full((3, 2), float("nan"), dtype=torch.float64),
    torch.ones(2, 2, dtype=torch.float64),
])
def test_dtype_graph_finiteness_and_shape_are_enforced(bad: torch.Tensor) -> None:
    candidate = responses()
    candidate[0] = bad
    with pytest.raises(ValueError):
        response_bank.TangentResponseBankTransaction(plan()).add_row("r0", candidate)


def test_document_crossing_is_rejected_by_plan() -> None:
    with pytest.raises(ValueError, match="crosses"):
        response_bank.TangentResponsePlan(
            experiment_id="bad", row_artifact_sha256="b" * 64,
            row_ids=("a", "b"), document_ids=("same", "same"),
            splits=("primary", "replication"), input_dims=((0, 2),),
            scored_positions=(64, 65),
            target_site=1, probes_per_row=2, direction_seed=1, probe_seed=2,
            position_seed=3,
        )
