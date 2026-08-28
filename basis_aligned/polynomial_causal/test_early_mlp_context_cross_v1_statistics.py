from __future__ import annotations

import pytest
import torch

import early_mlp_context_cross_v1 as cross
import early_mlp_context_cross_v1_statistics as statistics


AUTHORITY = "a" * 64
DOCUMENTS = "b" * 64


def _rank_cost(rank: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(9100 + rank)
    rows = cross.PIVOT_ROWS[rank]
    columns = cross.PIVOT_COLUMNS[rank]
    left = 0.15 * torch.randn((7, rank), generator=generator, dtype=torch.float64)
    right = 0.15 * torch.randn((rank, 7), generator=generator, dtype=torch.float64)
    left[[row - 1 for row in rows]] = torch.eye(rank, dtype=torch.float64)
    right[:, [column - 1 for column in columns]] = torch.eye(
        rank, dtype=torch.float64,
    )
    interaction = torch.zeros((8, 8), dtype=torch.float64)
    interaction[1:, 1:] = left @ right
    row_effect = torch.linspace(0.0, 0.14, 8, dtype=torch.float64)[:, None]
    column_effect = torch.linspace(0.0, 0.07, 8, dtype=torch.float64)[None, :]
    return row_effect + column_effect + interaction


def _stage(cost: torch.Tensor, stage: str, *, role: str = "skip7000") -> statistics.StageStatistics:
    cells = statistics.STAGE_CELLS[stage]
    tokens = torch.tensor([100_000, 120_000, 80_000], dtype=torch.long)
    ce_rate = torch.tensor([
        [6.0 + float(cost[cell]) for cell in cells] for _ in tokens
    ], dtype=torch.float64)
    accuracy = torch.tensor([
        [0.75 - 0.001 * float(cost[cell]) for cell in cells] for _ in tokens
    ], dtype=torch.float64)
    correct = torch.round(accuracy * tokens[:, None]).long().contiguous()
    return statistics.StageStatistics(
        role=role, stage=stage, authority_sha256=AUTHORITY,
        ordered_document_ids_sha256=DOCUMENTS,
        document_token_count=tokens.contiguous(), top1_correct=correct,
        ce_sum=(ce_rate * tokens[:, None]).contiguous(),
    )


def test_stage_statistics_are_sealed_and_capability_sized() -> None:
    stage = _stage(_rank_cost(3), "validation")
    assert stage.top1_correct.shape == (3, 7)
    clone = stage.ce_sum
    clone[0, 0] += 1.0
    assert stage.sha256 == stage.sha256
    with pytest.raises(AttributeError):
        stage._stage = "heldout"
    with pytest.raises(ValueError):
        statistics.StageStatistics(
            role="skip7000", stage="validation", authority_sha256=AUTHORITY,
            ordered_document_ids_sha256=DOCUMENTS,
            document_token_count=torch.ones(3, dtype=torch.long),
            top1_correct=torch.zeros((3, 8), dtype=torch.long),
            ce_sum=torch.zeros((3, 8), dtype=torch.float64),
        )


def test_bootstrap_uses_one_token_weighted_multiplicity_vector() -> None:
    cost = _rank_cost(3)
    discovery = _stage(cost, "discovery")
    weights = statistics.bootstrap_multiplicities("skip7000", 3)
    draws = statistics.point_and_bootstrap_costs((discovery,), "ce_nats")
    cell = cross.RANK3_DISCOVERY_CELLS[-1]
    column = cross.RANK3_DISCOVERY_CELLS.index(cell)
    anchor_column = cross.RANK3_DISCOVERY_CELLS.index((0, 0))
    multiplicity = weights[0]
    denominator = multiplicity @ discovery.document_token_count.double()
    manual = (
        multiplicity @ discovery.ce_sum[:, column] / denominator
        - multiplicity @ discovery.ce_sum[:, anchor_column] / denominator
    )
    assert draws.shape == (cross.BOOTSTRAP_DRAWS + 1, 8, 8)
    assert torch.equal(weights.sum(1), torch.full(
        (cross.BOOTSTRAP_DRAWS,), 3.0, dtype=torch.float64,
    ))
    assert float(draws[1][cell]) == pytest.approx(float(manual), abs=1e-12)


@pytest.mark.parametrize("rank", [3, 4])
def test_batched_cross_recovers_exact_registered_rank(rank: int) -> None:
    cost = _rank_cost(rank)
    fit = cross.RANK3_DISCOVERY_CELLS if rank == 3 else cross.RANK4_FIT_CELLS
    licensed = torch.full((2, 8, 8), float("nan"), dtype=torch.float64)
    for cell in fit:
        licensed[:, cell[0], cell[1]] = torch.tensor(
            [cost[cell], 2.0 * cost[cell]], dtype=torch.float64,
        )
    prediction, condition, singular = statistics.batched_cross_prediction(
        licensed, rank,
    )
    assert not bool(singular.any())
    assert bool((condition < 20.0).all())
    assert torch.allclose(prediction[0], cost, atol=1e-10, rtol=1e-10)
    assert torch.allclose(prediction[1], 2.0 * cost, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("rank", [3, 4])
def test_als_is_scale_equivariant_under_registered_normalization(rank: int) -> None:
    cost = _rank_cost(rank)
    fit = cross.RANK3_DISCOVERY_CELLS if rank == 3 else cross.RANK4_FIT_CELLS
    licensed = torch.full((2, 8, 8), float("nan"), dtype=torch.float64)
    for cell in fit:
        licensed[0][cell] = cost[cell]
        licensed[1][cell] = 7.0 * cost[cell]
    prediction, failed = statistics.batched_als_prediction(licensed, rank)
    assert not bool(failed.any())
    assert torch.allclose(prediction[1], 7.0 * prediction[0], atol=1e-9, rtol=1e-9)


def test_rank_three_api_has_no_heldout_capability() -> None:
    cost = _rank_cost(3)
    discovery = _stage(cost, "discovery")
    validation = _stage(cost, "validation")
    heldout = _stage(cost, "heldout")
    with pytest.raises(ValueError, match="rank-three"):
        statistics.score_rank(discovery, validation, heldout, rank=3)
    with pytest.raises(RuntimeError, match="one role"):
        statistics.score_rank(
            discovery, _stage(cost, "validation", role="skip11000"), None,
            rank=3,
        )


def test_end_to_end_rank_three_score_on_exact_state(monkeypatch: pytest.MonkeyPatch) -> None:
    # The production constant is separately frozen at 2,000.  A three-draw replay
    # exercises the identical staging/gate path without adding a minute to unit tests.
    monkeypatch.setattr(cross, "BOOTSTRAP_DRAWS", 3)
    cost = _rank_cost(3)
    result = statistics.score_rank(
        _stage(cost, "discovery"), _stage(cost, "validation"), None, rank=3,
    )
    assert result["ce_useful_pass"]
    assert result["targets"]["ce_nats"]["rmse"]["point"] < 1e-10
    assert result["targets"]["ce_nats"]["singular_or_zero_rms_draw_count"] == 0
