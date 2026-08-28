from __future__ import annotations

import math

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


def _reference_als_one(cost: torch.Tensor, rank: int) -> torch.Tensor:
    """Scalar/restart-loop reference, independent of the batched implementation."""

    entries = cross.cross_cells(rank)
    target = torch.tensor([
        float(cost[i, j] - cost[i, 0] - cost[0, j] + cost[0, 0])
        for i, j in entries
    ], dtype=torch.float64)
    scale = target.square().mean().sqrt()
    normalized = target / scale
    index = {cell: ordinal for ordinal, cell in enumerate(entries)}
    penalty = len(entries) * cross.ALS_RELATIVE_RIDGE / (7 * rank)
    identity = torch.eye(rank, dtype=torch.float64)
    candidates = []
    objectives = []
    for restart in range(cross.ALS_RESTARTS):
        generator = torch.Generator().manual_seed(
            cross.ALS_SEED + 1000 * rank + restart
        )
        left = torch.randn(
            (7, rank), generator=generator, dtype=torch.float64,
        ) / math.sqrt(rank)
        right = torch.randn(
            (7, rank), generator=generator, dtype=torch.float64,
        ) / math.sqrt(rank)
        for _ in range(100):
            updated_rows = []
            for i in range(1, 8):
                columns = [j for j in range(1, 8) if (i, j) in index]
                design = right[[j - 1 for j in columns]]
                response = normalized[[index[(i, j)] for j in columns]]
                updated_rows.append(torch.linalg.solve(
                    design.T @ design + penalty * identity,
                    design.T @ response,
                ))
            left = torch.stack(updated_rows)
            updated_columns = []
            for j in range(1, 8):
                rows = [i for i in range(1, 8) if (i, j) in index]
                design = left[[i - 1 for i in rows]]
                response = normalized[[index[(i, j)] for i in rows]]
                updated_columns.append(torch.linalg.solve(
                    design.T @ design + penalty * identity,
                    design.T @ response,
                ))
            right = torch.stack(updated_columns)
        fitted = torch.tensor([
            float(left[i - 1] @ right[j - 1]) for i, j in entries
        ], dtype=torch.float64)
        objective = (fitted - normalized).square().mean() + (
            cross.ALS_RELATIVE_RIDGE
            * (left.square().mean() + right.square().mean())
        )
        candidates.append(left @ right.T)
        objectives.append(objective)
    # Python min keeps the first occurrence, matching torch.argmin's frozen tie rule.
    selected = min(range(len(objectives)), key=lambda value: float(objectives[value]))
    total = torch.stack([
        torch.stack([
            cost[i, 0] + cost[0, j] - cost[0, 0] for j in range(8)
        ]) for i in range(8)
    ])
    total[1:, 1:] += scale * candidates[selected]
    return total


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
    ce = stage.ce_sum
    ce[0, 0] = float("nan")
    with pytest.raises(ValueError, match="bounds"):
        statistics.StageStatistics(
            role="skip7000", stage="validation", authority_sha256=AUTHORITY,
            ordered_document_ids_sha256=DOCUMENTS,
            document_token_count=stage.document_token_count,
            top1_correct=stage.top1_correct, ce_sum=ce.contiguous(),
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


def test_als_matches_independent_scalar_known_answer_and_exact_sweep_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = 3
    cost = _rank_cost(rank)
    # Add a deterministic off-rank perturbation so restart selection and update
    # order matter rather than every restart sharing an exact zero-residual answer.
    for i, j in cross.cross_cells(rank):
        cost[i, j] += 0.007 * math.sin(11 * i + 7 * j)
    licensed = torch.full((1, 8, 8), float("nan"), dtype=torch.float64)
    for cell in cross.RANK3_DISCOVERY_CELLS:
        licensed[0][cell] = cost[cell]
    reference = _reference_als_one(cost, rank)
    calls = 0
    original = torch.linalg.solve

    def counted_solve(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve", counted_solve)
    observed, failed = statistics.batched_als_prediction(licensed, rank)
    assert not bool(failed[0])
    assert calls == cross.ALS_SWEEPS * 14
    assert torch.allclose(observed[0], reference, atol=1e-11, rtol=1e-11)


def test_als_restart_ties_select_the_first_seeded_restart() -> None:
    objective = torch.tensor([
        [2.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float64)
    assert statistics._select_restart(objective).tolist() == [1, 0]


def test_type7_quantiles_have_a_known_answer() -> None:
    values = torch.cat((
        torch.tensor([-1.0], dtype=torch.float64),
        torch.arange(cross.BOOTSTRAP_DRAWS, dtype=torch.float64),
    ))
    summary = statistics._summary(values)
    assert summary == pytest.approx({
        "point": -1.0, "q025": 49.975, "q95": 1899.05, "q975": 1949.025,
    }, abs=1e-12)


def test_singular_draw_is_retained_and_never_replaced() -> None:
    rank = 3
    cost = _rank_cost(rank)
    additive = torch.stack([
        torch.stack([
            cost[i, 0] + cost[0, j] - cost[0, 0] for j in range(8)
        ]) for i in range(8)
    ])
    licensed = torch.full((2, 8, 8), float("nan"), dtype=torch.float64)
    for cell in cross.RANK3_DISCOVERY_CELLS:
        licensed[0][cell] = cost[cell]
        licensed[1][cell] = additive[cell]
    prediction, _, singular = statistics.batched_cross_prediction(licensed, rank)
    assert singular.tolist() == [False, True]
    assert bool(torch.isnan(prediction[1]).all())


def test_zero_metric_denominators_are_retained_as_nonfinite_hard_failures() -> None:
    additive = torch.stack([
        torch.stack([
            torch.tensor(float(i + j), dtype=torch.float64) for j in range(8)
        ]) for i in range(8)
    ])
    cost = additive.repeat(4, 1, 1).contiguous()
    metrics = statistics._metric_vectors(
        cost, cost.clone(), cost.clone(), cross.RANK4_VALIDATION_CELLS,
    )
    # Exactly additive truth gives zero interaction and additive-error
    # denominators. They remain NaN instead of being replaced by 0, 1, epsilon,
    # or a dropped draw; score_rank's finite-every-draw branch therefore fails.
    assert bool(torch.isnan(metrics["interaction_nre"]).all())
    assert bool(torch.isnan(metrics["rmse_over_additive"]).all())
    assert any(not bool(torch.isfinite(value).all()) for value in metrics.values())


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
