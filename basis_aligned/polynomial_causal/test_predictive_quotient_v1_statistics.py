from __future__ import annotations

import hashlib

import pytest
import torch

import predictive_quotient_v1_statistics as statistics


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_ephemeral_sketches_reduce_to_exact_row_and_assigned_outer_sums() -> None:
    sketches = torch.zeros(
        statistics.PROBES_PER_BANK, statistics.BATCH_ROWS,
        statistics.SCORED_POSITIONS, statistics.CODE_DIM,
    )
    for row in range(statistics.BATCH_ROWS):
        sketches[:, row, :, row] = row + 1
    targets = torch.zeros(
        statistics.PROBES_PER_BANK, statistics.BATCH_ROWS,
        statistics.SCORED_POSITIONS, dtype=torch.long,
    )
    positions = torch.tensor([64, 65, 66, 67])
    direction_indices = torch.tensor([0, 1, 2, 3])
    summary = statistics.summarize_fisher_batch(
        sketches, targets, bank="primary", batch_ordinal=0,
        source_identity_sha256=sha("identity"), assigned_positions=positions,
        assigned_direction_indices=direction_indices,
    )
    assert summary.sketch_count == 8 * 4 * 192
    expected_row = torch.zeros(4, 64, 64, dtype=torch.float64)
    expected_assigned = torch.zeros_like(expected_row)
    for row in range(4):
        expected_row[row, row, row] = 8 * 192 * (row + 1) ** 2
        expected_assigned[row, row, row] = 8 * (row + 1) ** 2
    torch.testing.assert_close(summary.row_outer_product_sums, expected_row)
    torch.testing.assert_close(summary.assigned_position_outer_sums, expected_assigned)
    torch.testing.assert_close(summary.outer_product_sum, expected_row.sum(0))
    assert len(summary.sha256) == 64 and len(summary.target_ids_sha256) == 64


def fake_summary(bank: str, ordinal: int, identity: str) -> statistics.FisherBatchSummary:
    rows = torch.zeros(4, 64, 64, dtype=torch.float64)
    assigned = torch.zeros_like(rows)
    for row in range(4):
        rows[row, row, row] = ordinal + row + 1
        assigned[row, row, row] = ordinal + row + 0.5
    return statistics.FisherBatchSummary(
        bank=bank, probe_seeds=statistics.PROBE_SEEDS[bank],
        batch_ordinal=ordinal,
        ordered_row_indices=tuple(range(ordinal * 4, ordinal * 4 + 4)),
        source_identity_sha256=identity, target_ids_sha256=sha(f"target-{bank}-{ordinal}"),
        sketch_count=statistics.SKETCHES_PER_BATCH,
        row_sketch_count=statistics.PROBES_PER_BANK * statistics.SCORED_POSITIONS,
        outer_product_sum=rows.sum(0), row_outer_product_sums=rows,
        assigned_position_outer_sums=assigned,
        assigned_positions=torch.arange(4) + 64 + ordinal % 180,
        assigned_direction_indices=(torch.arange(4) + ordinal) % 32,
    )


def identity_plan() -> dict[str, tuple[str, ...]]:
    shared = tuple(sha(f"identity-{ordinal}") for ordinal in range(48))
    return {
        bank: shared
        for bank in statistics.PROBE_SEEDS
    }


def test_collector_requires_every_planned_summary_and_preserves_row_statistics() -> None:
    plan = identity_plan()
    collector = statistics.FisherStatisticsCollector(
        common_support_sha256=sha("support"),
        expected_source_identity_sha256=plan,
    )
    for ordinal in reversed(range(48)):
        for bank in reversed(tuple(statistics.PROBE_SEEDS)):
            collector.add(fake_summary(bank, ordinal, plan[bank][ordinal]))
    result = collector.finalize()
    assert result.count_per_bank == 8 * 192 * 192
    assert result.row_outer_product_sums["primary"].shape == (192, 64, 64)
    assert result.assigned_position_outer_sums["replication"].shape == (192, 64, 64)
    expected = result.row_outer_product_sums["primary"].sum(0) / result.count_per_bank
    torch.testing.assert_close(result.observability("primary"), expected)
    directions = torch.eye(64, dtype=torch.float64)[:4].repeat(48, 1)
    response = result.assigned_quadratic_response("primary", directions)
    assert response.shape == (192,) and bool((response >= 0).all())


def test_collector_and_batch_summary_fail_closed_on_replay_or_assignment_drift() -> None:
    plan = identity_plan()
    collector = statistics.FisherStatisticsCollector(
        common_support_sha256=sha("support"),
        expected_source_identity_sha256=plan,
    )
    summary = fake_summary("primary", 0, plan["primary"][0])
    collector.add(summary)
    with pytest.raises(RuntimeError, match="duplicated"):
        collector.add(summary)
    with pytest.raises(RuntimeError, match="missing"):
        collector.finalize()

    wrong = fake_summary("primary", 1, plan["primary"][1])
    with pytest.raises(ValueError, match="intervention assignment"):
        statistics.FisherBatchSummary(
            **{
                **wrong.__dict__,
                "assigned_positions": torch.tensor([63, 64, 65, 66]),
            }
        )


def test_collector_rejects_probe_banks_from_different_source_plans() -> None:
    plan = identity_plan()
    plan["replication"] = tuple(
        sha(f"different-source-{ordinal}") for ordinal in range(48)
    )
    with pytest.raises(ValueError, match="share one source identity"):
        statistics.FisherStatisticsCollector(
            common_support_sha256=sha("support"),
            expected_source_identity_sha256=plan,
        )


def test_collector_detects_tensor_mutation_after_admission() -> None:
    plan = identity_plan()
    collector = statistics.FisherStatisticsCollector(
        common_support_sha256=sha("support"),
        expected_source_identity_sha256=plan,
    )
    admitted = None
    for ordinal in range(48):
        for bank in statistics.PROBE_SEEDS:
            summary = fake_summary(bank, ordinal, plan[bank][ordinal])
            collector.add(summary)
            if bank == "primary" and ordinal == 0:
                admitted = summary
    assert admitted is not None
    admitted.row_outer_product_sums[0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="mutated"):
        collector.finalize()
