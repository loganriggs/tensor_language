import pytest
import torch

import finite_horizon_tangent_realization as realization
import finite_horizon_tangent_realization_proof as proof


def fixture():
    return proof.time_varying_fixture()


def test_cut_rank_removes_unreachable_unobservable_state():
    blocks, inputs, outputs = fixture()
    analyses = realization.analyze_all_cuts(blocks, inputs, outputs, (1, 2, 3))
    assert {row["exact_cut_rank"] for row in analyses.values()} == {2}


def test_rank2_factorization_is_exact_and_rank1_tail_is_positive():
    blocks, inputs, outputs = fixture()
    matrix, _ = realization.assemble_cut(blocks, inputs, outputs, 2)
    left, right, receipt = realization.truncated_factorization(matrix, 2)
    assert torch.allclose(left @ right, matrix, atol=1e-12, rtol=1e-12)
    assert receipt["maximum_absolute_error"] < 1e-12
    analysis = realization.analyze_cut(matrix)
    assert analysis["optimal_squared_frobenius_tail_by_rank"][1] > 0
    assert analysis["optimal_squared_frobenius_tail_by_rank"][2] < 1e-20


def test_independent_orthogonal_gauges_preserve_every_cut_spectrum():
    blocks, inputs, outputs = fixture()
    gauges_in = {site: proof.orthogonal(width, 10 + site) for site, width in inputs.items()}
    gauges_out = {site: proof.orthogonal(width, 20 + site) for site, width in outputs.items()}
    transformed = realization.transform_blocks_orthogonal(blocks, gauges_in, gauges_out)
    for cut in (1, 2, 3):
        original, _ = realization.assemble_cut(blocks, inputs, outputs, cut)
        replay, _ = realization.assemble_cut(transformed, inputs, outputs, cut)
        assert torch.allclose(
            torch.linalg.svdvals(original), torch.linalg.svdvals(replay),
            atol=1e-12, rtol=1e-12,
        )


def test_missing_block_fails_closed_instead_of_becoming_zero():
    blocks, inputs, outputs = fixture()
    del blocks[(3, 0)]
    with pytest.raises(ValueError, match="missing explicit response block"):
        realization.assemble_cut(blocks, inputs, outputs, 2)


def test_nonorthogonal_gauge_is_rejected():
    blocks, inputs, outputs = fixture()
    gauges_in = {site: torch.eye(width, dtype=torch.float64) for site, width in inputs.items()}
    gauges_out = {site: torch.eye(width, dtype=torch.float64) for site, width in outputs.items()}
    gauges_in[0][0, 0] = 2.0
    with pytest.raises(ValueError, match="orthogonal"):
        realization.transform_blocks_orthogonal(blocks, gauges_in, gauges_out)


def test_energy_gap_rule_does_not_call_full_support_compression():
    matrix = torch.diag(torch.tensor([3.0, 2.0], dtype=torch.float64))
    result = realization.analyze_cut(matrix, energy_fraction=1.0)
    assert result["selected_rank"] is None
    assert result["certified_compression_knee"] is False


def test_split_comparison_uses_common_right_state_and_passes_left_row_rotation():
    generator = torch.Generator().manual_seed(31)
    right = torch.randn(2, 3, generator=generator, dtype=torch.float64)
    left_a, _ = torch.linalg.qr(torch.randn(6, 2, generator=generator, dtype=torch.float64))
    left_b, _ = torch.linalg.qr(torch.randn(6, 2, generator=generator, dtype=torch.float64))
    singular = torch.diag(torch.tensor([5.0, 1.0], dtype=torch.float64))
    primary = {(3, 0): left_a @ singular @ right}
    replication = {(3, 0): left_b @ singular @ right}
    result = realization.compare_split_cuts(
        primary, replication, {0: 3}, {3: 6}, (1,),
        energy_fraction=0.90, gap_ratio=2.0,
    )["1"]
    assert result["passes"] is True
    assert result["combined_selected_rank"] == 1
    assert result["normalized_right_projector_chordal_distance"] < 1e-12


def test_split_comparison_rejects_incompatible_right_state():
    primary = {(3, 0): torch.diag(torch.tensor([5.0, 1.0, 0.1], dtype=torch.float64))}
    replication = {(3, 0): torch.diag(torch.tensor([0.1, 1.0, 5.0], dtype=torch.float64))}
    result = realization.compare_split_cuts(
        primary, replication, {0: 3}, {3: 3}, (1,),
        energy_fraction=0.90, gap_ratio=2.0,
    )["1"]
    assert result["passes"] is False
    assert result["gates"]["right_projector_stability"] is False


def test_contextwise_rank_is_not_conflated_with_stacked_shared_rank():
    # Each context has a one-dimensional encoder, but the encoder rotates between
    # contexts. The stacked shared-linear interface therefore needs dimension two.
    blocks = {(3, 0): torch.tensor([
        [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [0.0, 3.0],
    ], dtype=torch.float64)}
    contextwise = realization.analyze_contextwise_cuts(
        blocks, {0: 2}, {3: 4}, (1,), probes_per_context=2,
    )["1"]
    stacked, _ = realization.assemble_cut(blocks, {0: 2}, {3: 4}, 1)
    assert contextwise["minimum_rank"] == contextwise["maximum_rank"] == 1
    assert realization.analyze_cut(stacked)["exact_cut_rank"] == 2
