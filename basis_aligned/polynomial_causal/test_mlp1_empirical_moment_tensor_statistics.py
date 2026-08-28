from __future__ import annotations

import math

import pytest
import torch

from mlp1_empirical_moment_tensor_statistics import (
    BilinearFactors,
    DocumentProbeGramAccumulator,
    PopulationMoments,
    ProbeGramAccumulator,
    StreamingPopulationMoments,
    average_ranks,
    bilinear_output,
    build_population_projectors,
    deterministic_haar_basis,
    evaluate_probe_bank,
    noncentral_gaussian_cross_inner_product,
    noncentral_gaussian_probe_gram,
    simultaneous_document_bootstrap,
    spearman_average_rank,
    teacher_minus_candidate_factors,
)


def _factors(seed: int, *, output: int = 3, gates: int = 4, dimension: int = 3):
    generator = torch.Generator().manual_seed(seed)
    return BilinearFactors(
        down=torch.randn(output, gates, generator=generator, dtype=torch.float64),
        left=torch.randn(gates, dimension, generator=generator, dtype=torch.float64),
        right=torch.randn(gates, dimension, generator=generator, dtype=torch.float64),
    )


def test_streaming_population_moments_match_fixed_order_direct_and_merge():
    generator = torch.Generator().manual_seed(1717)
    rows = torch.randn(29, 5, generator=generator, dtype=torch.float64) * torch.tensor(
        [1e-3, 1.0, 5.0, 100.0, 1e3], dtype=torch.float64,
    ) + torch.tensor([1e7, -3.0, 0.2, 20.0, -1e5], dtype=torch.float64)
    direct_mean = rows.mean(dim=0)
    direct_covariance = (rows - direct_mean).T @ (rows - direct_mean) / rows.shape[0]

    streamed = StreamingPopulationMoments(5)
    for start, stop in ((0, 1), (1, 7), (7, 18), (18, 29)):
        streamed.update(rows[start:stop])
    result = streamed.finalize()
    torch.testing.assert_close(result.mean, direct_mean, rtol=1e-13, atol=1e-9)
    # Fixed-order Chan and one-shot centering differ slightly under the deliberately
    # ill-scaled mean; both remain close to far below float32 capture precision.
    torch.testing.assert_close(result.covariance, direct_covariance, rtol=2e-7, atol=5e-8)
    expected_rms = torch.sqrt(torch.mean(torch.sum(rows.square(), dim=1)))
    assert result.input_rms == pytest.approx(float(expected_rms), rel=2e-15)

    left = StreamingPopulationMoments(5)
    right = StreamingPopulationMoments(5)
    left.update(rows[:13])
    right.update(rows[13:])
    left.merge(right.finalize())
    merged = left.finalize()
    torch.testing.assert_close(merged.mean, result.mean, rtol=1e-15, atol=1e-9)
    torch.testing.assert_close(merged.covariance, result.covariance, rtol=2e-7, atol=5e-8)
    with pytest.raises(ValueError, match="empty"):
        StreamingPopulationMoments(2).finalize()


def test_population_pca_mean_handling_signs_projectors_and_degenerate_boundary():
    signs = torch.tensor(
        [[a, b, c] for a in (-1.0, 1.0) for b in (-1.0, 1.0) for c in (-1.0, 1.0)],
        dtype=torch.float64,
    )
    rows = torch.cat(
        (
            torch.full((8, 1), 10.0, dtype=torch.float64),
            signs * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
        ),
        dim=1,
    )
    accumulator = StreamingPopulationMoments(4)
    accumulator.update(rows)
    projectors = build_population_projectors(accumulator.finalize(), (1, 2, 3, 4))
    assert projectors.mean_present
    assert projectors.mean_ratio > 1e-8
    torch.testing.assert_close(
        projectors.eigenvalues, torch.tensor([9.0, 4.0, 1.0, 0.0], dtype=torch.float64),
    )
    expected_pca_one = torch.diag(torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64))
    expected_mean_one = torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(projectors.pca_no_mean.at_rank(1)[1], expected_pca_one)
    torch.testing.assert_close(projectors.mean_plus_pca.at_rank(1)[1], expected_mean_one)
    mean_two = projectors.mean_plus_pca.at_rank(2)[1]
    torch.testing.assert_close(
        mean_two,
        torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)),
    )
    assert all(
        float(projectors.eigenvectors[int(torch.argmax(torch.abs(projectors.eigenvectors[:, j]))), j])
        >= 0.0
        for j in range(4)
    )

    tied = PopulationMoments(
        count=2,
        mean=torch.zeros(3, dtype=torch.float64),
        centered_outer_sum=2.0 * torch.eye(3, dtype=torch.float64),
    )
    tied_projectors = build_population_projectors(tied, (1, 2))
    assert tied_projectors.degenerate_boundaries == (True, True)
    assert not tied_projectors.mean_present
    torch.testing.assert_close(
        tied_projectors.mean_plus_pca.at_rank(2)[1],
        tied_projectors.pca_no_mean.at_rank(2)[1],
    )


def test_deterministic_pcg64dxsm_haar_is_repeatable_and_mean_orthogonal():
    mean = torch.tensor([1.0, -2.0, 0.0, 1.0, 0.5], dtype=torch.float64)
    first = deterministic_haar_basis(5, 3, seed=908, mean_direction=mean)
    second = deterministic_haar_basis(5, 3, seed=908, mean_direction=mean)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.T @ first, torch.eye(3, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(
        first.T @ (mean / torch.linalg.norm(mean)),
        torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0,
    )
    different = deterministic_haar_basis(5, 3, seed=909, mean_direction=mean)
    assert not torch.equal(first, different)


def test_bias_free_float64_factors_and_signed_teacher_minus_candidate_residual():
    teacher = _factors(1)
    candidate = _factors(2)
    rows = torch.randn(7, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float32)
    residual = teacher_minus_candidate_factors(teacher, candidate)
    expected = bilinear_output(teacher, rows) - bilinear_output(candidate, rows)
    torch.testing.assert_close(bilinear_output(residual, rows), expected, rtol=2e-14, atol=2e-14)
    opposite = teacher_minus_candidate_factors(candidate, teacher)
    torch.testing.assert_close(bilinear_output(opposite, rows), -expected, rtol=2e-14, atol=2e-14)
    with pytest.raises(ValueError, match="must be float64"):
        BilinearFactors(
            down=teacher.down.float(), left=teacher.left, right=teacher.right,
        )


def test_direct_probe_gram_streaming_and_document_sufficient_statistics():
    probes = [_factors(11), _factors(12), _factors(13)]
    rows = torch.randn(9, 3, generator=torch.Generator().manual_seed(14), dtype=torch.float64)
    outputs = evaluate_probe_bank(probes, rows)
    expected = torch.einsum("npo,nqo->pq", outputs, outputs) / rows.shape[0]
    accumulator = ProbeGramAccumulator(3)
    accumulator.update(outputs[:2])
    accumulator.update(outputs[2:7])
    accumulator.update(outputs[7:])
    torch.testing.assert_close(accumulator.finalize(), expected)

    ids = ("a", "a", "b", "a", "c", "c", "b", "c", "c")
    documents = DocumentProbeGramAccumulator(3)
    documents.update(outputs[:4], ids[:4])
    documents.update(outputs[4:], ids[4:])
    summary = documents.finalize()
    assert summary.document_ids == ("a", "b", "c")
    assert summary.row_counts.tolist() == [3, 2, 4]
    torch.testing.assert_close(summary.pooled_gram, expected)
    for index, document_id in enumerate(summary.document_ids):
        mask = torch.tensor([item == document_id for item in ids])
        explicit_sum = torch.einsum("npo,nqo->pq", outputs[mask], outputs[mask])
        torch.testing.assert_close(summary.gram_sums[index], explicit_sum)


def _explicit_noncentral_gaussian_fourth(mean: torch.Tensor, covariance: torch.Tensor):
    dimension = mean.numel()
    fourth = torch.empty((dimension,) * 4, dtype=torch.float64)
    for i in range(dimension):
        for j in range(dimension):
            for k in range(dimension):
                for l in range(dimension):
                    fourth[i, j, k, l] = (
                        mean[i] * mean[j] * mean[k] * mean[l]
                        + covariance[i, j] * mean[k] * mean[l]
                        + covariance[i, k] * mean[j] * mean[l]
                        + covariance[i, l] * mean[j] * mean[k]
                        + covariance[j, k] * mean[i] * mean[l]
                        + covariance[j, l] * mean[i] * mean[k]
                        + covariance[k, l] * mean[i] * mean[j]
                        + covariance[i, j] * covariance[k, l]
                        + covariance[i, k] * covariance[j, l]
                        + covariance[i, l] * covariance[j, k]
                    )
    return fourth


def _dense_quadratic_tensor(factors: BilinearFactors) -> torch.Tensor:
    return torch.einsum("og,gi,gj->oij", factors.down, factors.left, factors.right)


def test_noncentral_gaussian_wick_matches_explicit_fourth_tensor_without_large_allocation():
    first = _factors(21, output=2, gates=5, dimension=3)
    second = _factors(22, output=2, gates=4, dimension=3)
    mean = torch.tensor([0.7, -1.1, 0.2], dtype=torch.float64)
    covariance = torch.tensor(
        [[1.2, 0.1, 0.0], [0.1, 0.8, 0.2], [0.0, 0.2, 0.4]], dtype=torch.float64,
    )
    moments = PopulationMoments(
        count=7, mean=mean, centered_outer_sum=7.0 * covariance,
    )
    fourth = _explicit_noncentral_gaussian_fourth(mean, covariance)
    expected = torch.einsum(
        "oij,okl,ijkl->", _dense_quadratic_tensor(first),
        _dense_quadratic_tensor(second), fourth,
    )
    # block_size=1 exercises the gate-blocked path and cannot materialize q x q'.
    actual = noncentral_gaussian_cross_inner_product(first, second, moments, block_size=1)
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
    reverse = noncentral_gaussian_cross_inner_product(second, first, moments, block_size=2)
    torch.testing.assert_close(reverse, expected, rtol=2e-13, atol=2e-13)
    gram = noncentral_gaussian_probe_gram((first, first), moments, block_size=2)
    assert torch.equal(gram, gram.T)
    torch.testing.assert_close(gram[0, 0], gram[0, 1], rtol=0, atol=0)


def test_average_ranks_and_spearman_use_exact_average_ties():
    values = torch.tensor([30.0, 10.0, 20.0, 10.0, 30.0], dtype=torch.float64)
    torch.testing.assert_close(
        average_ranks(values), torch.tensor([4.5, 1.5, 3.0, 1.5, 4.5], dtype=torch.float64),
    )
    reversed_values = torch.tensor([0.0, 3.0, 2.0, 3.0, 0.0], dtype=torch.float64)
    assert spearman_average_rank(values, values) == pytest.approx(1.0)
    assert spearman_average_rank(values, reversed_values) == pytest.approx(-1.0)
    assert spearman_average_rank(torch.ones(4), torch.arange(4, dtype=torch.float64)) is None


def test_simultaneous_document_bootstrap_is_shared_row_weighted_nonlinear_and_exact_rank():
    sums = torch.tensor(
        [[4.0, 2.0], [3.0, 1.0], [2.0, 5.0], [8.0, 4.0]], dtype=torch.float64,
    )
    counts = torch.tensor([2, 1, 3, 2], dtype=torch.int64)

    def statistic(pooled: torch.Tensor) -> torch.Tensor:
        ratio = pooled[..., 0] / pooled[..., 1]
        difference = pooled[..., 0] - pooled[..., 1]
        return torch.stack((ratio, difference, ratio.square()), dim=-1)

    result = simultaneous_document_bootstrap(
        sums, counts, statistic, repetitions=23, seed=1234,
        confidence=0.8, draw_chunk_size=7,
    )
    point = statistic(sums.sum(0) / counts.sum())
    torch.testing.assert_close(result.point, point)
    assert result.critical_order_statistic_one_indexed == math.ceil(0.8 * 23)

    generator = torch.Generator().manual_seed(1234)
    manual_draws = []
    for start in range(0, 23, 7):
        chunk = min(7, 23 - start)
        indices = torch.randint(4, (chunk, 4), generator=generator)
        pooled = sums[indices].sum(1) / counts[indices].sum(1).to(torch.float64)[:, None]
        manual_draws.append(statistic(pooled))
    manual_draws = torch.cat(manual_draws)
    torch.testing.assert_close(result.draws, manual_draws, rtol=0, atol=0)
    errors = torch.abs(manual_draws - point).amax(1)
    expected_critical = torch.kthvalue(errors, math.ceil(0.8 * 23)).values
    assert result.critical_value == float(expected_critical)
    torch.testing.assert_close(result.lower, point - expected_critical)
    torch.testing.assert_close(result.upper, point + expected_critical)

    repeat = simultaneous_document_bootstrap(
        sums, counts, statistic, repetitions=23, seed=1234,
        confidence=0.8, draw_chunk_size=7,
    )
    torch.testing.assert_close(repeat.draws, result.draws, rtol=0, atol=0)
    with pytest.raises(ValueError, match="malformed"):
        simultaneous_document_bootstrap(
            sums[:1], counts[:1], statistic, repetitions=3, seed=1,
        )


def test_public_functions_reject_graphs_nonfinite_and_wrong_bootstrap_statistics():
    accumulator = StreamingPopulationMoments(2)
    with pytest.raises(ValueError, match="graph-free"):
        accumulator.update(torch.ones(2, 2, requires_grad=True))
    with pytest.raises(ValueError, match="finite"):
        accumulator.update(torch.tensor([[float("nan"), 0.0]]))

    sums = torch.ones(3, 2, dtype=torch.float64)
    counts = torch.ones(3, dtype=torch.int64)
    with pytest.raises(ValueError, match="statistic"):
        simultaneous_document_bootstrap(
            sums, counts, lambda pooled: pooled.float(), repetitions=2, seed=0,
        )
