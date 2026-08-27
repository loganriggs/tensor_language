import torch

from mlp0_native_down_program import (
    canonical_balanced_factors,
    deterministic_centroid_derangement,
    fit_reduced_rank_from_statistics,
    load_program,
    matched_hierarchy_rank,
    program_price_bytes,
    serialize_program,
)


def test_matched_rank_is_largest_rank_at_or_below_continuous_price():
    rank = matched_hierarchy_rank(32, occupied=5, vocab=100, d_model=8, hidden=12)
    ceiling = program_price_bytes(32, vocab=100, d_model=8, hidden=12)["total"]
    assert program_price_bytes(rank, occupied=5, vocab=100, d_model=8, hidden=12)["total"] <= ceiling
    assert program_price_bytes(rank + 1, occupied=5, vocab=100, d_model=8, hidden=12)["total"] > ceiling


def test_balanced_factors_preserve_product_and_fix_loading_sign():
    generator = torch.Generator().manual_seed(4)
    basis, _ = torch.linalg.qr(torch.randn(7, 3, generator=generator))
    coefficient = torch.randn(11, 7, generator=generator)
    left, right = canonical_balanced_factors(basis, coefficient)
    expected = basis @ basis.T @ coefficient.T
    assert torch.allclose(left @ right, expected, atol=2e-5, rtol=2e-5)
    for column in range(left.shape[1]):
        pivot = int(left[:, column].abs().argmax())
        assert left[pivot, column] >= 0


def test_reduced_rank_statistics_recover_exact_low_rank_map():
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(400, 6, generator=generator)
    true_left = torch.randn(4, 2, generator=generator)
    true_right = torch.randn(2, 6, generator=generator)
    y = x @ (true_left @ true_right).T
    covariance = x.T @ x / len(x)
    cross = x.T @ y / len(x)
    fit = fit_reduced_rank_from_statistics(covariance, cross, rank=2, ridge_fraction=1e-8)
    prediction = x @ (fit["left"] @ fit["right"]).T
    assert torch.allclose(prediction, y, atol=2e-4, rtol=2e-4)


def test_derangement_has_no_fixed_points_and_preserves_centroid_multiset():
    centroids = torch.tensor([[0., 0.], [1., 0.], [3., 0.], [7., 0.]])
    masses = torch.tensor([10., 11., 30., 31.])
    permuted, report = deterministic_centroid_derangement(centroids, masses)
    assert report["fixed_points"] == 0
    assert sorted(map(tuple, permuted.tolist())) == sorted(map(tuple, centroids.tolist()))
    assert all(index != target for index, target in enumerate(report["permutation"]))


def test_fixed_layout_serialization_roundtrips_bf16_and_matches_price(tmp_path):
    generator = torch.Generator().manual_seed(12)
    program = {
        "rank": 2,
        "intercept": torch.randn(3, generator=generator),
        "left": torch.randn(3, 2, generator=generator),
        "right": torch.randn(2, 5, generator=generator),
        "centroids": torch.randn(4, 3, generator=generator),
        "assignments": torch.tensor([0, 1, 4, 2, 0, 3, 1]),  # 4 is zero sentinel
    }
    path = tmp_path / "program.bin"
    receipt = serialize_program(path, program)
    loaded = load_program(path)
    assert receipt["bytes"] == program_price_bytes(
        2, occupied=4, vocab=7, d_model=3, hidden=5
    )["total"]
    for name in ("intercept", "left", "right", "centroids"):
        assert torch.equal(loaded[name], program[name].to(torch.bfloat16))
    assert torch.equal(loaded["assignments"], program["assignments"])
