import torch

import native_gate_subset as subset


def _case(seed: int = 0, *, n: int = 80, d: int = 6, m: int = 12):
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(m, d, generator=generator, dtype=torch.float64)
    right = torch.randn(m, d, generator=generator, dtype=torch.float64)
    down = torch.randn(d, m, generator=generator, dtype=torch.float64)
    bias = torch.randn(d, generator=generator, dtype=torch.float64)
    u = torch.randn(n, d, generator=generator, dtype=torch.float64)
    v = torch.randn(n, d, generator=generator, dtype=torch.float64)
    return left, right, down, bias, u, v


def test_full_gate_program_exactly_replays_all_four_terms():
    left, right, down, bias, u, v = _case()
    features = subset.typed_gate_features(left, right, u, v)
    matrix, writes = subset.stack_features_and_writes(features, down)
    gram, cross = subset.sufficient_statistics(matrix, writes)
    decoder = subset.fit_joint_decoder(gram, cross, relative_ridge=0.0)
    indices = torch.arange(left.shape[0])
    program = subset.build_program(left, right, bias, indices, decoder)
    observed = program.terms(u, v)
    for name in subset.TERM_NAMES:
        expected = torch.nn.functional.linear(features[name], down)
        assert torch.allclose(observed[name], expected, atol=1e-8)


def test_program_price_counts_removed_products_and_all_three_factor_banks():
    left, right, down, bias, _, _ = _case()
    indices = torch.tensor([1, 3, 8])
    program = subset.build_program(left, right, bias, indices, down[:, indices])
    assert program.product_count_per_token == 3
    assert program.float_parameter_count == 3 * 6 * 3 + 6
    assert program.float_parameter_count < 3 * 6 * 12 + 6


def test_batched_omp_finds_planted_predictive_feature_group():
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(300, 20, generator=generator, dtype=torch.float64)
    planted = torch.tensor([2, 7, 13, 18])
    coefficient = torch.randn(4, 5, generator=generator, dtype=torch.float64)
    y = x[:, planted] @ coefficient
    gram, cross = subset.sufficient_statistics(x, y)
    energy = cross.square().sum(dim=1)
    chosen = subset.batch_simultaneous_omp(
        gram, cross, energy, budget=4, prefilter=20, batch_size=1,
        relative_ridge=1e-10,
    )
    assert set(chosen.tolist()) == set(planted.tolist())


def test_scale_gauge_does_not_change_subset_term_outputs():
    left, right, down, bias, u, v = _case(seed=2)
    indices = torch.tensor([0, 4, 6, 9])
    first = subset.build_program(left, right, bias, indices, down[:, indices])
    scale = torch.exp(torch.randn(
        left.shape[0], generator=torch.Generator().manual_seed(3), dtype=left.dtype,
    ))
    second = subset.build_program(
        scale[:, None] * left, right / scale[:, None], bias, indices, down[:, indices],
    )
    for name in subset.TERM_NAMES:
        assert torch.allclose(first.terms(u, v)[name], second.terms(u, v)[name], atol=1e-10)


def test_masked_write_uses_exact_frozen_term_sets(monkeypatch):
    left, right, down, bias, u, v = _case(seed=4)
    indices = torch.arange(left.shape[0])
    program = subset.build_program(left, right, bias, indices, down)
    terms = program.terms(u, v)
    observed = program.masked_write(u, v, "no_cross")
    assert torch.allclose(observed, terms["uu"] + terms["vv"] + bias, atol=1e-10)

    expected_all = sum(terms.values()) + bias

    def forbidden_four_bank_expansion(*args, **kwargs):
        raise AssertionError("deployable all-term write expanded into four product banks")

    monkeypatch.setattr(subset, "typed_gate_features", forbidden_four_bank_expansion)
    deployed = program.masked_write(u, v, "all")
    assert torch.allclose(deployed, expected_all, atol=1e-10)
