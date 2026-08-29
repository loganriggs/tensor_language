import torch

import typed_block_rrr as rrr


def _factors(seed: int = 0, *, width: int = 7, gates: int = 11):
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(gates, width, generator=generator, dtype=torch.float64)
    right = torch.randn(gates, width, generator=generator, dtype=torch.float64)
    down = torch.randn(width, gates, generator=generator, dtype=torch.float64)
    return left, right, down


def test_full_rank_rrr_exactly_recovers_factor_map_on_full_support():
    left, right, down = _factors()
    generator = torch.Generator().manual_seed(1)
    values = torch.randn(100, 7, generator=generator, dtype=torch.float64)
    covariance = rrr.empirical_second_moment(values)
    program = rrr.fit_reduced_rank_factor_map(
        left, right, down, covariance, rank=7,
    )
    expected = rrr.native_factors(left, right, values)
    observed = program.factors(values)
    assert torch.allclose(observed[0], expected[0], atol=1e-9)
    assert torch.allclose(observed[1], expected[1], atol=1e-9)


def test_objective_energy_is_monotone_and_low_rank_map_has_real_cost():
    left, right, down = _factors()
    values = torch.randn(
        80, 7, generator=torch.Generator().manual_seed(2), dtype=torch.float64,
    )
    covariance = rrr.empirical_second_moment(values)
    programs = [
        rrr.fit_reduced_rank_factor_map(left, right, down, covariance, rank=rank)
        for rank in (1, 2, 4)
    ]
    energies = [program.objective_energy_fraction for program in programs]
    assert energies[0] <= energies[1] <= energies[2]
    assert programs[1].factor_parameter_count == 2 * (7 + 2 * 11)
    assert programs[1].factor_parameter_count < 2 * 11 * 7


def test_fit_and_typed_outputs_are_invariant_to_native_positive_scale_gauge():
    left, right, down = _factors()
    generator = torch.Generator().manual_seed(3)
    u = torch.randn(60, 7, generator=generator, dtype=torch.float64)
    v = torch.randn(60, 7, generator=generator, dtype=torch.float64)
    covariance = rrr.empirical_second_moment(u, v)
    scale = torch.exp(torch.randn(11, generator=generator, dtype=torch.float64))
    first = rrr.fit_reduced_rank_factor_map(
        left, right, down, covariance, rank=4,
    )
    second = rrr.fit_reduced_rank_factor_map(
        scale[:, None] * left, right / scale[:, None], down, covariance, rank=4,
    )
    terms1 = rrr.typed_terms(first, down, u, v)
    terms2 = rrr.typed_terms(second, down, u, v)
    for name in rrr.TERM_NAMES:
        assert torch.allclose(terms1[name], terms2[name], atol=1e-9)


def test_full_rank_typed_sum_equals_native_bilinear_write():
    left, right, down = _factors()
    generator = torch.Generator().manual_seed(4)
    u = torch.randn(40, 7, generator=generator, dtype=torch.float64)
    v = torch.randn(40, 7, generator=generator, dtype=torch.float64)
    covariance = rrr.empirical_second_moment(u, v)
    program = rrr.fit_reduced_rank_factor_map(
        left, right, down, covariance, rank=7,
    )
    bias = torch.randn(7, generator=generator, dtype=torch.float64)
    terms = rrr.typed_terms(program, down, u, v)
    observed = rrr.masked_write(terms, bias, "all")
    balanced = rrr.native_factors(left, right, u + v)
    expected = torch.nn.functional.linear(balanced[0] * balanced[1], down) + bias
    assert torch.allclose(observed, expected, atol=1e-8)


def test_mask_registry_has_the_frozen_five_distinct_edits():
    assert tuple(rrr.MASKS) == ("all", "no_vv", "no_cross", "cross_only", "uu_only")
    assert len({tuple(value) for value in rrr.MASKS.values()}) == 5

