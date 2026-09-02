from types import SimpleNamespace

import torch

import mlp1_finite_secant_factor_interchange_rung487 as subject


def _fake_mlp(seed=3):
    generator = torch.Generator().manual_seed(seed)
    return SimpleNamespace(
        Left=SimpleNamespace(weight=torch.randn(7, 5, generator=generator)),
        Right=SimpleNamespace(weight=torch.randn(7, 5, generator=generator)),
        Down=SimpleNamespace(weight=torch.randn(5, 7, generator=generator)),
        Down_bias=torch.randn(5, generator=generator),
    )


def test_polarization_reconstructs_exact_quadratic_difference():
    generator = torch.Generator().manual_seed(7)
    mlp = _fake_mlp()
    native = torch.randn(2, 4, 5, generator=generator)
    absent = torch.randn(2, 4, 5, generator=generator)
    delta = native - absent
    midpoint = (native + absent) / 2
    direct = subject._mlp_write(mlp, native) - subject._mlp_write(mlp, absent)
    secant = subject._secant(mlp, delta, midpoint)
    assert torch.allclose(secant, direct, atol=2e-5, rtol=2e-5)


def test_cross_factor_construction_preserves_own_and_both_identities():
    generator = torch.Generator().manual_seed(9)
    mlp = _fake_mlp()
    states = {name: torch.randn(2, 3, 5, generator=generator)
              for name in ("native", "T", "C")}
    secants, _ = subject._secants_for_pair(mlp, states, "T", "C")
    assert torch.allclose(
        secants["own"],
        subject._mlp_write(mlp, states["native"].float())
        - subject._mlp_write(mlp, states["T"].float()),
        atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        secants["both"],
        subject._mlp_write(mlp, states["native"].float())
        - subject._mlp_write(mlp, states["C"].float()),
        atol=2e-5, rtol=2e-5)


def test_analysis_finds_bidirectional_context_factor_edge():
    generator = torch.Generator().manual_seed(13)
    base = torch.randn(subject.SPLIT, subject.TOKENS, generator=generator).double()
    own_by_branch = {
        name: torch.cat([base * scale, base * scale], dim=0)
        for name, scale in zip(subject.BRANCHES, (1.0, 1.3, -.8))}
    benefits = torch.zeros(
        len(subject.ORDERED_PAIRS), 2 * subject.SPLIT,
        subject.TOKENS, len(subject.MODES), dtype=torch.float64)
    for index, (target, donor) in enumerate(subject.ORDERED_PAIRS):
        own = own_by_branch[target]
        benefits[index, ..., subject.MODES.index("own")] = own
        benefits[index, ..., subject.MODES.index("both")] = own_by_branch[donor]
        if {target, donor} == {"T", "C"}:
            benefits[index, ..., subject.MODES.index("context")] = own
            benefits[index, ..., subject.MODES.index("direction")] = \
                torch.roll(own, 17, dims=1)
        else:
            benefits[index, ..., subject.MODES.index("context")] = \
                torch.roll(own, 11, dims=1)
            benefits[index, ..., subject.MODES.index("direction")] = \
                torch.roll(own, 17, dims=1)
    write_cosines = torch.zeros(
        2, len(subject.ORDERED_PAIRS), 2, 1 + len(subject.POSITION_SHIFTS))
    for index, pair in enumerate(subject.ORDERED_PAIRS):
        if set(pair) == {"T", "C"}:
            write_cosines[:, index, 0, 0] = 1.0
    collected = {
        "arms": -benefits,
        "absent": torch.zeros(3, 2 * subject.SPLIT, subject.TOKENS),
        "write_cosines": write_cosines,
    }
    positive = torch.ones(2 * subject.SPLIT, subject.TOKENS, dtype=torch.bool)
    report = subject.analyze_phase(collected, positive)
    assert report["pred_b_own_responses_stable"] is True
    assert report["own_response_reports"]["T"]["half1_over_half0_rms"] == 1.0
    assert report["pred_c_at_least_one_factor_edge"] is True
    assert report["pred_d_factor_graph_stable"] is True
    assert report["descriptive_edges"] == [{"pair": ["T", "C"], "type": "context"}]


def test_scaled_error_accepts_a_nonunit_scale():
    predictor = torch.tensor([1.0, -2.0, 3.0])
    alpha, error = subject._scaled_error(predictor, 2.5 * predictor)
    assert abs(alpha - 2.5) < 1e-12
    assert error < 1e-12
