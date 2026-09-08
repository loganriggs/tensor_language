import run_temporal_iswas_v15_residual_state_boundary_mediation_atlas_v1 as runner


def fake_reports(passing):
    reports = {}
    for expert, panel in runner.EXPERTS.items():
        reports[expert] = {}
        for parity in (0, 1):
            reports[expert][str(parity)] = {}
            for boundary in runner.executor.BOUNDARIES:
                value = 1.0 if boundary in passing else 0.5
                metrics = {name: {"signed_projection": value, "direction_fraction": 1.0}
                           for name in ("head_reset_loss", "head_rescue")}
                reports[expert][str(parity)][boundary] = {panel: {"metrics": metrics}}
    return reports


def test_frozen_boundary_scope_and_price_are_exact():
    assert len(runner.executor.BOUNDARIES) == 13
    assert runner.executor.BOUNDARIES[0] == "entry12"
    assert runner.executor.BOUNDARIES[-1] == "post_mlp17"
    assert runner.PRICE_MAX["differentiable_transformer_forwards"] == 120
    assert runner.PRICE_MAX["transformer_backward_forwards"] == 0


def test_lockin_requires_every_later_boundary_and_is_preterminal():
    boundaries = runner.executor.BOUNDARIES
    assert runner.lockin_boundary(fake_reports(set(boundaries[5:]))) == boundaries[5]
    assert runner.lockin_boundary(fake_reports({"post_mlp17"})) is None
    assert runner.lockin_boundary(fake_reports(set(boundaries) - {boundaries[9]})) == boundaries[10]


def test_all_authorities_are_hash_bound_and_finite_is_recursive():
    assert set(runner.FILES) == set(runner.EXPECTED)
    assert all(len(value) == 64 for value in runner.EXPECTED.values())
    assert runner.finite({"x": [1.0]})
    assert not runner.finite({"x": [float("inf")]})
