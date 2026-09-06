import run_temporal_auxiliary_will_had_local_l9_value_reuse_v1 as runner


def test_authority_and_prospective_boundary():
    rows = runner.validate_static()
    assert len(rows) == 64
    assert {row["transform_id"] for row in rows} == {"A1", "A2"}
    assert {row["direction_id"] for row in rows} == {
        "future_to_anterior",
        "anterior_to_future",
    }


def test_inherited_factorial_is_exactly_frozen():
    assert runner.FACTORS == (
        "routing_on_base_value",
        "local_v9_content_change",
        "routing_local_interaction",
    )
    assert len(runner.inherited.subsets()) == 8


def test_shapley_efficiency_contract():
    weights = dict(zip(runner.FACTORS, (0.2, 0.8, -0.1)))
    values = {
        subset: sum(weights[name] for name in subset)
        for subset in runner.inherited.subsets()
    }
    shapley = runner.inherited.factorial_shapley(values)
    assert all(abs(shapley[name] - weights[name]) < 1e-12 for name in runner.FACTORS)
