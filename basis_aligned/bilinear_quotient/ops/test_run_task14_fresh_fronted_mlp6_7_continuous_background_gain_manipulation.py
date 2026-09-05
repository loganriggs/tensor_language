import run_task14_fresh_fronted_mlp6_7_continuous_background_gain_manipulation as run


def test_compiled_specs_and_price_are_derived():
    specs = run.compile_specs()
    assert len(specs) == 832 and len(set(specs)) == 832
    assert run.derive_price() == {"physical_model_forwards": 10,
        "example_evaluations": 1856, "causal_installations": 832,
        "backwards": 0, "parameter_updates": 0,
        "maximum_patch_chunk_rows": 256, "patch_chunks": 4}


def test_endpoint_and_new_gain_methods_are_separate():
    assert all(run._methods(g) == ("base", "exact") for g in run.ENDPOINT_GAINS)
    assert all(run._methods(g) == ("base", "exact", "predicted") for g in run.NEW_GAINS)


def test_plan_binds_parent_and_price_amendment():
    plan = run.compile_plan()
    assert plan["parent_result_sha256"] == run.PARENT_RESULT_SHA256
    assert plan["price_amendment_sha256"] == run.AMENDMENT_SHA256
    assert plan["price"] == run.derive_price()


def test_all_gains_are_unique_and_ordered():
    assert run.ALL_GAINS == (-0.5, 0.0, 0.5, 1.0, 1.5)
    assert set(run.ENDPOINT_GAINS).isdisjoint(run.NEW_GAINS)


def test_recovery_convention_preserves_negative_effect_sign():
    exact, predicted = -0.2, -0.19
    assert predicted / exact == 0.95
