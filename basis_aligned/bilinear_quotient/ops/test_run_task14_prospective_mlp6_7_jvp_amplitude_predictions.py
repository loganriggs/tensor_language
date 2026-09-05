import run_task14_prospective_mlp6_7_jvp_amplitude_predictions as run


def test_prediction_price_is_exact_and_has_no_causal_intervention():
    assert run.derive_price() == {"physical_model_forwards": 3,
        "example_evaluations": 224, "backwards": 2, "causal_interventions": 0,
        "parameter_updates": 0, "sealed_predictions": 64}


def test_plan_binds_license_and_keeps_causal_outcomes_closed():
    plan = run.compile_plan()
    assert plan["causal_outcomes_opened"] is False
    assert plan["capability_license_sha256"] == run.CAPABILITY_LICENSE_SHA256
    assert plan["price"]["sealed_predictions"] == 64
