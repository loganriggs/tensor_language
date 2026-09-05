import run_task14_prospective_mlp6_7_single_reader_lattice_predictions as run


def test_price_has_two_reader_gradients_and_no_causal_outcomes():
    assert run.derive_price()=={"physical_model_forwards":3,"example_evaluations":160,
        "backwards":2,"causal_interventions":0,"sealed_predictions":512,"parameter_updates":0}


def test_plan_binds_parent_and_keeps_intermediates_closed():
    plan=run.compile_plan()
    assert plan["parent_result_sha256"]==run.PARENT_RESULT_SHA256
    assert len(plan["background_subsets"])==16
    assert "448 INTERMEDIATE TARGETS CLOSED" in plan["causal_outcomes_opened"]
