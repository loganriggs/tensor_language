import run_task14_prospective_mlp6_7_single_reader_lattice_causal_validation as run


def test_price_is_complete_lattice_in_four_chunks():
    assert run.derive_price()=={"physical_model_forwards":5,"example_evaluations":1120,
        "causal_interventions":1024,"backwards":0,"parameter_updates":0,
        "maximum_patch_chunk_rows":256,"patch_chunks":4}


def test_plan_binds_prediction_and_literal_scorer():
    plan=run.compile_plan(); assert plan["sealed_prediction_sha256"]==run.PREDICTION_SHA256
    assert "no fitted scale" in plan["literal_scorer"] and len(plan["background_subsets"])==16


def test_score_passes_when_sealed_central_values_are_exact():
    sealed=run._load_prediction()["evidence"]
    causal=[{"row_id":x["row_id"],"direction":x["direction"],"template":x["template"],
        "background":x["background"],"cardinality":x["cardinality"],"actual_q":x["central_reader_q"]}
        for x in sealed]
    scored=run.score(causal,{"closure":0.0})
    assert all(scored["predictions"].values())
    assert scored["intermediate_only"]["count"]==448
