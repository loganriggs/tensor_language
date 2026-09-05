import run_task14_prospective_mlp6_7_jvp_amplitude_causal_validation as run


def test_causal_price_counts_only_role_generation_and_one_patch_batch():
    assert run.derive_price()=={"physical_model_forwards":2,"example_evaluations":224,
        "causal_interventions":128,"backwards":0,"parameter_updates":0}


def test_plan_binds_immutable_prediction_and_price_amendment():
    plan=run.compile_plan()
    assert plan["sealed_prediction_sha256"]==run.PREDICTION_SHA256
    assert plan["price_amendment_sha256"]==run.AMENDMENT_SHA256
    assert "no fitted scale" in plan["literal_scorer"]


def test_literal_score_passes_exact_sealed_prediction():
    sealed=run._load_prediction()["evidence"]
    causal=[{"row_id":x["row_id"],"direction":x["direction"],"template":x["template"],
        "background":x["background"],"actual_q":x["midpoint_jvp_q"]} for x in sealed]
    score=run.score(causal,{"closure":0.0})
    assert all(score["predictions"].values())
    assert score["overall"]["midpoint"]["relative_l2_error"]==0
    assert score["descriptive_post_gate_affine_repair"]["gate_effect"]=="DESCRIPTIVE_ONLY_NONE"
