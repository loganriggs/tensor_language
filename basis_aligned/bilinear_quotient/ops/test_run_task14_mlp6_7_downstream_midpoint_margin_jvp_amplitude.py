import run_task14_mlp6_7_downstream_midpoint_margin_jvp_amplitude as run


def test_price_is_derived_from_role_and_amplitude_counts():
    assert run.derive_price() == {"physical_model_forwards": 3,
        "example_evaluations": 280, "backwards": 2, "causal_interventions": 0,
        "parameter_updates": 0, "predicted_amplitudes_per_linearization": 80}


def test_frozen_targets_have_exactly_two_endpoints_per_row():
    targets = run._frozen_targets()
    assert len(targets) == 80
    assert {background for _, background in targets} == set(run.BACKGROUNDS)


def test_vector_stats_reports_exact_prediction():
    stats = run._vector_stats([-2., 1., 3.], [-2., 1., 3.])
    assert abs(stats["cosine"] - 1) < 1e-12
    assert stats["relative_l2_error"] == 0
    assert stats["sign_agreement"] == 1


def test_score_gates_midpoint_and_controls_separately():
    evidence = []
    for phase, count in (("FIT", 64), ("HOLDOUT", 16)):
        for i in range(count):
            actual = 1.0 if i % 2 else -1.0
            evidence.append({"row_id": f"{phase}-{i}", "phase": phase,
                "direction": "plural_to_singular" if actual < 0 else "singular_to_plural",
                "template": "t", "background": run.BACKGROUNDS[i % 2],
                "actual_q": actual, "base_jvp_q": .5 * actual,
                "midpoint_jvp_q": actual})
    exactness = {"x": 0.0}
    gradients = {point: {"finite": True, "l2_norm": 1., "nonzero_row_count": 80}
                 for point in run.POINTS}
    scored = run.score(evidence, exactness, gradients)
    assert scored["predictions"]["pred_a_gradient_instrument"]
    assert scored["predictions"]["pred_b_midpoint_generates_amplitude"]
    assert scored["predictions"]["pred_c_midpoint_generalizes_by_phase"]
    assert scored["predictions"]["pred_d_both_backgrounds_are_predictable"]
    assert scored["predictions"]["pred_e_midpoint_improves_endpoint"]
