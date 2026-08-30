import toy_projected_causal_abstraction as toy


def test_projected_causal_abstraction_positive_null_gauge_and_pca_control():
    result = toy.run()
    assert all(result["checks"].values())
    assert result["hidden_fiber_separation"] > 1.0
    assert result["matched_rank_pca_mse"] > 0.1
    assert result["causal_code_mse"] < 1e-20
