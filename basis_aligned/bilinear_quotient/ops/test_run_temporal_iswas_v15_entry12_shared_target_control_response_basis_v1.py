import run_temporal_iswas_v15_entry12_shared_target_control_response_basis_v1 as runner


def test_frozen_bases_and_price_are_exact():
    assert runner.BASES == ("shared_dim_rank1", "construction_union_rank_le_2",
                            "p_complement_union_rank_le_2")
    assert runner.PRICE_MAX["differentiable_transformer_forwards"] == 32
    assert runner.PRICE_MAX["transformer_backward_forwards"] == 0
    assert runner.PRICE_MAX["fit_parameters"] == 0


def test_all_authorities_are_hash_bound():
    assert set(runner.FILES) == set(runner.EXPECTED)
    assert all(len(value) == 64 for value in runner.EXPECTED.values())


def test_finite_rejects_nested_nonfinite_values():
    assert runner.finite({"x": [1.0]})
    assert not runner.finite({"x": [float("nan")]})
