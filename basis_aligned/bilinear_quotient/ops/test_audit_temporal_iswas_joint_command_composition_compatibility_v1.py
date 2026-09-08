import audit_temporal_iswas_joint_command_composition_compatibility_v1 as audit


def test_authorities_are_hash_bound():
    assert set(audit.FILES) == set(audit.EXPECTED)
    assert all(len(value) == 64 for value in audit.EXPECTED.values())


def test_price_is_literal_zero_forward():
    assert audit.PRICE == {"model_forwards": 0, "transformer_backwards": 0, "model_updates": 0,
                           "checkpoint_loads": 0, "example_evaluations": 0, "fit_parameters": 0}


def test_finite_rejects_nested_nan():
    assert audit.finite({"x": [1.0]})
    assert not audit.finite({"x": [float("nan")]})
