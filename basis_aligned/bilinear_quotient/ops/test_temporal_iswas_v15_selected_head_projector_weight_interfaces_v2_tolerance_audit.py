import math

import run_temporal_iswas_v15_selected_head_projector_weight_interfaces_v2_tolerance_audit as audit


def test_finite_scores_rejects_nested_nonfinite_values():
    assert audit.finite_scores({"fold": [{"score": 0.25}, 1]})
    assert not audit.finite_scores({"fold": [{"score": float("nan")}]})
    assert not audit.finite_scores([float("inf")])


def test_registered_relative_bound_is_scale_aware_fp32_tolerance():
    closure = 2.384185791015625e-7
    minimum_write_norm = 9.2426
    relative = closure / minimum_write_norm
    assert math.isfinite(relative)
    assert relative <= 8 * 2.0 ** -23


def test_audit_prediction_key_differs_from_original_v1_key():
    assert audit.PREDICTION_KEYS[0] == "pred_a_authority_original_disposition_and_price"
    assert audit.ORIGINAL_FAILURE_KEY not in audit.PREDICTION_KEYS
    assert set(audit.ORIGINAL_REQUIRED_KEYS).isdisjoint(audit.PREDICTION_KEYS)
