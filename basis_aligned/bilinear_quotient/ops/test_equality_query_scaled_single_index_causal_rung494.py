import numpy as np
import pytest
import torch

import equality_query_scaled_single_index_causal_rung494 as rung


def test_literal_price_and_frozen_grid():
    assert rung.SCALES == (0.5, 1.5)
    assert rung.EXPECTED_BATCHES == 72
    assert rung.FORWARDS_PER_BATCH == 59
    assert rung.EXPECTED_FORWARDS == 4266
    assert rung.EXPECTED_PATCH_CALLS == 5634
    assert len(rung.POSITION_SHIFTS) == 16


def test_scaled_patch_rejects_invalid_scale_before_model_access():
    with pytest.raises(ValueError, match="delta_scale"):
        rung.run_scaled_patch(
            None, None, arm="N", scale=None, deltas={}, sites=(),
            position_mask=None, delta_scale=-0.5)


def test_isotonic_curves_are_monotone_and_clip():
    # Seven subset effects in rung.SUBSETS order; singleton effects are
    # deliberately accompanied by non-additive pair/triple effects.
    effects = torch.tensor([
        [1.0, 1.2], [2.0, 2.1], [3.4, 3.0], [4.0, 4.2],
        [4.7, 4.5], [5.1, 5.0], [5.6, 5.4],
    ], dtype=torch.float64)
    mains, curves = rung._fit_curves(effects)
    assert mains.shape == (3, 2)
    for x, y in curves:
        assert np.all(np.diff(x) >= 0)
        assert np.all(np.diff(y) >= -1e-12)
    values = np.array([-1e6, 1e6])
    predicted = rung._predict(curves, values)
    assert predicted[0] == curves[0][1][0]
    assert predicted[1] == curves[1][1][-1]


def test_registered_error_report_can_pass_and_fail():
    actual = np.linspace(-1.0, 1.0, 12)
    prediction = actual + 0.01
    additive = actual + 0.20
    permuted = [actual + 0.30 + 0.01 * index for index in range(16)]
    half_masks = (
        np.array([True, True, False, False]),
        np.array([False, False, True, True]),
    )
    report = rung._report(
        actual, prediction, additive, permuted, half_masks,
        np.ones_like(actual, dtype=bool))
    assert report["primary_holds"] is True
    assert all(row["holds"] for row in report["halves"])

    failed = rung._report(
        actual, additive, prediction, permuted, half_masks,
        np.ones_like(actual, dtype=bool))
    assert failed["primary_holds"] is False


def test_source_contains_frozen_prediction_keys():
    source = rung.__file__
    text = open(source, encoding="utf-8").read()
    for key in (
        "pred_a_exact_live_scaled_intervention",
        "pred_b_half_strength_causal_interpolation",
        "pred_c_one_and_half_strength_causal_transfer",
        "pred_d_document_half_stability",
        "pred_e_local_causal_single_index_interpretation",
    ):
        assert f"'{key}'" in text
