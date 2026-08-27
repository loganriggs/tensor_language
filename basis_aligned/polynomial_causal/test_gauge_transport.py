import pytest
import torch

from gauge_transport import (
    beats_all_nulls,
    centered_logit_response_sums,
    commuting_output_metrics,
    fit_delta_ridge,
    haar_basis_in_support,
    physical_transport,
    powered_sign_agreement,
    response_r2,
    response_metrics,
    rewrite_coordinate_gauge,
)


def _well_conditioned_gauge(dimension, seed):
    generator = torch.Generator().manual_seed(seed)
    left, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    )
    right, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    )
    scales = torch.logspace(0, 2, dimension, dtype=torch.float64)
    return left @ torch.diag(scales) @ right.T


def test_complete_coordinate_rewrite_preserves_physical_map_and_responses():
    generator = torch.Generator().manual_seed(11)
    encoder = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    transport = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    decoder = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    source_gauge = _well_conditioned_gauge(3, 12)
    destination_gauge = _well_conditioned_gauge(4, 13)
    rewritten = rewrite_coordinate_gauge(
        decoder, transport, encoder, source_gauge, destination_gauge
    )
    original_physical = physical_transport(decoder, transport, encoder)
    rewritten_physical = physical_transport(*rewritten)
    torch.testing.assert_close(original_physical, rewritten_physical, rtol=1e-11, atol=1e-11)
    interventions = torch.randn(19, 7, generator=generator, dtype=torch.float64)
    original_response = interventions @ original_physical.T
    rewritten_response = interventions @ rewritten_physical.T
    assert response_metrics(original_response, rewritten_response)["nre"] < 1e-11


def test_incomplete_rewrite_is_a_detectable_negative_control():
    generator = torch.Generator().manual_seed(21)
    encoder = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    transport = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    decoder = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    source_gauge = _well_conditioned_gauge(3, 22)
    interventions = torch.randn(40, 7, generator=generator, dtype=torch.float64)
    reference = interventions @ physical_transport(decoder, transport, encoder).T
    # Rewriting the encoder but not its adjacent transport changes the function.
    broken = interventions @ physical_transport(
        decoder, transport, source_gauge @ encoder
    ).T
    assert response_metrics(reference, broken)["nre"] > 0.5


def test_response_metrics_have_registered_scale_and_sign_behavior():
    reference = torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    exact = response_metrics(reference, reference)
    assert exact["nre"] == 0.0
    assert exact["per_example_relative_q90"] == 0.0
    half = response_metrics(reference, 0.5 * reference)
    assert half["nre"] == pytest.approx(0.5)
    assert half["response_cosine"] == pytest.approx(1.0)
    effects = torch.tensor([2.0, -3.0, 0.1, -0.2])
    predictions = torch.tensor([1.0, -1.0, -0.1, -0.3])
    powered = torch.tensor([True, True, True, False])
    assert powered_sign_agreement(effects, predictions, powered) == pytest.approx(2 / 3)


def test_invalid_or_singular_interfaces_fail_closed():
    decoder = torch.randn(5, 3, dtype=torch.float64)
    transport = torch.randn(3, 2, dtype=torch.float64)
    encoder = torch.randn(2, 7, dtype=torch.float64)
    with pytest.raises(ValueError, match="source_gauge"):
        rewrite_coordinate_gauge(
            decoder, transport, encoder, torch.eye(3), torch.eye(3)
        )
    singular = torch.zeros(2, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="nonsingular"):
        rewrite_coordinate_gauge(
            decoder, transport, encoder, singular, torch.eye(3, dtype=torch.float64)
        )
    with pytest.raises(ValueError, match="all-zero"):
        response_metrics(torch.zeros(3, 2), torch.zeros(3, 2))


def test_zero_origin_ridge_recovers_direct_and_composed_response_maps():
    generator = torch.Generator().manual_seed(31)
    source = torch.randn(400, 5, generator=generator, dtype=torch.float64)
    map_8_11 = torch.randn(5, 7, generator=generator, dtype=torch.float64)
    map_11_14 = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    delta_11 = source @ map_8_11
    delta_14 = delta_11 @ map_11_14
    fit_8_11 = fit_delta_ridge(source, delta_11, relative_ridge=1e-10)
    fit_11_14 = fit_delta_ridge(delta_11, delta_14, relative_ridge=1e-10)
    fit_8_14 = fit_delta_ridge(source, delta_14, relative_ridge=1e-10)
    direct = source @ fit_8_14
    chained = source @ fit_8_11 @ fit_11_14
    assert response_r2(delta_14, direct) > 1 - 1e-16
    assert response_r2(delta_14, chained) > 1 - 1e-16


def test_triangle_maps_transform_consistently_under_orthogonal_gauges():
    generator = torch.Generator().manual_seed(41)
    delta_8 = torch.randn(100, 3, generator=generator, dtype=torch.float64)
    t_8_11 = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    t_11_14 = torch.randn(4, 2, generator=generator, dtype=torch.float64)
    q8, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    q11, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator, dtype=torch.float64))
    q14, _ = torch.linalg.qr(torch.randn(2, 2, generator=generator, dtype=torch.float64))
    # Row-coordinate convention: c' = c Q, T_ij' = Q_i^T T_ij Q_j.
    delta_8_prime = delta_8 @ q8
    t_8_11_prime = q8.T @ t_8_11 @ q11
    t_11_14_prime = q11.T @ t_11_14 @ q14
    original = delta_8 @ t_8_11 @ t_11_14
    rewritten = delta_8_prime @ t_8_11_prime @ t_11_14_prime
    torch.testing.assert_close(original @ q14, rewritten, rtol=1e-12, atol=1e-12)


def test_commuting_output_error_has_exact_and_null_controls():
    baseline = torch.tensor(
        [[2.0, 0.0, -1.0], [0.0, 1.0, -2.0]], dtype=torch.float64
    )
    early = baseline + torch.tensor(
        [[0.4, -0.2, 0.1], [-0.3, 0.2, 0.5]], dtype=torch.float64
    )
    exact = commuting_output_metrics(baseline, early, early + 17.0)
    assert exact["e_out"] == pytest.approx(0.0, abs=1e-12)
    assert exact["centered_logit_relative_rmse"] == pytest.approx(0.0, abs=1e-12)
    null = commuting_output_metrics(baseline, early, baseline)
    assert null["e_out"] == pytest.approx(1.0)
    assert null["centered_logit_relative_rmse"] == pytest.approx(1.0)
    raw = centered_logit_response_sums(baseline, early, early + 17.0)
    assert raw["centered_logit_error_sum_squares"] == pytest.approx(0.0, abs=1e-22)
    assert raw["centered_logit_target_sum_squares"] > 0


def test_matched_haar_basis_stays_inside_support_and_null_gate_is_exact():
    generator = torch.Generator().manual_seed(51)
    support, _ = torch.linalg.qr(
        torch.randn(9, 6, generator=generator, dtype=torch.float64), mode="reduced"
    )
    basis = haar_basis_in_support(support, 4, generator=generator)
    torch.testing.assert_close(basis.T @ basis, torch.eye(4, dtype=torch.float64))
    residual = basis - support @ (support.T @ basis)
    assert float(residual.norm()) < 1e-12
    gate = beats_all_nulls(0.1, [0.2 + 0.01 * index for index in range(20)],
                           lower_is_better=True)
    assert gate == {
        "passed": True,
        "null_at_least_as_good": 0,
        "finite_null_p": pytest.approx(1 / 21),
        "n_nulls": 20,
    }
    assert not beats_all_nulls(0.3, [0.2, 0.4], lower_is_better=True)["passed"]
