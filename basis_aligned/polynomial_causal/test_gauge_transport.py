import pytest
import torch

from gauge_transport import (
    physical_transport,
    powered_sign_agreement,
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
