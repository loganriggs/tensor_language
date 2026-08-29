import torch

import rms_bilinear_polarization as polarization


def _fixture(seed=3):
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    residual = torch.randn(4, 7, 9, generator=generator, dtype=dtype)
    attention = torch.randn(4, 7, 9, generator=generator, dtype=dtype)
    left = torch.randn(13, 9, generator=generator, dtype=dtype)
    right = torch.randn(13, 9, generator=generator, dtype=dtype)
    down = torch.randn(9, 13, generator=generator, dtype=dtype)
    bias = torch.randn(9, generator=generator, dtype=dtype)
    return left, right, down, bias, residual, attention


def test_exact_polarization_replays_native_post_attention_mlp():
    values = _fixture()
    assert polarization.exact_replay_error(*values, eps=1e-6) < 2e-15


def test_zero_attention_reduces_to_base_quadratic_plus_bias():
    left, right, down, bias, residual, attention = _fixture()
    terms = polarization.polarized_terms(
        left, right, down, bias, residual, torch.zeros_like(attention), eps=1e-6,
    )
    normalized = polarization.rms_scale(residual, eps=1e-6) * residual
    expected = polarization.native_mlp(left, right, down, bias, normalized)
    assert torch.allclose(terms.output, expected, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(terms.left_residual_right_attention) == 0
    assert torch.count_nonzero(terms.left_attention_right_residual) == 0
    assert torch.count_nonzero(terms.attention_quadratic) == 0


def test_gate_scale_gauge_preserves_every_typed_term():
    left, right, down, bias, residual, attention = _fixture()
    scale = torch.logspace(-3, 3, left.shape[0], dtype=left.dtype)
    original = polarization.polarized_terms(
        left, right, down, bias, residual, attention, eps=1e-6,
    )
    gauged = polarization.polarized_terms(
        left * scale[:, None], right / scale[:, None], down,
        bias, residual, attention, eps=1e-6,
    )
    for name in (
        "scaled_base_quadratic", "left_residual_right_attention",
        "left_attention_right_residual", "attention_quadratic", "bias",
    ):
        assert torch.allclose(
            getattr(original, name), getattr(gauged, name), atol=2e-12, rtol=2e-12,
        )


def test_both_asymmetric_cross_terms_are_required():
    values = _fixture(7)
    terms = polarization.polarized_terms(*values, eps=1e-6)
    assert torch.linalg.vector_norm(terms.left_residual_right_attention) > 0
    assert torch.linalg.vector_norm(terms.left_attention_right_residual) > 0
    assert not torch.allclose(
        terms.left_residual_right_attention, terms.left_attention_right_residual,
    )


def test_nonzero_eps_and_batch_axes_are_part_of_exact_identity():
    values = _fixture(11)
    for eps in (1e-12, 1e-6, 1e-2):
        assert polarization.exact_replay_error(*values, eps=eps) < 2e-15
