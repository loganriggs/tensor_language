import pytest
import torch

import quadratic_tensor_rank_certificate as cert


def factors():
    left = torch.tensor([[1., 0., 1.], [0., 1., 1.]], dtype=torch.float64)
    right = torch.tensor([[1., 1., 0.], [1., -1., 1.]], dtype=torch.float64)
    decoder = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    return left, right, decoder


def test_full_unfolding_certifies_exact_quadratic_product_rank():
    result = cert.certify_quadratic_product_rank(*factors())
    assert result.exact_quadratic_product_rank_certified
    assert result.unfolding_rank_lower_bound == result.explicit_product_upper_bound == 2
    assert result.decoder.certified_lambda_min_lower_bound > 0
    assert result.symmetrized_products.certified_lambda_min_lower_bound > 0


def test_duplicate_symmetrized_product_fails_closed():
    left, right, decoder = factors()
    left[1], right[1] = left[0].clone(), right[0].clone()
    result = cert.certify_quadratic_product_rank(left, right, decoder)
    assert not result.exact_quadratic_product_rank_certified
    assert result.unfolding_rank_lower_bound == 0


def test_dependent_decoder_fails_closed():
    left, right, decoder = factors()
    decoder[:, 1] = decoder[:, 0]
    result = cert.certify_quadratic_product_rank(left, right, decoder)
    assert not result.exact_quadratic_product_rank_certified


def test_gate_reciprocal_gauge_preserves_certificate_spectrum():
    left, right, decoder = factors()
    scales = torch.tensor([7.0, 0.2], dtype=torch.float64)
    first = cert.certify_quadratic_product_rank(left, right, decoder)
    second = cert.certify_quadratic_product_rank(
        left * scales[:, None], right / scales[:, None], decoder,
    )
    assert second.exact_quadratic_product_rank_certified
    assert second.symmetrized_products.lambda_min_computed == pytest.approx(
        first.symmetrized_products.lambda_min_computed, rel=1e-12,
    )
    assert second.symmetrized_products.lambda_max_computed == pytest.approx(
        first.symmetrized_products.lambda_max_computed, rel=1e-12,
    )


def test_left_right_swap_is_the_same_quadratic_tensor():
    left, right, decoder = factors()
    first = cert.certify_quadratic_product_rank(left, right, decoder)
    second = cert.certify_quadratic_product_rank(right, left, decoder)
    assert second.symmetrized_products.lambda_min_computed == pytest.approx(
        first.symmetrized_products.lambda_min_computed,
    )


def test_schema_validation_fails_closed():
    left, right, decoder = factors()
    with pytest.raises(ValueError):
        cert.certify_quadratic_product_rank(left[:, :2], right, decoder)
    with pytest.raises(ValueError):
        cert.certify_quadratic_product_rank(left, right, decoder[:, :1])
