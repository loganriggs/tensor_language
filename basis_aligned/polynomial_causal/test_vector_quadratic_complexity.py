import numpy as np

from vector_quadratic_complexity import (
    certificate_dict,
    evaluate_product_factors,
    evaluate_tensor,
    product_bounds,
    tensor_from_product_factors,
)


def test_factor_tensor_evaluates_exactly():
    rng = np.random.default_rng(2)
    c = rng.normal(size=(3, 5))
    a = rng.normal(size=(5, 4))
    b = rng.normal(size=(5, 4))
    x = rng.normal(size=(11, 4))
    tensor = tensor_from_product_factors(c, a, b)
    np.testing.assert_allclose(evaluate_tensor(tensor, x), evaluate_product_factors(c, a, b, x), atol=1e-11)


def test_indefinite_rank_two_scalar_needs_one_product():
    c = np.ones((1, 1))
    a = np.array([[1.0, 1.0]])
    b = np.array([[1.0, -1.0]])
    tensor = tensor_from_product_factors(c, a, b)
    bounds = product_bounds(tensor, explicit_products=1, output_directions=[np.ones(1)])
    assert bounds.certified_lower == 1
    assert certificate_dict(bounds)["status"] == "minimum_certified"


def test_independent_vector_squares_need_one_product_each():
    eye = np.eye(3)
    tensor = tensor_from_product_factors(eye, eye, eye)
    bounds = product_bounds(tensor, explicit_products=3, output_directions=np.eye(3))
    assert bounds.output_flattening_lower == 3
    assert bounds.certified_lower == 3
    assert certificate_dict(bounds)["status"] == "minimum_certified"


def test_shared_product_across_outputs_is_priced_once():
    c = np.array([[1.0], [2.0], [-4.0]])
    a = np.array([[1.0, 3.0]])
    b = np.array([[2.0, -1.0]])
    tensor = tensor_from_product_factors(c, a, b)
    bounds = product_bounds(tensor, explicit_products=1, output_directions=np.eye(3))
    assert bounds.output_flattening_lower == 1
    assert bounds.certified_lower == 1


def test_bounds_are_invariant_under_input_and_output_gauges():
    rng = np.random.default_rng(4)
    c = rng.normal(size=(3, 4))
    a = rng.normal(size=(4, 3))
    b = rng.normal(size=(4, 3))
    tensor = tensor_from_product_factors(c, a, b)
    base = product_bounds(tensor)
    output_gauge = rng.normal(size=(3, 3)) + 3 * np.eye(3)
    input_gauge = rng.normal(size=(3, 3)) + 3 * np.eye(3)
    transformed = np.einsum("zy,yij,ip,jq->zpq", output_gauge, tensor, input_gauge, input_gauge)
    changed = product_bounds(transformed)
    assert changed.output_flattening_lower == base.output_flattening_lower
    assert changed.input_flattening_lower == base.input_flattening_lower
