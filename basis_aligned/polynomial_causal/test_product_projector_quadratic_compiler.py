import pytest
import torch

import product_projector_quadratic_compiler as compiler


def orthonormal(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(matrix, mode="reduced").Q


def random_fixture(*, product=9, input_dimension=5, output=4, rank=3, seed=536):
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    left = torch.randn(product, input_dimension, generator=generator, dtype=dtype)
    right = torch.randn(product, input_dimension, generator=generator, dtype=dtype)
    down = torch.randn(output, product, generator=generator, dtype=dtype)
    basis = orthonormal(torch.randn(product, rank, generator=generator, dtype=dtype))
    states = torch.randn(17, input_dimension, generator=generator, dtype=dtype)
    return left, right, down, basis, states


def direct_projected_output(left, right, down, basis, states):
    product = (states @ left.T) * (states @ right.T)
    return product @ (basis @ basis.T) @ down.T


def test_random_projected_activation_equals_factorized_and_dense_quadratics():
    left, right, down, basis, states = random_fixture()
    compiled = compiler.compile_product_projector(left, right, down, basis)

    direct = direct_projected_output(left, right, down, basis, states)
    factorized = compiled.evaluate(states)
    dense = torch.einsum(
        "...i,oij,...j->...o", states, compiled.dense_quadratic_weights(), states
    )

    torch.testing.assert_close(factorized, direct, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(dense, direct, atol=1e-11, rtol=1e-11)


def test_donor_minus_recipient_interchange_compiles_exactly():
    left, right, down, basis, recipients = random_fixture(seed=540)
    donors = torch.flip(recipients, dims=(0,)) + 0.125
    compiled = compiler.compile_product_projector(left, right, down, basis)

    direct_delta = direct_projected_output(left, right, down, basis, donors)
    direct_delta -= direct_projected_output(left, right, down, basis, recipients)
    compiled_delta = compiled.evaluate(donors) - compiled.evaluate(recipients)

    torch.testing.assert_close(compiled_delta, direct_delta, atol=1e-11, rtol=1e-11)


def test_orthogonal_basis_rotation_changes_factors_but_not_projector_or_function():
    left, right, down, basis, states = random_fixture(rank=4, seed=556)
    rotation = orthonormal(
        torch.randn(4, 4, generator=torch.Generator().manual_seed(557), dtype=torch.float64)
    )
    original = compiler.compile_product_projector(left, right, down, basis)
    rotated = compiler.compile_product_projector(left, right, down, basis @ rotation)

    torch.testing.assert_close(rotated.projector, original.projector, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        rotated.dense_quadratic_weights(),
        original.dense_quadratic_weights(),
        atol=1e-11,
        rtol=1e-11,
    )
    torch.testing.assert_close(rotated.evaluate(states), original.evaluate(states), atol=1e-11, rtol=1e-11)


def test_symmetric_q_l_matches_manual_formula():
    left, right, down, basis, _ = random_fixture(product=7, input_dimension=4, rank=2)
    compiled = compiler.compile_product_projector(left, right, down, basis)

    manual = []
    for column in range(basis.shape[1]):
        ordered = left.T @ torch.diag(basis[:, column]) @ right
        manual.append(0.5 * (ordered + ordered.T))
    manual = torch.stack(manual)

    torch.testing.assert_close(compiled.quadratic_forms, manual)
    torch.testing.assert_close(
        compiled.quadratic_forms,
        compiled.quadratic_forms.transpose(-1, -2),
    )
    torch.testing.assert_close(compiled.output_directions, down @ basis)


def test_planted_coordinate_subspace_has_expected_closed_form():
    dtype = torch.float64
    left = torch.tensor([[1.0, 2.0], [3.0, -1.0], [2.0, 4.0]], dtype=dtype)
    right = torch.tensor([[5.0, -2.0], [1.0, 6.0], [-3.0, 2.0]], dtype=dtype)
    down = torch.tensor([[2.0, 7.0, -1.0], [-4.0, 3.0, 5.0]], dtype=dtype)
    basis = torch.tensor([[0.0], [1.0], [0.0]], dtype=dtype)
    states = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=dtype)

    compiled = compiler.compile_product_projector(left, right, down, basis)
    expected_q = 0.5 * (
        torch.outer(left[1], right[1]) + torch.outer(right[1], left[1])
    )

    torch.testing.assert_close(compiled.quadratic_forms[0], expected_q)
    torch.testing.assert_close(compiled.output_directions[:, 0], down[:, 1])
    torch.testing.assert_close(
        compiled.evaluate(states),
        direct_projected_output(left, right, down, basis, states),
    )


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_proper_low_rank_projectors_including_zero_rank(rank):
    left, right, down, _, states = random_fixture(product=8, rank=1, seed=600 + rank)
    raw = torch.randn(
        8, rank, generator=torch.Generator().manual_seed(700 + rank), dtype=torch.float64
    )
    basis = orthonormal(raw) if rank else raw
    compiled = compiler.compile_product_projector(left, right, down, basis)

    assert compiled.quadratic_forms.shape == (rank, 5, 5)
    assert compiled.output_directions.shape == (4, rank)
    torch.testing.assert_close(
        compiled.evaluate(states),
        direct_projected_output(left, right, down, basis, states),
        atol=1e-11,
        rtol=1e-11,
    )


def test_nonorthonormal_basis_is_rejected_by_default():
    left, right, down, basis, _ = random_fixture(rank=2)
    distorted = basis @ torch.tensor([[2.0, 0.3], [0.0, 0.5]], dtype=torch.float64)
    with pytest.raises(ValueError, match="must be orthonormal"):
        compiler.compile_product_projector(left, right, down, distorted)


def test_explicit_normalization_compiles_column_span_and_drops_dependent_column():
    left, right, down, basis, states = random_fixture(rank=2)
    # Three supplied columns span only the original two-dimensional subspace.
    dependent = torch.column_stack((2.0 * basis[:, 0], basis[:, 1], basis[:, 0] + basis[:, 1]))
    compiled = compiler.compile_product_projector(
        left, right, down, dependent, normalize_basis=True
    )

    assert compiled.basis.shape == (left.shape[0], 2)
    torch.testing.assert_close(
        compiled.basis.T @ compiled.basis,
        torch.eye(2, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(compiled.projector, basis @ basis.T, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(
        compiled.evaluate(states),
        direct_projected_output(left, right, down, basis, states),
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    "mutation,exception",
    [
        (lambda left, right, down, basis: (left, right[:-1], down, basis), ValueError),
        (lambda left, right, down, basis: (left, right, down[:, :-1], basis), ValueError),
        (lambda left, right, down, basis: (left, right, down, basis[:-1]), ValueError),
        (lambda left, right, down, basis: (left, right, down, basis.to(torch.float32)), TypeError),
        (lambda left, right, down, basis: (left, right, down, basis.fill_(float("nan"))), ValueError),
    ],
)
def test_malformed_weights_fail_closed(mutation, exception):
    left, right, down, basis, _ = random_fixture()
    arguments = mutation(left, right, down, basis.clone())
    with pytest.raises(exception):
        compiler.compile_product_projector(*arguments)
