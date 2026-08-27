from __future__ import annotations

import torch

import early_mlp_state_complete_compiler_v2 as compiler


def _fixture() -> tuple[torch.Generator, torch.Tensor, torch.Tensor, torch.Tensor,
                        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260827)
    d_model, products, coefficients, rows = 13, 31, 5, 19
    left = torch.randn(products, d_model, generator=generator, dtype=torch.float64)
    right = torch.randn(products, d_model, generator=generator, dtype=torch.float64)
    down = torch.randn(d_model, products, generator=generator, dtype=torch.float64)
    bias = torch.randn(d_model, generator=generator, dtype=torch.float64)
    basis = torch.linalg.qr(
        torch.randn(d_model, coefficients, generator=generator, dtype=torch.float64),
        mode="reduced",
    ).Q
    z = torch.randn(rows, d_model, generator=generator, dtype=torch.float64)
    mo = torch.randn(rows, d_model, generator=generator, dtype=torch.float64)
    return generator, left, right, down, bias, basis, z, mo


def test_full_native_program_exactly_matches_projected_bilinear_mlp() -> None:
    _, left, right, down, bias, basis, z, _ = _fixture()
    state = compiler.project_native_weights(left, right, down, bias, basis)
    original = ((z @ left.T) * (z @ right.T)) @ down.T + bias
    predicted = compiler.native_projected_output(z, state)
    assert torch.allclose(predicted, original @ basis, atol=1e-10, rtol=1e-10)
    assert not any(value.data_ptr() == left.data_ptr() for value in state.values())


def test_state_complete_interface_replaces_only_admitted_physical_subspace() -> None:
    _, left, right, down, bias, basis, z, mo = _fixture()
    state = compiler.project_native_weights(left, right, down, bias, basis)
    projected = compiler.native_projected_output(z, state)
    coefficients = compiler.state_complete_coefficients(z, mo, basis, state)
    deployed = mo + compiler.physical_correction(coefficients, basis)
    expected = mo - (mo @ basis) @ basis.T + projected @ basis.T
    assert torch.allclose(deployed, expected, atol=1e-10, rtol=1e-10)


def test_native_canonicalization_removes_reciprocal_scale_and_preserves_swap() -> None:
    generator, left, right, _, _, _, z, _ = _fixture()
    q = torch.randn(left.shape[0], 7, generator=generator, dtype=torch.float64)
    scale = torch.linspace(0.2, 3.0, left.shape[0], dtype=torch.float64)
    a = compiler.canonicalize_native_terms(left, right, q)
    b = compiler.canonicalize_native_terms(left * scale[:, None],
                                           right / scale[:, None], q)
    c = compiler.canonicalize_native_terms(right, left, q)
    out_a = ((z @ a[0].T) * (z @ a[1].T)) @ a[2]
    out_b = ((z @ b[0].T) * (z @ b[1].T)) @ b[2]
    out_c = ((z @ c[0].T) * (z @ c[1].T)) @ c[2]
    assert torch.allclose(out_a, out_b, atol=1e-10, rtol=1e-10)
    assert torch.allclose(out_a, out_c, atol=1e-10, rtol=1e-10)
    assert torch.allclose(a[0].norm(dim=1), torch.ones(left.shape[0], dtype=torch.float64))
    assert torch.allclose(a[1].norm(dim=1), torch.ones(left.shape[0], dtype=torch.float64))
    for row in range(a[0].shape[0]):
        assert a[0][row, int(a[0][row].abs().argmax())] >= 0.0


def test_registered_native_and_corrected_affine_prices_include_live_state_cost() -> None:
    expected = {8: 92_736, 16: 111_680, 32: 149_568, 64: 225_344,
                128: 376_896, 256: 680_000, 4608: 10_985_536}
    for k, reals in expected.items():
        price = compiler.native_program_price(k, include_basis=True)
        assert price["total_reals"] == reals
        assert price["inference_multiplies_per_token"] == 147_456 + 2_368 * k
    affine = compiler.corrected_affine_price(64, include_basis=True)
    assert affine["total_reals"] == 153_920
    assert affine["inference_multiplies_per_token"] == 225_280


def test_empirical_fisher_loss_weights_suffix_read_direction() -> None:
    adjoint = torch.zeros(6, 4, dtype=torch.float64)
    adjoint[:, 0] = 2.0
    used_error = torch.zeros_like(adjoint)
    used_error[:, 0] = 1.0
    unused_error = torch.zeros_like(adjoint)
    unused_error[:, 1] = 1.0
    used = compiler.empirical_fisher_loss(used_error, adjoint)
    unused = compiler.empirical_fisher_loss(unused_error, adjoint)
    # The exact registered 0.05 Euclidean floor gives a ratio of 81 here.
    assert used > 80.0 * unused
    assert unused > 0.0


def test_empirical_fisher_floor_has_exact_registered_scaling() -> None:
    error = torch.arange(1, 13, dtype=torch.float64).view(3, 4)
    adjoint = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand_as(error).clone()
    observed = compiler.empirical_fisher_loss(error, adjoint, isotropic_floor=0.05)
    directional = (adjoint * error).sum(dim=1).square().mean()
    directional /= adjoint.square().sum(dim=1).mean()
    expected = directional + 0.05 * error.square().mean()
    assert torch.allclose(observed, expected)


def test_frozen_global_fisher_denominator_composes_across_minibatches() -> None:
    generator = torch.Generator().manual_seed(29)
    error = torch.randn(13, 4, generator=generator, dtype=torch.float64)
    adjoint = torch.randn(13, 4, generator=generator, dtype=torch.float64)
    denominator = adjoint.square().sum(dim=1).mean()
    full = compiler.empirical_fisher_loss(
        error, adjoint, directional_denominator=denominator
    )
    pieces = []
    counts = []
    for slc in (slice(0, 5), slice(5, 13)):
        pieces.append(compiler.empirical_fisher_loss(
            error[slc], adjoint[slc], directional_denominator=denominator
        ))
        counts.append(len(error[slc]))
    composed = sum(value * count for value, count in zip(pieces, counts)) / sum(counts)
    assert torch.allclose(full, composed)


def test_signed_output_gauge_preserves_physical_correction_with_live_mo() -> None:
    _, left, right, down, bias, basis, z, mo = _fixture()
    state = compiler.project_native_weights(left, right, down, bias, basis)
    signs = torch.tensor([1, -1, 1, -1, -1], dtype=torch.float64)
    moved, moved_basis = compiler.transport_signed_output_gauge(state, basis, signs)
    correction = compiler.physical_correction(
        compiler.state_complete_coefficients(z, mo, basis, state), basis
    )
    moved_correction = compiler.physical_correction(
        compiler.state_complete_coefficients(z, mo, moved_basis, moved), moved_basis
    )
    assert torch.equal(correction, moved_correction)


def test_contract_rejects_invalid_native_programs() -> None:
    _, left, right, down, bias, basis, _, _ = _fixture()
    try:
        compiler.project_native_weights(left, right, down[:, :-1], bias, basis)
    except ValueError as error:
        assert "dimensions" in str(error)
    else:
        raise AssertionError("invalid native down projection was accepted")
    try:
        compiler.native_program_price(7, include_basis=True)
    except ValueError as error:
        assert "registered" in str(error)
    else:
        raise AssertionError("unregistered native K was accepted")
