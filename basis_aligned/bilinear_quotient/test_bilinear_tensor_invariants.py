import torch

from . import bilinear_tensor_invariants as inv


def factors(seed=0, din=5, components=4, dout=3):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(din, components, generator=generator, dtype=torch.double),
            torch.randn(din, components, generator=generator, dtype=torch.double),
            torch.randn(components, dout, generator=generator, dtype=torch.double))


def test_factor_gram_and_output_gram_equal_explicit_tensor():
    A, B, C = factors()
    tensor = inv.explicit_symmetric_tensor(A, B, C)
    flat = tensor.permute(2, 0, 1).reshape(C.shape[1], -1)
    assert torch.allclose(inv.output_unfolding_gram(A, B, C), flat@flat.T,
                          atol=1e-10, rtol=1e-10)
    assert abs(inv.tensor_frobenius_sq(A, B, C)-float(tensor.square().sum())) < 1e-10


def test_cross_inner_product_and_relative_error_equal_dense_oracle():
    A1, B1, C1 = factors(seed=10)
    A2, B2, C2 = factors(seed=11)
    T1 = inv.explicit_symmetric_tensor(A1, B1, C1)
    T2 = inv.explicit_symmetric_tensor(A2, B2, C2)
    assert abs(inv.tensor_inner_product(A1, B1, C1, A2, B2, C2)
               - float((T1*T2).sum())) < 1e-10
    expected = float((T1-T2).norm()/T1.norm())
    assert abs(inv.relative_tensor_frobenius_error(
        A1, B1, C1, A2, B2, C2)-expected) < 1e-10
    assert abs(inv.tensor_frobenius_error(A1, B1, C1, A2, B2, C2)
               - float((T1-T2).norm())) < 1e-10


def test_quadratic_jvp_and_rms_sphere_composition_bound():
    A1, B1, C1 = factors(seed=20)
    A2, B2, C2 = factors(seed=21)
    generator = torch.Generator().manual_seed(22)
    z = torch.randn(6, A1.shape[0], generator=generator, dtype=torch.double)
    direction = torch.randn(z.shape, generator=generator, dtype=torch.double)
    epsilon = 1e-5
    finite_difference = (inv.execute_quadratic(A1, B1, C1, z+epsilon*direction)
                         - inv.execute_quadratic(A1, B1, C1, z-epsilon*direction))/(2*epsilon)
    assert torch.allclose(inv.quadratic_jvp(A1, B1, C1, z, direction),
                          finite_difference, atol=1e-8, rtol=1e-8)
    jacobian = inv.quadratic_jacobian(A1, B1, C1, z)
    assert torch.allclose(torch.einsum("bi,bio->bo", direction, jacobian),
                          inv.quadratic_jvp(A1, B1, C1, z, direction))
    radius = A1.shape[0]**.5
    z1 = radius*z/z.norm(dim=-1, keepdim=True)
    z2 = radius*(z+.2*direction)/(z+.2*direction).norm(dim=-1, keepdim=True)
    residual1 = inv.execute_quadratic(A1, B1, C1, z1) \
        - inv.execute_quadratic(A2, B2, C2, z1)
    residual2 = inv.execute_quadratic(A1, B1, C1, z2) \
        - inv.execute_quadratic(A2, B2, C2, z2)
    bound = inv.rms_sphere_residual_lipschitz_bound(A1, B1, C1, A2, B2, C2) \
        * (z2-z1).norm(dim=-1)
    assert bool(((residual2-residual1).norm(dim=-1) <= bound+1e-10).all())
    spectral_bound = inv.rms_sphere_residual_spectral_bound(
        A1, B1, C1, A2, B2, C2)*(z2-z1).norm(dim=-1)
    assert bool(((residual2-residual1).norm(dim=-1) <= spectral_bound+1e-10).all())
    assert bool((spectral_bound <= bound+1e-10).all())
    local = inv.midpoint_residual_lipschitz_bound(
        A1, B1, C1, A2, B2, C2, z1, z2)*(z2-z1).norm(dim=-1)
    assert bool(((residual2-residual1).norm(dim=-1) <= local+1e-10).all())


def test_residual_unfolding_spectral_norm_matches_dense_oracle():
    A1, B1, C1 = factors(seed=30)
    A2, B2, C2 = factors(seed=31)
    residual = inv.explicit_symmetric_tensor(A1, B1, C1) \
        - inv.explicit_symmetric_tensor(A2, B2, C2)
    expected = torch.linalg.matrix_norm(residual.reshape(-1, residual.shape[-1]), 2)
    actual = inv.residual_output_unfolding_spectral_norm(A1, B1, C1, A2, B2, C2)
    assert abs(actual-float(expected)) < 1e-10


def test_invariants_survive_factor_gauges_leg_swaps_and_permutation():
    A, B, C = factors(seed=1)
    reference = inv.output_unfolding_gram(A, B, C)
    alpha = torch.tensor([2., -.5, 3., -4.], dtype=torch.double)
    beta = torch.tensor([-.25, 5., -2., .125], dtype=torch.double)
    A2 = A*alpha; B2 = B*beta; C2 = C/(alpha*beta)[:, None]
    A2[:, [0, 2]], B2[:, [0, 2]] = B2[:, [0, 2]].clone(), A2[:, [0, 2]].clone()
    permutation = torch.tensor([2, 0, 3, 1])
    A2, B2, C2 = A2[:, permutation], B2[:, permutation], C2[permutation]
    assert torch.allclose(inv.output_unfolding_gram(A2, B2, C2), reference,
                          atol=1e-10, rtol=1e-10)


def test_output_spectrum_matches_dense_svd():
    A, B, C = factors(seed=2)
    tensor = inv.explicit_symmetric_tensor(A, B, C)
    flat = tensor.permute(2, 0, 1).reshape(C.shape[1], -1)
    result = inv.output_mode_spectrum(A, B, C)
    expected = torch.linalg.svdvals(flat)
    assert torch.allclose(result["singular_values"], expected,
                          atol=1e-10, rtol=1e-10)
    assert 1 <= result["stable_rank"] <= result["rank"]
    assert 1 <= result["entropy_rank"] <= result["rank"]


def test_antisymmetric_factor_change_is_observationally_removed():
    # Swapping the two input legs changes a*b^T but not z^T(a*b^T)z.
    A, B, C = factors(seed=3)
    assert torch.equal(inv.explicit_symmetric_tensor(A, B, C),
                       inv.explicit_symmetric_tensor(B, A, C))


def test_energy_rank_and_eckart_young_tail_are_exact():
    singular = torch.tensor([4., 3., 0.], dtype=torch.double)
    assert inv.energy_rank(singular, .60) == 1
    assert inv.energy_rank(singular, .70) == 2
    assert inv.energy_rank(singular, 1.0) == 2
    assert abs(inv.best_rank_relative_frobenius_error(singular, 1)-3/5) < 1e-12
    assert inv.best_rank_relative_frobenius_error(singular, 2) == 0


def test_energy_majorization_distinguishes_dominance_and_crossing():
    dominant = inv.energy_majorization(torch.tensor([8., 2.]).sqrt(),
                                       torch.tensor([5., 5.]).sqrt())
    assert dominant["relation"] == "left_majorizes_right"
    crossing = inv.energy_majorization(torch.tensor([6., 2., 2.]).sqrt(),
                                       torch.tensor([5., 5.]).sqrt())
    assert crossing["relation"] == "incomparable_crossing"
    assert crossing["strict_sign_crossings"] >= 1
