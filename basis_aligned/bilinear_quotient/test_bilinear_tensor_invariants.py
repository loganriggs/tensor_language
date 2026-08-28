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
