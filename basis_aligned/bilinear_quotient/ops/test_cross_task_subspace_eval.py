import torch

import cross_task_subspace_eval as subject


def test_energy_basis_and_projection_are_exact_and_gauge_invariant():
    g = torch.Generator().manual_seed(10)
    matrix = torch.randn(20, 6, generator=g)
    basis, _singular, explained = subject.energy_basis(matrix, retained=0.8)
    assert subject.projection_energy(matrix, basis) >= 0.8
    rotation = torch.linalg.qr(torch.randn(basis.shape[1], basis.shape[1], generator=g)).Q
    assert abs(subject.projection_energy(matrix, basis @ rotation) - explained) < 1e-5


def test_principal_cosines_and_shared_midpoints_find_only_shared_modes():
    eye = torch.eye(5)
    first = eye[:, :2]
    second = torch.stack((eye[:, 0], eye[:, 2]), dim=1)
    cosines = subject.principal_cosines(first, second)
    shared, reported = subject.shared_midpoint_basis(first, second, cosine_threshold=0.8)
    assert torch.allclose(cosines, torch.tensor([1.0, 0.0]))
    assert torch.allclose(reported, cosines)
    assert shared.shape == (5, 1)
    assert abs(float(shared[:, 0] @ eye[:, 0])) > 0.999
