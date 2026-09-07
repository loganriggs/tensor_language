import torch

import task_mode_hidden_groups as h


def test_canonical_pair_axes_are_orthonormal_and_separate_shared_from_contrast():
    temporal = torch.eye(6)[:, :2]
    iswas = torch.stack((torch.tensor([.8, 0., .6, 0., 0., 0.]), torch.tensor([0., 0., 0., 1., 0., 0.])), dim=1)
    result = h.canonical_pair_directions(torch, temporal, iswas)
    joined = torch.cat((result["mean"], result["contrast"]), dim=1)
    assert torch.allclose(joined.T @ joined, torch.eye(4), atol=1e-6)
    assert torch.allclose(result["principal_cosines"], torch.tensor([.8, 0.]), atol=1e-6)


def test_participation_is_invariant_to_parent_coordinate_gauge():
    generator = torch.Generator().manual_seed(7)
    delta = torch.randn(9, 11, generator=generator)
    weight = torch.randn(11, 6, generator=generator)
    temporal = torch.linalg.qr(torch.randn(6, 2, generator=generator)).Q
    iswas = torch.linalg.qr(torch.randn(6, 2, generator=generator)).Q
    directions = h.canonical_pair_directions(torch, temporal, iswas)
    rotation = torch.linalg.qr(torch.randn(6, 6, generator=generator)).Q
    rotated = h.canonical_pair_directions(torch, rotation.T @ temporal, rotation.T @ iswas)
    first = h.participation_energy(torch, delta, weight, directions)
    second = h.participation_energy(torch, delta, weight @ rotation, rotated)
    assert all(torch.allclose(first[key], second[key], atol=2e-5, rtol=2e-5) for key in first)


def test_bad_shapes_fail_closed():
    try:
        h.participation_energy(torch, torch.zeros(2, 3), torch.zeros(4, 2), {"mean": torch.eye(2)})
    except ValueError:
        pass
    else:
        raise AssertionError("hidden mismatch should fail")
