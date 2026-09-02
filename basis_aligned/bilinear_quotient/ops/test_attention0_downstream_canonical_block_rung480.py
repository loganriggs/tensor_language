"""CPU algebra tests for rung480's affine and gauge-covariant objects."""

from __future__ import annotations

import torch

import attention0_downstream_canonical_block_rung480 as rung


def test_affine_basis_reconstructs_projector():
    generator = torch.Generator().manual_seed(4801)
    for width, rank in ((9, 6), (144, 32)):
        mean = torch.randn(1, width, generator=generator)
        basis = torch.linalg.qr(torch.randn(width, rank, generator=generator)).Q
        value = torch.randn(23, width, generator=generator)
        block = {"mean1": mean, "basis1": basis,
                 "mean2": mean, "basis2": basis,
                 "meanv": mean, "basisv": basis}
        _, _, augmented = rung._affine_parts(block)[0]
        coordinates = torch.cat((torch.ones(len(value), 1), (value - mean) @ basis), 1)
        expected = mean + ((value - mean) @ basis) @ basis.T
        assert torch.allclose(coordinates @ augmented.T, expected, atol=2e-5, rtol=2e-5)


def test_response_operator_conjugates():
    assert rung._rotation_identity() <= 1e-20


def test_projector_is_rank_one_and_sign_invariant():
    generator = torch.Generator().manual_seed(4802)
    operators = torch.randn(17, 6, 6, generator=generator, dtype=torch.float64)
    operators = (operators + operators.transpose(-1, -2)) / 2
    projector, _ = rung._projector(operators)
    assert torch.allclose(projector @ projector, projector, atol=1e-10, rtol=1e-10)
    assert abs(float(torch.trace(projector)) - 1) <= 1e-10
    profile = rung._profile(operators, projector)
    assert abs(float(profile.mean())) <= 1e-12


def test_payload_head_output_axes_are_flattened_for_coordinates():
    generator = torch.Generator().manual_seed(4803)
    block = {}
    for suffix, width, rank in (("1", 9, 6), ("2", 9, 6), ("v", 144, 32)):
        block[f"mean{suffix}"] = torch.randn(1, width, generator=generator)
        block[f"basis{suffix}"] = torch.linalg.qr(
            torch.randn(width, rank, generator=generator)).Q
    score1 = torch.randn(2, 9, 3, 3, generator=generator)
    score2 = torch.randn(2, 9, 3, 3, generator=generator)
    tokens = torch.tensor([[0, 1, 2], [2, 1, 0]])
    payload = torch.randn(3, 9, 16, generator=generator)
    coordinates, core = rung._coordinates(score1, score2, tokens, payload, block)
    assert coordinates[0].shape == (2, 3, 3, 7)
    assert coordinates[1].shape == (2, 3, 3, 7)
    assert coordinates[2].shape == (2, 3, 33)
    assert core.shape == (7, 7, 33, 16)


if __name__ == "__main__":
    test_affine_basis_reconstructs_projector()
    test_response_operator_conjugates()
    test_projector_is_rank_one_and_sign_invariant()
    test_payload_head_output_axes_are_flattened_for_coordinates()
    print("rung480 algebra tests passed")
