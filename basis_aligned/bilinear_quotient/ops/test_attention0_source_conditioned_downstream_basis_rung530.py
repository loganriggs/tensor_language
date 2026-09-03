from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).with_name("attention0_source_conditioned_downstream_basis_rung530.py")
SPEC = importlib.util.spec_from_file_location("r530", PATH)
R = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(R)


def test_projector_and_profile_are_rotation_covariant():
    generator = torch.Generator().manual_seed(530)
    operators = torch.randn(32, 6, 6, generator=generator, dtype=torch.float64)
    operators = (operators + operators.transpose(-1, -2)) / 2
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=generator, dtype=torch.float64))
    p, _ = R.projector(operators)
    rotated = torch.einsum("ai,cij,jb->cab", q.T, operators, q)
    p_rotated, _ = R.projector(rotated)
    assert R.projector_overlap(q.T @ p @ q, p_rotated) == pytest.approx(1.0, abs=1e-10)
    assert torch.allclose(R.profile(operators, p), R.profile(rotated, p_rotated), atol=1e-10)


def test_contrasts_recover_member_minus_control_means():
    counts = torch.ones(2, 2, 32, dtype=torch.float64)
    counts[:, 0] = 2
    family = torch.zeros(2, 2, 2, 32, 6, 6, dtype=torch.float64)
    family[:, :, 0] = 6
    family[:, :, 1] = 1
    result = R.contrasts((family, family, torch.zeros(2, 2, 2, 32, 32, 32)), counts)
    assert torch.equal(result[0], torch.full((2, 2, 32, 6, 6), 2.0, dtype=torch.float64))


def test_root_parser_is_fail_closed():
    assert R.root_of("r.18.2.0") == 18
    with pytest.raises(ValueError, match="malformed"):
        R.root_of("x.18.2")
