"""Structural CPU tests for rung 524's planted runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


OPS = Path(__file__).parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUN = _load("attention8_direct_grassmann_optimizer_falsifier_rung524_run")
GM = _load("attention8_direct_grassmann_optimizer_falsifier_rung524_math")


def test_split_shapes_controls_and_targets_match_planted_computation():
    planted, readouts = RUN._planted_objects()
    split = RUN.build_split(
        "FIT", seed=RUN.FIT_SEED, examples=GM.FIT_EXAMPLES,
        planted=planted, readouts=readouts, ood=False,
    )
    assert split.member_inputs.shape == (3, 4, 96, 64)
    assert split.target_outputs.shape == (3, 4, 96, 12)
    assert split.control_inputs.shape == (3, 4, 96, 64)
    assert torch.allclose(split.control_inputs @ planted, torch.zeros(3, 4, 96, 4, dtype=torch.float64), atol=1e-12)
    expected = torch.einsum(
        "tmnd,tmod->tmno",
        (split.member_inputs @ planted) @ planted.mT,
        readouts,
    )
    assert torch.allclose(split.target_outputs, expected, atol=1e-12)


def test_objective_is_basis_invariant_and_planted_loss_is_zero():
    planted, readouts = RUN._planted_objects()
    split = RUN.build_split(
        "FIT", seed=RUN.FIT_SEED, examples=GM.FIT_EXAMPLES,
        planted=planted, readouts=readouts, ood=False,
    )
    scales = RUN.fixed_fit_scales(split)
    rotation = RUN._random_frame(9)[:GM.RANK]
    rotation = GM.canonical_qr(rotation)
    base = RUN.normalized_objective(planted, split, readouts, scales, RUN.TARGETS)
    rotated = RUN.normalized_objective(planted @ rotation, split, readouts, scales, RUN.TARGETS)
    assert float(base) < 1e-20
    assert torch.allclose(base, rotated, atol=1e-20)


def test_data_are_deterministic_and_ood_differs():
    planted, readouts = RUN._planted_objects()
    first = RUN.build_split(
        "VALIDATION", seed=RUN.VALIDATION_SEED, examples=GM.VALIDATION_EXAMPLES,
        planted=planted, readouts=readouts, ood=False,
    )
    second = RUN.build_split(
        "VALIDATION", seed=RUN.VALIDATION_SEED, examples=GM.VALIDATION_EXAMPLES,
        planted=planted, readouts=readouts, ood=False,
    )
    ood = RUN.build_split(
        "OOD", seed=RUN.OOD_SEED, examples=GM.OOD_EXAMPLES,
        planted=planted, readouts=readouts, ood=True,
    )
    assert torch.equal(first.member_inputs, second.member_inputs)
    assert not torch.equal(first.member_inputs, ood.member_inputs)


def test_seal_rejects_ood_before_pretest_pass():
    seal = RUN.SplitSeal()
    with pytest.raises(RuntimeError, match="failed pretest"):
        seal.open_ood({"pretest_passes": False})
    seal.open_ood({"pretest_passes": True})
    assert seal.ood_opened
    assert seal.requested == ["FIT", "VALIDATION", "OOD"]
