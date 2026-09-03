"""CPU structural tests for the rung 525 managed runner."""

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


RUN = _load("mlp0_token_context_operator_quotient_rung525_run")


def test_derangement_is_per_token_permutation_and_changes_coupling():
    values = torch.arange(10 * 256, dtype=torch.float32).reshape(10, 256)
    changed = RUN._derange_coordinates(values)
    assert not torch.equal(values, changed)
    for token in range(len(values)):
        for block in range(4):
            sl = slice(block * 64, (block + 1) * 64)
            assert torch.equal(values[token, sl].sort().values, changed[token, sl].sort().values)


def test_far_random_controls_respect_cosine_ceiling():
    generator = torch.Generator().manual_seed(8)
    raw = torch.randn(80, 20, generator=generator)
    receivers = torch.arange(0, 20, dtype=torch.int64)
    donors = torch.arange(20, 80, dtype=torch.int64)
    selected, cosines = RUN._fixed_far_random_controls(
        raw, receivers, donors, seed=4, count=16
    )
    assert selected.shape == (20, 16)
    assert bool((cosines <= 0.5).all())


def test_probe_rows_are_unique_and_deterministic():
    context = torch.arange(3 * 256 * RUN.D, dtype=torch.float32).reshape(3, 256, RUN.D)
    first, ids1 = RUN._probe_rows(context, 12)
    second, ids2 = RUN._probe_rows(context, 12)
    assert first.shape == (256, RUN.D)
    assert torch.equal(first, second)
    assert torch.equal(ids1, ids2)
    assert len(ids1.unique()) == 256


def test_unknown_probe_shape_is_rejected():
    with pytest.raises(ValueError, match="context rows"):
        RUN._probe_rows(torch.zeros(3, 255, RUN.D), 1)
