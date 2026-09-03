"""CPU tests for the preregistered rung-523 optimizer-repair rules."""

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


MATH = _load("attention8_projector_optimizer_repair_rung523_math")


def test_fixed_scales_use_every_eligible_fit_member_position_per_map():
    full = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 4.0], [6.0, 8.0]],
    ])
    mask = torch.zeros((4, 2), dtype=torch.bool)
    mask[1] = torch.tensor([True, False])
    mask[3] = torch.tensor([False, True])
    rows = torch.tensor([1, 3], dtype=torch.int64)
    scales = MATH.fixed_target_map_scales(full, mask, rows, epsilon=1e-12)
    assert torch.allclose(scales, torch.tensor([(1.0 + 16.0) / 2, (4.0 + 64.0) / 2]))


def test_explicit_denominator_controls_both_member_and_control_terms():
    loss = MATH.normalized_target_loss(
        torch.tensor([2.0]),
        torch.tensor([1.0]),
        torch.tensor([0.5]),
        denominator=2.0,
        control_coefficient=4.0,
    )
    assert float(loss) == pytest.approx(1.0)


def _healthy_fit(*, spike: float | None = None):
    losses = [2.0] * 20 + [1.5] * 160 + [1.0] * 20
    if spike is not None:
        losses[50] = spike
    return MATH.FitHealth(
        losses=losses,
        initial_common_validation=1.0,
        final_common_validation=0.9,
        orthonormality_error=1e-6,
        projector_distance=1.0,
    )


def test_cell_requires_all_fits_and_rejects_catastrophic_tail():
    good = MATH.score_candidate_cell([_healthy_fit() for _ in range(15)])
    assert good["passes"]
    too_many_spikes = [_healthy_fit(spike=101.0) for _ in range(4)]
    too_many_spikes += [_healthy_fit() for _ in range(11)]
    bad = MATH.score_candidate_cell(too_many_spikes)
    assert not bad["passes"]
    assert "more_than_three_losses_above_100" in bad["failures"]
    extreme = MATH.score_candidate_cell(
        [_healthy_fit(spike=1001.0)] + [_healthy_fit() for _ in range(14)]
    )
    assert not extreme["passes"]
    assert "one_or_more_losses_above_1000" in extreme["failures"]


def test_decision_table_uses_frozen_minimal_change_order():
    cells = {name: {"passes": False} for name in MATH.PROSPECTIVE_ARMS}
    assert MATH.adoption_decision(cells)["diagnosis"] == "raw_adam_through_qr_closed"
    cells[MATH.FIXED_LOW_LR]["passes"] = True
    assert MATH.adoption_decision(cells)["adopted_arm"] == MATH.FIXED_LOW_LR
    cells[MATH.ROW_LOW_LR]["passes"] = True
    assert MATH.adoption_decision(cells)["adopted_arm"] == MATH.ROW_LOW_LR
    cells[MATH.FIXED_HIGH_LR]["passes"] = True
    decision = MATH.adoption_decision(cells)
    assert decision["adopted_arm"] == MATH.FIXED_HIGH_LR
    assert decision["diagnosis"] == "both_single_changes_sufficient_scale_not_uniquely_identified"
