"""CPU/static tests for the fail-closed rung-522 scientific runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


PATH = Path(__file__).with_name("attention8_selective_shared_projector_rung522_run.py")
SPEC = importlib.util.spec_from_file_location("rung522_partial_runner", PATH)
RUN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUN
SPEC.loader.exec_module(RUN)


def _pair(target: str, member_row: int, control_row: int, position: int):
    return RUN.TargetPairs(
        target,
        torch.tensor([member_row * RUN.TOKENS + position]),
        torch.tensor([control_row * RUN.TOKENS + position]),
        torch.tensor([0]),
    )


class _FakeInstrument:
    def __init__(self, rows):
        self.split_rows = {"fit": rows, "validation": rows}
        row_to_local = torch.full((1000,), -1, dtype=torch.int64)
        row_to_local[rows] = torch.arange(rows.numel())
        self.row_to_local = {"fit": row_to_local, "validation": row_to_local}
        maps = [torch.tensor(index) for index in range(4)]
        self.design = {"donors": {"fit": {"maps": maps}, "validation": {"maps": maps}}}
        self.calls = []

    def projected_delta(self, split, rows, donor_map, frame, *, optimization):
        self.calls.append((split, rows.tolist(), int(donor_map), optimization))
        values = torch.arange(rows.numel() * RUN.TOKENS, dtype=frame.dtype).view(
            rows.numel(), RUN.TOKENS
        )
        return values + frame.sum() * 0, torch.ones(rows.numel())


def test_callback_keeps_roles_targets_and_map_axis_explicit():
    omitted = RUN.FITTED_TAGS[0]
    spec = RUN.state_guard.EXPECTED_FRAME_SPECS[
        f"real_leave_one_out:{omitted}:52200"
    ]
    rows = torch.tensor([10, 11, 12, 13])
    first, second = spec.training_targets
    pairs = {
        first: _pair(first, 10, 11, 3),
        second: _pair(second, 12, 13, 7),
    }
    row_mask = torch.zeros(1000, dtype=torch.bool)
    row_mask[rows] = True
    balanced = RUN._make_balanced_scheduler(spec, pairs, row_mask)
    full = torch.empty((4, 4, RUN.TOKENS))
    for map_index in range(4):
        for local in range(4):
            full[map_index, local] = 10_000 * map_index + 100 * local + torch.arange(RUN.TOKENS)
    instrument = _FakeInstrument(rows)
    callback = RUN.ProjectedResponseCallback(
        instrument,
        spec,
        split="fit",
        pairs=pairs,
        balanced=balanced,
        full_by_map=full,
        optimization=True,
    )
    response = callback(torch.eye(RUN.D)[:, : RUN.RANK], 2)
    assert tuple(sorted(response)) == tuple(sorted(spec.training_targets))
    assert instrument.calls[0][2:] == (2, True)
    assert response[first].full_member.tolist() == [20_003]
    assert response[second].full_member.tolist() == [20_207]
    assert response[first].projected_member.tolist() == [3]
    assert response[first].projected_control.tolist() == [RUN.TOKENS + 3]
    assert response[second].projected_member.tolist() == [2 * RUN.TOKENS + 7]
    assert response[second].projected_control.tolist() == [3 * RUN.TOKENS + 7]


def test_health_callback_ignores_minus_one_and_uses_validation_d0_map_zero():
    omitted = RUN.FITTED_TAGS[1]
    spec = RUN.state_guard.EXPECTED_FRAME_SPECS[
        f"real_leave_one_out:{omitted}:52201"
    ]
    rows = torch.tensor([20, 21, 22, 23])
    first, second = spec.training_targets
    pairs = {
        first: _pair(first, 20, 21, 1),
        second: _pair(second, 22, 23, 2),
    }
    row_mask = torch.zeros(1000, dtype=torch.bool)
    row_mask[rows] = True
    balanced = RUN._make_balanced_scheduler(spec, pairs, row_mask)
    instrument = _FakeInstrument(rows)
    callback = RUN.ProjectedResponseCallback(
        instrument,
        spec,
        split="validation",
        pairs=pairs,
        balanced=balanced,
        full_by_map=torch.ones((1, 4, RUN.TOKENS)),
        optimization=False,
        fixed_health_batch=True,
    )
    callback(torch.eye(RUN.D)[:, : RUN.RANK], -1)
    assert instrument.calls[0][0] == "validation"
    assert instrument.calls[0][1] == [role.row_index for role in balanced.batch(0).roles]
    assert instrument.calls[0][2:] == (0, False)


def test_scientific_main_retains_explicit_kill_switch_until_runner_is_complete():
    source = PATH.read_text()
    assert "RUNG522 SCIENCE CLOSED" in source
    assert "raise RuntimeError" in source[source.index("def main"):]
    for label in ("pred_a", "pred_b", "pred_c", "pred_d"):
        assert label in source
