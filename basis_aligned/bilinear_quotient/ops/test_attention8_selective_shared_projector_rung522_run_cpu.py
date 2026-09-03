"""CPU/static tests for the fail-closed rung-522 scientific runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
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


def test_swap_arm_plan_has_explicit_complete_axes_and_no_reshape_assumption():
    rows = [31, 47, 52, 80, 91, 103]
    arms = RUN._swap_arm_plan(rows)
    assert len(arms) == 96
    assert len(set(arms)) == 96
    assert {arm.cell for arm in arms} == {
        "D0:forward", "D0:reverse", "D1:forward", "D1:reverse"
    }
    for ensemble, offset in (("D0", 0), ("D1", 4)):
        for direction in ("forward", "reverse"):
            subset = [
                arm for arm in arms
                if arm.ensemble == ensemble and arm.direction == direction
            ]
            assert [arm.map_index for arm in subset] == [
                value for value in range(offset, offset + 4) for _ in rows
            ]
            assert [arm.recipient_row for arm in subset] == rows * 4


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


def test_response_scoring_uses_rms_bootstrap_full_a8_comparator_and_exact_token_tier():
    positions = torch.arange(40)
    pairs = RUN.TargetPairs(
        "r.2.0.2",
        10 * RUN.TOKENS + positions,
        11 * RUN.TOKENS + positions,
        torch.zeros(40, dtype=torch.int64),
    )
    projected = torch.zeros((2, RUN.TOKENS))
    projected[0, positions] = 1.0
    projected[1, positions] = 0.1
    full = torch.zeros_like(projected)
    full[0, positions] = 1.0
    full[1, positions] = 0.5
    row_to_local = torch.full((1000,), -1, dtype=torch.int64)
    row_to_local[10] = 0
    row_to_local[11] = 1
    score = RUN.score_response_cell(
        projected,
        full,
        pairs,
        row_to_local,
        cell_id="validation:r.2.0.2:D0:forward:seed52200",
    )
    assert score["member_rms"] == 1
    assert score["control_rms"] == pytest.approx(0.1)
    assert score["concentration"] == pytest.approx(10)
    assert score["full_attention8_concentration"] == pytest.approx(2)
    assert score["concentration_improvement_over_full_attention8"] == pytest.approx(8)
    assert score["fourfold_margin_lower95"] == pytest.approx(0.6)
    assert score["exact_token_tier0_or1"]["pair_count"] == 40
    assert score["exact_token_tier0_or1"]["passes"]
    assert score["base_gates_pass"]


def test_fit_projection_mean_uses_every_fit_row_and_position_not_circuit_masks():
    writes = torch.zeros((2, 3, RUN.D), dtype=torch.float32)
    writes[..., : RUN.RANK] = torch.arange(
        2 * 3 * RUN.RANK, dtype=torch.float32
    ).view(2, 3, RUN.RANK)
    frame = torch.eye(RUN.D, dtype=torch.float32)[:, : RUN.RANK]
    fake = SimpleNamespace(captures={"fit": writes})
    mean = RUN.Rung522Instrument.fit_projection_mean(fake, frame)
    assert mean.tolist() == pytest.approx(
        writes[..., : RUN.RANK].double().mean((0, 1)).tolist()
    )


def test_fingerprint_uses_same_rms_coordinate_for_all_32_circuits():
    tags = RUN.stage_a.FINGERPRINT_TAGS
    response = torch.zeros((2 * len(tags), RUN.TOKENS))
    row_to_local = torch.full((1000,), -1, dtype=torch.int64)
    pairs = {}
    for index, target in enumerate(tags):
        member_row, control_row = 2 * index, 2 * index + 1
        row_to_local[member_row] = member_row
        row_to_local[control_row] = control_row
        pairs[target] = _pair(target, member_row, control_row, 0)
        response[member_row, 0] = 2.0 if target in RUN.QUARTET_TAGS else 1.0
    coordinates = RUN.fingerprint_coordinates(response, pairs, row_to_local)
    separation = RUN.quartet_separation(coordinates)
    assert len(coordinates) == 32
    assert separation["minimum_quartet_coordinate"] == 2
    assert separation["maximum_nonquartet_coordinate"] == 1
    assert separation["separation"] == 1


def test_scientific_main_retains_explicit_kill_switch_until_runner_is_complete():
    source = PATH.read_text()
    assert "RUNG522 SCIENCE CLOSED" in source
    assert "raise RuntimeError" in source[source.index("def main"):]
    for label in ("pred_a", "pred_b", "pred_c", "pred_d"):
        assert label in source
