"""CPU-only tests for the frozen rung-522 balanced-row schedule."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PATH = Path(__file__).with_name(
    "attention8_selective_shared_projector_rung522_scheduler.py"
)
SPEC = importlib.util.spec_from_file_location("rung522_scheduler", PATH)
SCHED = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCHED
SPEC.loader.exec_module(SCHED)


TARGETS = ("r.2.0.2", "r.2.1.1", "r.2.2.1")
MEMBERS = {
    TARGETS[0]: (10, 11, 12, 13, 14),
    TARGETS[1]: (20, 21, 22, 23, 24, 25),
    TARGETS[2]: (30, 31, 32, 33, 34, 35, 36),
}
CONTROLS = {
    TARGETS[0]: (110, 111, 112, 113),
    TARGETS[1]: (120, 121, 122, 123, 124),
    TARGETS[2]: (130, 131, 132, 133, 134, 135),
}


def _assert_exhaustion_before_cycle(scheduler):
    for role in scheduler.roles:
        observed = tuple(
            batch.rows_by_role()[role.name]
            for batch in (scheduler.batch(update) for update in range(len(role.permutation)))
        )
        assert observed == role.permutation
        assert len(set(observed)) == len(observed)
        assert scheduler.batch(len(role.permutation)).rows_by_role()[role.name] == observed[0]


def test_two_target_schedule_has_four_balanced_roles_and_cycles_maps():
    scheduler = SCHED.two_target_scheduler(
        TARGETS[:2], MEMBERS, CONTROLS, seed=52200
    )
    assert scheduler.mode == "two_target"
    batch = scheduler.batch(0)
    assert len(batch.roles) == 4
    assert [(role.target, role.kind, role.replica) for role in batch.roles] == [
        (TARGETS[0], "member", 0),
        (TARGETS[0], "control", 0),
        (TARGETS[1], "member", 0),
        (TARGETS[1], "control", 0),
    ]
    assert [scheduler.batch(update).donor_map_index for update in range(10)] == [
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1
    ]
    _assert_exhaustion_before_cycle(scheduler)


def test_single_target_oracle_has_two_member_and_two_control_roles():
    scheduler = SCHED.single_target_oracle_scheduler(
        TARGETS[0], MEMBERS, CONTROLS, seed=52201
    )
    roles = scheduler.batch(0).roles
    assert len(roles) == 4
    assert [(role.kind, role.replica) for role in roles] == [
        ("member", 0),
        ("member", 1),
        ("control", 0),
        ("control", 1),
    ]
    assert {role.target for role in roles} == {TARGETS[0]}
    for update in range(40):
        batch_roles = scheduler.batch(update).roles
        assert batch_roles[0].row_index != batch_roles[1].row_index
        assert batch_roles[2].row_index != batch_roles[3].row_index
    _assert_exhaustion_before_cycle(scheduler)


def test_all_three_has_exactly_one_member_and_control_role_per_target():
    scheduler = SCHED.all_three_scheduler(
        TARGETS, MEMBERS, CONTROLS, seed=52202
    )
    roles = scheduler.batch(0).roles
    assert len(roles) == 6
    for target in TARGETS:
        assert sum(role.target == target and role.kind == "member" for role in roles) == 1
        assert sum(role.target == target and role.kind == "control" for role in roles) == 1
    _assert_exhaustion_before_cycle(scheduler)


def test_same_seed_is_identical_and_role_permutations_are_seed_defined():
    first = SCHED.all_three_scheduler(TARGETS, MEMBERS, CONTROLS, seed=52203)
    repeat = SCHED.all_three_scheduler(TARGETS, MEMBERS, CONTROLS, seed=52203)
    other = SCHED.all_three_scheduler(TARGETS, MEMBERS, CONTROLS, seed=52204)
    assert first.fingerprint == repeat.fingerprint
    assert [first.batch(step) for step in range(30)] == [
        repeat.batch(step) for step in range(30)
    ]
    assert first.fingerprint != other.fingerprint
    assert any(
        first.role_permutation(name) != other.role_permutation(name)
        for name in first.role_names
    )


@pytest.mark.parametrize(
    "factory,targets",
    [
        (SCHED.two_target_scheduler, TARGETS[:2]),
        (SCHED.all_three_scheduler, TARGETS),
    ],
)
def test_multi_target_schedules_fail_when_any_member_or_control_role_is_empty(
    factory, targets
):
    empty_members = dict(MEMBERS)
    empty_members[targets[-1]] = ()
    with pytest.raises(ValueError, match="no eligible row"):
        factory(targets, empty_members, CONTROLS, seed=52200)
    empty_controls = dict(CONTROLS)
    empty_controls[targets[0]] = ()
    with pytest.raises(ValueError, match="no eligible row"):
        factory(targets, MEMBERS, empty_controls, seed=52200)


def test_oracle_fails_when_either_shared_pool_is_empty():
    with pytest.raises(ValueError, match="no eligible row"):
        SCHED.single_target_oracle_scheduler(
            TARGETS[0], {**MEMBERS, TARGETS[0]: ()}, CONTROLS, seed=52200
        )
    with pytest.raises(ValueError, match="no eligible row"):
        SCHED.single_target_oracle_scheduler(
            TARGETS[0], MEMBERS, {**CONTROLS, TARGETS[0]: ()}, seed=52200
        )
    with pytest.raises(ValueError, match="at least two rows"):
        SCHED.single_target_oracle_scheduler(
            TARGETS[0], {**MEMBERS, TARGETS[0]: (1,)}, CONTROLS, seed=52200
        )


def test_scheduler_is_import_safe_and_rejects_bad_updates_and_layouts():
    assert "torch" not in SCHED.__dict__
    scheduler = SCHED.two_target_scheduler(TARGETS[:2], MEMBERS, CONTROLS, seed=52200)
    with pytest.raises(ValueError, match="nonnegative"):
        scheduler.batch(-1)
    with pytest.raises(ValueError, match="exactly 2 distinct"):
        SCHED.two_target_scheduler((TARGETS[0], TARGETS[0]), MEMBERS, CONTROLS, seed=1)
    with pytest.raises(ValueError, match="duplicates"):
        SCHED.two_target_scheduler(
            TARGETS[:2], {**MEMBERS, TARGETS[0]: (1, 1)}, CONTROLS, seed=1
        )
