"""Deterministic balanced-row scheduling for rung 522.

The scheduler is deliberately independent of PyTorch, model code, data loading,
and CUDA.  It selects one row for every registered training role.  The caller is
responsible for expanding each selected recipient row through donor map
``update % 4`` and for averaging the eligible token responses within that row.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import operator
from typing import Literal, Mapping, Sequence


SCHEDULER_NAMESPACE = "a8-r522-balanced-rows-v1"
DONOR_MAP_COUNT = 4
RoleKind = Literal["member", "control"]


@dataclass(frozen=True)
class RolePlan:
    """One independently permuted member or control role."""

    name: str
    target: str
    kind: RoleKind
    replica: int
    permutation: tuple[int, ...]


@dataclass(frozen=True)
class ScheduledRole:
    """The row selected for one role in one optimizer update."""

    name: str
    target: str
    kind: RoleKind
    replica: int
    row_index: int


@dataclass(frozen=True)
class ScheduledBatch:
    """Every balanced role and the frozen donor-map index for one update."""

    update: int
    donor_map_index: int
    roles: tuple[ScheduledRole, ...]

    def rows_by_role(self) -> dict[str, int]:
        return {role.name: role.row_index for role in self.roles}


class BalancedRowScheduler:
    """Infinite deterministic schedule over a finite collection of role plans."""

    def __init__(self, mode: str, seed: int, roles: Sequence[RolePlan]) -> None:
        if mode not in {"two_target", "single_target_oracle", "all_three"}:
            raise ValueError(f"unknown balanced scheduler mode {mode!r}")
        self.mode = mode
        self.seed = operator.index(seed)
        self._roles = tuple(roles)
        expected_count = {
            "two_target": 4,
            "single_target_oracle": 4,
            "all_three": 6,
        }[mode]
        if len(self._roles) != expected_count:
            raise ValueError(f"{mode} requires exactly {expected_count} roles")
        names = [role.name for role in self._roles]
        if len(set(names)) != len(names):
            raise ValueError("role names must be unique")
        if any(not role.permutation for role in self._roles):
            raise ValueError("every member/control role must contain at least one eligible row")
        self.fingerprint = _scheduler_fingerprint(mode, self.seed, self._roles)

    @property
    def role_names(self) -> tuple[str, ...]:
        return tuple(role.name for role in self._roles)

    @property
    def roles(self) -> tuple[RolePlan, ...]:
        return self._roles

    def role_permutation(self, role_name: str) -> tuple[int, ...]:
        for role in self._roles:
            if role.name == role_name:
                return role.permutation
        raise KeyError(role_name)

    def batch(self, update: int) -> ScheduledBatch:
        """Select one row per role; no role repeats before its pool is exhausted."""
        update = operator.index(update)
        if update < 0:
            raise ValueError("update must be nonnegative")
        scheduled = tuple(
            ScheduledRole(
                name=role.name,
                target=role.target,
                kind=role.kind,
                replica=role.replica,
                row_index=role.permutation[update % len(role.permutation)],
            )
            for role in self._roles
        )
        return ScheduledBatch(
            update=update,
            donor_map_index=update % DONOR_MAP_COUNT,
            roles=scheduled,
        )


def _rows(
    pools: Mapping[str, Sequence[int]], target: str, kind: RoleKind
) -> tuple[int, ...]:
    if target not in pools:
        raise ValueError(f"missing {kind} row pool for target {target!r}")
    try:
        rows = tuple(operator.index(row) for row in pools[target])
    except TypeError as error:
        raise ValueError(f"{kind} rows for target {target!r} must be integer indices") from error
    if not rows:
        raise ValueError(f"{kind} role for target {target!r} has no eligible row")
    if len(set(rows)) != len(rows):
        raise ValueError(f"{kind} rows for target {target!r} contain duplicates")
    return rows


def _targets(values: Sequence[str], expected: int) -> tuple[str, ...]:
    targets = tuple(values)
    if len(targets) != expected or len(set(targets)) != expected:
        raise ValueError(f"expected exactly {expected} distinct target names")
    if any(not isinstance(target, str) or not target for target in targets):
        raise ValueError("target names must be nonempty strings")
    return targets


def _digest(seed: int, role_name: str, row: int) -> bytes:
    payload = f"{SCHEDULER_NAMESPACE}:{seed}:{role_name}:{row}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def _permutation(rows: tuple[int, ...], seed: int, role_name: str) -> tuple[int, ...]:
    return tuple(sorted(rows, key=lambda row: (_digest(seed, role_name, row), row)))


def _role(
    target: str,
    kind: RoleKind,
    replica: int,
    rows: tuple[int, ...],
    seed: int,
) -> RolePlan:
    name = f"{target}:{kind}:{replica}"
    return RolePlan(
        name=name,
        target=target,
        kind=kind,
        replica=replica,
        permutation=_permutation(rows, seed, name),
    )


def _scheduler_fingerprint(mode: str, seed: int, roles: Sequence[RolePlan]) -> str:
    payload = {
        "namespace": SCHEDULER_NAMESPACE,
        "mode": mode,
        "seed": seed,
        "donor_map_rule": "update_mod_4",
        "roles": [
            {
                "name": role.name,
                "target": role.target,
                "kind": role.kind,
                "replica": role.replica,
                "permutation": list(role.permutation),
            }
            for role in roles
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def two_target_scheduler(
    targets: Sequence[str],
    member_rows: Mapping[str, Sequence[int]],
    control_rows: Mapping[str, Sequence[int]],
    *,
    seed: int,
) -> BalancedRowScheduler:
    """Four roles: one member and one control row for each of two targets."""
    chosen = _targets(targets, 2)
    seed = operator.index(seed)
    roles = []
    for target in chosen:
        roles.append(_role(target, "member", 0, _rows(member_rows, target, "member"), seed))
        roles.append(_role(target, "control", 0, _rows(control_rows, target, "control"), seed))
    return BalancedRowScheduler("two_target", seed, roles)


def single_target_oracle_scheduler(
    target: str,
    member_rows: Mapping[str, Sequence[int]],
    control_rows: Mapping[str, Sequence[int]],
    *,
    seed: int,
) -> BalancedRowScheduler:
    """Four roles: two independently permuted member and two control rows."""
    chosen = _targets((target,), 1)[0]
    seed = operator.index(seed)
    member_pool = _rows(member_rows, chosen, "member")
    control_pool = _rows(control_rows, chosen, "control")
    if len(member_pool) < 2 or len(control_pool) < 2:
        raise ValueError("oracle member and control pools must each contain at least two rows")

    def distinct_pair(kind: RoleKind, pool: tuple[int, ...]) -> list[RolePlan]:
        first = _role(chosen, kind, 0, pool, seed)
        # Use the same seed-defined ordering at a one-position phase offset.
        # This guarantees two distinct rows in every oracle update while each
        # role still exhausts the complete pool before cycling.
        rotated = first.permutation[1:] + first.permutation[:1]
        second = RolePlan(
            name=f"{chosen}:{kind}:1",
            target=chosen,
            kind=kind,
            replica=1,
            permutation=rotated,
        )
        return [first, second]

    roles = distinct_pair("member", member_pool) + distinct_pair("control", control_pool)
    return BalancedRowScheduler("single_target_oracle", seed, roles)


def all_three_scheduler(
    targets: Sequence[str],
    member_rows: Mapping[str, Sequence[int]],
    control_rows: Mapping[str, Sequence[int]],
    *,
    seed: int,
) -> BalancedRowScheduler:
    """Six roles: one member and one control row for every fitted target."""
    chosen = _targets(targets, 3)
    seed = operator.index(seed)
    roles = []
    for target in chosen:
        roles.append(_role(target, "member", 0, _rows(member_rows, target, "member"), seed))
        roles.append(_role(target, "control", 0, _rows(control_rows, target, "control"), seed))
    return BalancedRowScheduler("all_three", seed, roles)


__all__ = [
    "BalancedRowScheduler",
    "DONOR_MAP_COUNT",
    "RolePlan",
    "SCHEDULER_NAMESPACE",
    "ScheduledBatch",
    "ScheduledRole",
    "all_three_scheduler",
    "single_target_oracle_scheduler",
    "two_target_scheduler",
]
