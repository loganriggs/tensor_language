#!/usr/bin/env python3
"""Typed composition semantics for the baseline-conditioned margin actuator."""

from __future__ import annotations

from dataclasses import dataclass, replace

import task14_bracket_margin_actuator as actuator


@dataclass(frozen=True)
class SlotState:
    slot_id: str
    behavior: str
    native_baseline_margin: float
    edit: tuple | None = None


def effect(artifact: dict, state: SlotState) -> float:
    if state.behavior not in {"task14", "bracket"}:
        raise actuator.ActuatorError("unknown behavior")
    if state.edit is None:
        return 0.0
    if state.behavior == "task14" and len(state.edit) == 3:
        return actuator.task14_effect(artifact, *state.edit)
    if state.behavior == "bracket" and len(state.edit) == 2:
        return actuator.bracket_effect(artifact, *state.edit)
    raise actuator.ActuatorError("cross-typed or malformed edit")


def evaluate(artifact: dict, state: SlotState) -> float:
    return actuator.actuate(state.native_baseline_margin, effect(artifact, state))


def overwrite(artifact: dict, state: SlotState, edit: tuple | None) -> SlotState:
    candidate = replace(state, edit=edit)
    effect(artifact, candidate)  # validate without changing the immutable baseline
    return candidate


def update_slot(artifact: dict, states: dict[str, SlotState], slot_id: str, edit: tuple | None) -> dict[str, SlotState]:
    if slot_id not in states:
        raise actuator.ActuatorError("unknown slot")
    updated = dict(states)
    updated[slot_id] = overwrite(artifact, states[slot_id], edit)
    return updated
