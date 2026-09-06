#!/usr/bin/env python3
"""Transparent baseline-conditioned counterfactual answer-margin actuator."""

from __future__ import annotations


class ActuatorError(ValueError):
    pass


def task14_effect(artifact: dict, recipient_number: str, donor_number: str, cardinality: int) -> float:
    if (recipient_number, donor_number) not in {
        ("singular", "plural"),
        ("plural", "singular"),
    } or not isinstance(cardinality, int) or not 0 <= cardinality <= 4:
        raise ActuatorError("invalid Task14 edit specification")
    key = f"{recipient_number}_to_{donor_number}.cardinality_{cardinality}"
    try:
        return float(artifact["effects"]["task14"][key])
    except KeyError as exc:
        raise ActuatorError(f"missing Task14 effect: {key}") from exc


def bracket_effect(artifact: dict, recipient_closer_id: int, donor_closer_id: int) -> float:
    closers = {1, 8, 60}
    if recipient_closer_id not in closers or donor_closer_id not in closers:
        raise ActuatorError("invalid bracket edit specification")
    if recipient_closer_id == donor_closer_id:
        return 0.0
    key = f"{recipient_closer_id}->{donor_closer_id}"
    try:
        return float(artifact["effects"]["bracket"][key])
    except KeyError as exc:
        raise ActuatorError(f"missing bracket effect: {key}") from exc


def actuate(native_unedited_answer_margin: float, predicted_effect: float) -> float:
    """Return the predicted internally-intervened answer margin."""
    return float(native_unedited_answer_margin) + float(predicted_effect)
