#!/usr/bin/env python3
"""Pure specification-to-vector dispatch for the compiled Task14/bracket package."""
from __future__ import annotations

from typing import Mapping


NUMBERS = ("singular", "plural")
CLOSERS = (1, 8, 60)
WIDTH = 1152


class DispatchError(ValueError):
    pass


def _vectors(package: Mapping[str, object], program: str) -> Mapping[str, list[float]]:
    try:
        vectors = package["programs"][program]["vectors"]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise DispatchError("compiled package schema is invalid") from error
    if not isinstance(vectors, Mapping):
        raise DispatchError("compiled vectors are invalid")
    return vectors  # type: ignore[return-value]


def dispatch_task14(
    package: Mapping[str, object], *, recipient_number: str, donor_number: str,
    cardinality: int,
) -> list[float]:
    if recipient_number not in NUMBERS or donor_number not in NUMBERS:
        raise DispatchError("number must be singular or plural")
    if recipient_number == donor_number:
        raise DispatchError("Task14 requires an opposite-number edit")
    if type(cardinality) is not int or cardinality not in range(5):
        raise DispatchError("cardinality must be an integer from 0 through 4")
    key = f"{recipient_number}_to_{donor_number}.cardinality_{cardinality}"
    vector = _vectors(package, "task14").get(key)
    if not isinstance(vector, list) or len(vector) != WIDTH:
        raise DispatchError(f"missing or malformed Task14 vector {key}")
    return vector


def dispatch_bracket(
    package: Mapping[str, object], *, recipient_closer_id: int, donor_closer_id: int,
) -> list[float]:
    if recipient_closer_id not in CLOSERS or donor_closer_id not in CLOSERS:
        raise DispatchError("closer IDs must belong to the licensed vocabulary")
    if recipient_closer_id == donor_closer_id:
        return [0.0] * WIDTH
    key = f"{recipient_closer_id}->{donor_closer_id}"
    vector = _vectors(package, "bracket").get(key)
    if not isinstance(vector, list) or len(vector) != WIDTH:
        raise DispatchError(f"missing or malformed bracket vector {key}")
    return vector

