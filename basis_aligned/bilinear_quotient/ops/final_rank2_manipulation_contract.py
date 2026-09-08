"""Gauge-invariant final-state removal, sufficiency, and paired construction swap."""

from __future__ import annotations


class FinalManipulationError(ValueError):
    pass


def projected_delta(off, on, basis):
    if off.shape != on.shape or off.ndim != 2 or basis.ndim != 2 or basis.shape[0] != off.shape[1]:
        raise FinalManipulationError("states or basis have incompatible shapes")
    return ((on.float() - off.float()) @ basis.float()) @ basis.float().T


def removal_and_sufficiency(off, on, basis):
    parallel = projected_delta(off, on, basis)
    return {"removed": on.float() - parallel, "sufficient": off.float() + parallel,
            "parallel": parallel}


def paired_payload_swap(torch, off, parallel, rows):
    if off.shape != parallel.shape or off.ndim != 2 or len(rows) != off.shape[0]:
        raise FinalManipulationError("row/state inventory mismatch")
    by_group = {}
    for index, row in enumerate(rows):
        by_group.setdefault(int(row["group_number"]), {})[row["transform_id"]] = index
    if any(set(items) != {"A1", "A2", "P", "C"} for items in by_group.values()):
        raise FinalManipulationError("each group must contain exactly A1/A2/P/C")
    result = off.float().clone()
    for items in by_group.values():
        result[items["A1"]] += parallel[items["A2"]]
        result[items["A2"]] += parallel[items["A1"]]
    return result


def paired_scaled_payload_swap(torch, off, parallel, rows, gains):
    if set(gains) != {"A1_from_A2", "A2_from_A1"}:
        raise FinalManipulationError("scaled swap requires both directed gains")
    if any(float(value) <= 0 for value in gains.values()):
        raise FinalManipulationError("swap gains must be positive")
    if off.shape != parallel.shape or off.ndim != 2 or len(rows) != off.shape[0]:
        raise FinalManipulationError("row/state inventory mismatch")
    by_group = {}
    for index, row in enumerate(rows):
        by_group.setdefault(int(row["group_number"]), {})[row["transform_id"]] = index
    if any(set(items) != {"A1", "A2", "P", "C"} for items in by_group.values()):
        raise FinalManipulationError("each group must contain exactly A1/A2/P/C")
    result = off.float().clone()
    for items in by_group.values():
        result[items["A1"]] += float(gains["A1_from_A2"]) * parallel[items["A2"]]
        result[items["A2"]] += float(gains["A2_from_A1"]) * parallel[items["A1"]]
    return result
