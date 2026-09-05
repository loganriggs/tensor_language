from __future__ import annotations

from collections import Counter

import run_task14_cardinality_prototype_transfer_native_capability as capability


def test_gate_and_plan_are_candidate_scoped() -> None:
    gate = capability.build_gate()
    plan = capability.compile_plan()
    assert gate.capability_id == capability.authority.CAPABILITY_ID
    assert plan["causal_candidate_id"] == capability.authority.CAUSAL_CANDIDATE_ID
    assert len(gate.cells) == 12
    assert {cell.expected_count for cell in gate.cells} == {8}


def test_registered_cells_match_rows() -> None:
    counts = Counter(
        capability._cell(row, role)
        for row in capability.authority.build_rows()
        for role in capability.authority.ROLES
    )
    assert len(counts) == 12
    assert set(counts.values()) == {8}
