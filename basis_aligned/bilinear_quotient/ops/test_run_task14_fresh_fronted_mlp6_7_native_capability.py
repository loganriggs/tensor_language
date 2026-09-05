from collections import Counter

import run_task14_fresh_fronted_mlp6_7_native_capability as run


def test_gate_has_twelve_balanced_cells():
    gate = run.build_gate()
    assert len(gate.cells) == 12
    assert {cell.expected_count for cell in gate.cells} == {8}
    assert {cell.minimum_accuracy for cell in gate.cells} == {.875}


def test_roles_map_to_distinct_authority_endpoints():
    assert set(run.ROLES) == set(run.ROLE_SOURCE)
    assert set(run.ROLE_SOURCE.values()) == {"base", "opposite", "same"}


def test_plan_is_native_only_and_zero_intervention():
    plan = run.compile_plan()
    assert plan["native_only"] is True
    assert plan["endpoint_evaluations"] == 96
    assert plan["price"]["causal_interventions"] == 0


def test_authority_balance_matches_gate():
    counts = Counter(run._cell_id(row, role) for row in run.authority.build_rows()
                     for role in run.ROLES)
    assert len(counts) == 12 and set(counts.values()) == {8}
