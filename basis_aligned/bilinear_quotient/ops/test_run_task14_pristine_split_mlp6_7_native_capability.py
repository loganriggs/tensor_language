from collections import Counter

import circuit_fast_screen_candidate_task14_pristine_split_mlp6_7_absolute_composition as authority
import run_task14_pristine_split_mlp6_7_native_capability as run


def test_authority_is_pristine_balanced_and_frozen():
    rows = authority.build_rows()
    assert len(rows) == 40 and authority.validate_rows(rows) == authority.EXPECTED_AUTHORITY_SHA256
    assert Counter(row["phase"] for row in rows) == {"FIT": 32, "HOLDOUT": 8}
    assert {row["template_id"] for row in rows if row["phase"] == "HOLDOUT"} == {"past_outside"}


def test_gate_has_eighteen_registered_cells():
    gate = run.build_gate()
    assert len(gate.cells) == 18
    assert sorted(cell.expected_count for cell in gate.cells) == [4]*6 + [8]*12
    assert {cell.minimum_accuracy for cell in gate.cells} == {.75}


def test_plan_is_native_only_and_zero_intervention():
    plan = run.compile_plan()
    assert plan["native_only"] is True and plan["endpoint_evaluations"] == 120
    assert plan["price"]["causal_interventions"] == 0


def test_endpoints_change_only_at_subject():
    for row in authority.build_rows():
        recipient = row["endpoints"]["recipient"]["ids"]
        for role in ("opposite_same_lemma", "same_number_different_lemma"):
            assert [i for i, pair in enumerate(zip(recipient, row["endpoints"][role]["ids"]))
                    if pair[0] != pair[1]] == [8]
