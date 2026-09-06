import pytest

import run_task14_bracket_fixed_program_stress_composition as composition


def test_price_and_assignment_are_fixed():
    assert composition.derive_price()["physical_model_forwards"] == 10
    assert composition.derive_price()["example_evaluations"] == 2720
    choices = ["a", "b", "c"]
    assert composition._choice("fixed", choices) == composition._choice("fixed", choices)
    assert composition.compile_plan()["price"]["fits"] == 0


def test_panel_score_exact_additivity_passes():
    rows=[]
    for _ in range(10):
        rows.append({"isolated_own":1.0,"own_under_stress":1.0,
                     "foreign_stress":.2,"interaction":0.0})
    scored=composition._panel_score(rows)
    assert scored["foreign_stress_to_own_norm_ratio"] == pytest.approx(.2)
    assert composition._passes(scored)
