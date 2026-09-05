"""Focused CPU tests for the A1 tense-template capability selector."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import run_narrative_tense_a1_direct_template_capability_select_holdout as run


def test_plan_price_and_split_boundary_are_frozen():
    plan = run.compile_plan()
    assert plan["price"] == {"model_forwards": 2, "example_evaluations": 160,
                             "backwards": 0, "parameter_updates": 0}
    assert plan["selection_groups"] == list(range(16))
    assert plan["construction_holdout_groups"] == list(range(16, 32))
    assert plan["circuit_interventions"] == 0
    assert run.MINIMUM_HOLDOUT_ACCURACY_EACH_TENSE == .875


def test_templates_and_direction_side_cells_are_exact():
    expected_lengths = {"remained": 12, "served_one_purpose": 13,
                        "had_one_purpose": 13}
    fit = run.build_pairs(run.TEMPLATE_ORDER, 0, 16)
    assert len(fit) == 48
    for pair in fit:
        base, donor = pair["endpoints"]
        assert len(base["ids"]) == len(donor["ids"]) == expected_lengths[pair["template_id"]]
        assert tuple(i for i, values in enumerate(zip(base["ids"], donor["ids"]))
                     if values[0] != values[1]) == (0, 3)
    holdout = run.build_pairs(("remained",), 16, 32, ("A1", "P"))
    assert len(holdout) == 32
    assert {pair["family"] for pair in holdout} == {"A1", "P"}
    assert {pair["group_number"] for pair in holdout} == set(range(16, 32))
    for pair in holdout:
        base, donor = pair["endpoints"]
        assert len(base["ids"]) == len(donor["ids"])
        assert base["ids"][-1] == donor["ids"][-1]
        changed = tuple(i for i, values in enumerate(zip(base["ids"], donor["ids"]))
                        if values[0] != values[1])
        assert changed == ((0, 3) if pair["family"] == "A1" else (2,))


def _selection_summary(worst_accuracy, worst_margin):
    summary = {}
    for template, accuracy, margin in zip(run.TEMPLATE_ORDER, worst_accuracy, worst_margin):
        cells = {}
        for direction in ("past_to_present", "present_to_past"):
            for side in ("base", "donor"):
                cells[f"A1/{direction}/{side}"] = {
                    "accuracy": accuracy, "mean_signed_margin": margin,
                    "mean_full_vocab_CE": 1.0, "count": 8,
                }
        summary[template] = {"direction_side_cells": cells}
    return summary


def test_selector_uses_global_eligibility_then_worst_cell_margin_then_order():
    # Template 0 has a larger margin but is ineligible; templates 1 and 2 tie on
    # accuracy, and template 2 wins on the preregistered worst-cell margin.
    summary = _selection_summary((.75, .875, .875), (9.0, .2, .3))
    assert run.select_template(summary) == "had_one_purpose"
    # Exact eligible ties resolve only by the frozen order.
    summary = _selection_summary((.875, .875, .875), (.3, .3, .3))
    assert run.select_template(summary) == "remained"


def test_no_eligible_fit_does_not_build_or_evaluate_holdout(monkeypatch):
    summary = _selection_summary((.75, .75, .75), (3.0, 2.0, 1.0))
    build_calls = []
    evaluation_calls = []

    def fake_build(template_ids, start, stop, families=("A1",)):
        build_calls.append((tuple(template_ids), start, stop, tuple(families)))
        return ["fit"]

    def fake_evaluate(model, pairs, torch, F):
        evaluation_calls.append(tuple(pairs))
        return ["evidence"]

    monkeypatch.setattr(run, "build_pairs", fake_build)
    monkeypatch.setattr(run, "evaluate_pairs", fake_evaluate)
    monkeypatch.setattr(run, "_summarize", lambda evidence: summary)
    result = run.evaluate_two_stage(None, None, None)
    assert not result[3]
    assert result[4:] == ([], None)
    assert build_calls == [(run.TEMPLATE_ORDER, 0, 16, ("A1",))]
    assert evaluation_calls == [("fit",)]


def test_no_model_environment_stops_at_plan():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               check=True, capture_output=True, text=True)
    assert '"model_loaded": false' in completed.stdout
    assert '"circuit_interventions": 0' in completed.stdout
