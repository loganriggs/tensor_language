#!/usr/bin/env python3

import run_task14_head11_3_subject_score_context_gate_factorial as v1
import run_task14_head11_3_subject_score_context_gate_factorial_v2 as runner


def test_v2_is_create_only_and_keeps_three_forwards():
    plan = runner.compile_plan()
    assert runner.OUT != v1.OUT
    assert plan["price"] == {"model_forwards": 3, "example_evaluations": 832,
                             "backwards": 0, "parameter_updates": 0}
    assert plan["candidate_id"] == (
        "subject_verb.number_agreement."
        "head11_3_subject_score_context_gate_factorial_capability_repair_v2"
    )
    assert "minimum_weak_cell_relative_score_contrast_fraction" not in plan["bars"]
    assert plan["closed_splits"] == ["OOD"]


def test_patch_adds_zero_and_twice_native_score_arms():
    import torch
    base = {"p": torch.tensor([[2., 3.]]), "u": torch.tensor([[[5.], [7.]]]),
            "head": torch.tensor([[11.]])}
    context = {"p": torch.tensor([[13., 17.]]), "u": torch.zeros(1, 2, 1),
               "head": torch.tensor([[19.]])}
    opposite = {"p": torch.zeros(1, 2), "u": torch.tensor([[[23.], [29.]]]),
                "head": torch.tensor([[31.]])}
    batch = runner._patch_batch(torch.zeros(1, 2, dtype=torch.long), torch.tensor([1]),
                                base, context, opposite, torch)
    assert batch["replacement_terms"][:, 0].tolist() == [21., 119., 87., 493., 0., 174., 21.]
    assert batch["replacement_heads"][:, 0].tolist() == [11.] * 6 + [31.]


def _capability():
    return {cell: {name: 1.0 for name in ("base", "score_context", "opposite_value")}
            for cell in (v1.WEAK_CELL, "b", "c", "d")}


def _contrast(relative_fraction=0.0, epsilon_fraction=1.0):
    return {cell: {"mean_absolute_difference": .01, "maximum_absolute_difference": .02,
                   "fraction_above_epsilon": epsilon_fraction,
                   "mean_absolute_relative_difference": .01,
                   "fraction_above_relative_threshold": relative_fraction}
            for cell in _capability()}


def _evidence(span=.3, span_direction=1.0, natural_interaction=0.0):
    output = []
    positive = int(16 * span_direction)
    for cell in _capability():
        native_value = .18 if cell == v1.WEAK_CELL else .30
        for index in range(16):
            signed_span = span if index < positive else -.01
            values = {
                "native_score_native_value": (0.0, 0.0),
                "opposite_syntax_score_native_value": (0.0, 0.0),
                "native_score_opposite_value": (native_value, native_value),
                "opposite_syntax_score_opposite_value": (
                    native_value + natural_interaction, native_value + natural_interaction,
                ),
                "zero_score_opposite_value": (0.0, 0.0),
                "twice_native_score_opposite_value": (signed_span, signed_span),
                "complete_head": (1.0, 1.0),
            }
            for condition, (margin, ce_gain) in values.items():
                output.append({"cell_id": cell, "condition": condition,
                               "donor_margin": 2 * margin, "margin_delta": 2 * margin,
                               "donor_ce": 2.0 - ce_gain, "donor_ce_gain": ce_gain})
    return output


def test_task_span_replaces_arbitrary_relative_score_gate():
    bars = runner.compile_plan()["bars"]
    scored = runner.score(_evidence(), _capability(), 0, 0, 0, 0,
                          _contrast(relative_fraction=0.0), 0, bars)
    assert scored["predictions"]["pred_a_instrument_live"]
    assert scored["predictions"]["pred_b_causal_score_span_live"]


def test_task_span_requires_recovery_direction_ce_and_nonzero_natural_score():
    bars = runner.compile_plan()["bars"]
    assert not runner.score(_evidence(span=.05), _capability(), 0, 0, 0, 0,
                            _contrast(), 0, bars)["predictions"]["pred_a_instrument_live"]
    assert not runner.score(_evidence(span=.3, span_direction=.5), _capability(), 0, 0, 0, 0,
                            _contrast(), 0, bars)["predictions"]["pred_a_instrument_live"]
    harmful_ce = _evidence()
    for row in harmful_ce:
        if row["cell_id"] == v1.WEAK_CELL and \
                row["condition"] == "twice_native_score_opposite_value":
            row["donor_ce"] = 3.0
            row["donor_ce_gain"] = -1.0
    assert not runner.score(harmful_ce, _capability(), 0, 0, 0, 0,
                            _contrast(), 0, bars)["predictions"]["pred_a_instrument_live"]
    dead = _contrast()
    dead[v1.WEAK_CELL]["fraction_above_epsilon"] = .9375
    assert not runner.score(_evidence(), _capability(), 0, 0, 0, 0,
                            dead, 0, bars)["predictions"]["pred_a_instrument_live"]
