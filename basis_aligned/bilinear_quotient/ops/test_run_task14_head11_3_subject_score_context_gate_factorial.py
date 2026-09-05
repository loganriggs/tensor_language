#!/usr/bin/env python3

import run_task14_head11_3_subject_score_context_gate_factorial as runner


def test_score_context_authority_preserves_subject_and_number_but_changes_syntax():
    rows = runner.build_rows()
    assert len(rows) == 64
    assert all(row["score_context_ids"][1] == row["base_ids"][1] for row in rows)
    assert all(row["score_context_ids"][:2] == row["base_ids"][:2] for row in rows)
    assert all(row["score_context_ids"][-1] == row["base_ids"][-1] for row in rows)
    assert all(row["score_context_subject_number"] == row["base_subject_number"] for row in rows)
    assert all(row["score_context_ids"] != row["base_ids"] for row in rows)
    plan = runner.compile_plan()
    assert plan["price"] == {"model_forwards": 3, "example_evaluations": 704,
                             "backwards": 0, "parameter_updates": 0}
    assert plan["closed_splits"] == ["OOD"]


def test_patch_batch_is_the_exact_two_by_two_plus_complete_head():
    import torch
    base = {"p": torch.tensor([[2., 3.]]), "u": torch.tensor([[[5.], [7.]]]),
            "head": torch.tensor([[11.]])}
    context = {"p": torch.tensor([[13., 17.]]), "u": torch.zeros(1, 2, 1),
               "head": torch.tensor([[19.]])}
    opposite = {"p": torch.zeros(1, 2), "u": torch.tensor([[[23.], [29.]]]),
                "head": torch.tensor([[31.]])}
    batch = runner._patch_batch(torch.zeros(1, 2, dtype=torch.long), torch.tensor([1]),
                                base, context, opposite, torch)
    assert batch["replacement_terms"][:, 0].tolist() == [21., 119., 87., 493., 21.]
    assert batch["replacement_heads"][:, 0].tolist() == [11., 11., 11., 11., 31.]


def _capability():
    return {cell: {name: 1.0 for name in ("base", "score_context", "opposite_value")}
            for cell in (runner.WEAK_CELL, "b", "c", "d")}


def _score_contrast():
    return {cell: {"mean_absolute_difference": .1, "maximum_absolute_difference": .2,
                   "fraction_above_epsilon": 1.0,
                   "mean_absolute_relative_difference": .5,
                   "fraction_above_relative_threshold": 1.0} for cell in _capability()}


def _evidence(weak_native=.18, weak_alternate=.35, alternate_native=0.0):
    output = []
    for cell in _capability():
        native_opposite = weak_native if cell == runner.WEAK_CELL else .30
        alternate_opposite = weak_alternate if cell == runner.WEAK_CELL else .32
        for _ in range(16):
            values = {
                "native_score_native_value": 0.0,
                "opposite_syntax_score_native_value": alternate_native,
                "native_score_opposite_value": native_opposite,
                "opposite_syntax_score_opposite_value": alternate_opposite,
                "complete_head": 1.0,
            }
            output.extend({"cell_id": cell, "condition": condition,
                           "margin_delta": 2 * value, "donor_ce_gain": value}
                          for condition, value in values.items())
    return output


def test_score_distinguishes_context_gate_from_independence():
    bars = runner.compile_plan()["bars"]
    rescue = runner.score(_evidence(), _capability(), 0, 0, 0, 0,
                          _score_contrast(), 0, bars)["predictions"]
    independent = runner.score(_evidence(.18, .20), _capability(), 0, 0, 0, 0,
                               _score_contrast(), 0, bars)["predictions"]
    assert rescue["pred_b_opposite_syntax_score_rescues_weak_cell"]
    assert not rescue["pred_c_weak_pp_plural_opposite_syntax_score_no_interaction"]
    assert independent["pred_c_weak_pp_plural_opposite_syntax_score_no_interaction"]

    baseline_shift_only = runner.score(
        _evidence(.18, .35, alternate_native=.17), _capability(), 0, 0, 0, 0,
        _score_contrast(), 0, bars,
    )["predictions"]
    assert not baseline_shift_only["pred_b_opposite_syntax_score_rescues_weak_cell"]
    assert baseline_shift_only["pred_c_weak_pp_plural_opposite_syntax_score_no_interaction"]


def test_parent_reproduction_and_native_corner_are_instrument_gates():
    bars = runner.compile_plan()["bars"]
    assert not runner.score(_evidence(), _capability(), 0, 1e-3, 0, 0,
                            _score_contrast(), 0, bars)["predictions"]["pred_a_instrument_live"]
    assert not runner.score(_evidence(), _capability(), 0, 0, 0, 1e-3,
                            _score_contrast(), 0, bars)["predictions"]["pred_a_instrument_live"]
    dead = _score_contrast()
    dead[runner.WEAK_CELL]["fraction_above_relative_threshold"] = 0.0
    assert not runner.score(_evidence(), _capability(), 0, 0, 0, 0, dead, 0,
                            bars)["predictions"]["pred_a_instrument_live"]
