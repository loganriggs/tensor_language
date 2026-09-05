#!/usr/bin/env python3

import run_task14_head11_3_subject_payload_number_specificity as runner


def test_both_conditions_keep_recipient_score_and_replace_only_payload():
    import torch
    recipient = {"p": torch.tensor([[2., 3.], [5., 7.]]),
                 "u": torch.zeros(2, 2, 1)}
    same = {"p": torch.tensor([[101., 103.], [107., 109.]]),
            "u": torch.tensor([[[11.], [13.]], [[17.], [19.]]])}
    opposite = {"p": torch.tensor([[127., 131.], [137., 139.]]),
                "u": torch.tensor([[[23.], [29.]], [[31.], [37.]]])}
    terms = runner._payload_terms(recipient, same, opposite, torch.tensor([0, 1]), torch)
    assert terms[:, 0].tolist() == [22., 133., 46., 259.]


def test_frozen_triples_have_two_capable_cross_noun_cross_syntax_controls():
    rows = runner.build_triples()
    assert len(rows) == 64
    assert all(row["target_group_id"] != row["donor_group_id"] for row in rows)
    assert all(len({row["base_ids"][1], row["same_ids"][1],
                       row["opposite_ids"][1]}) == 3 for row in rows)
    assert all(row["same_ids"][:1] + row["same_ids"][2:]
               == row["opposite_ids"][:1] + row["opposite_ids"][2:] for row in rows)
    assert all(row["target_family"] != row["donor_family"] for row in rows)
    assert all(row["base_subject_number"] == row["same_subject_number"] for row in rows)
    assert all(row["base_subject_number"] != row["opposite_subject_number"] for row in rows)
    assert all(row["base_attractor_plural"] == row["same_attractor_plural"]
               == row["opposite_attractor_plural"] for row in rows)
    plan = runner.compile_plan()
    assert plan["price"]["model_forwards"] == 3
    assert plan["closed_splits"] == ["TEST", "OOD"]


def _evidence(same_margin, same_ce, opposite_margin=2.0, opposite_ce=1.0):
    return [
        {"condition": condition,
         "cell_id": "cell",
         "task_margin_delta": same_margin if condition == "same_number_payload" else opposite_margin,
         "directed_margin_delta": same_margin if condition == "same_number_payload" else opposite_margin,
         "answer_ce_gain": same_ce if condition == "same_number_payload" else opposite_ce}
        for condition in runner.CONDITIONS for _ in range(32)
    ]


def test_score_requires_live_control_and_distinguishes_specificity_from_leakage():
    specific = runner.score(_evidence(.1, .1), 0, 0, 0, 1)["predictions"]
    leaky = runner.score(_evidence(.8, .4), 0, 0, 0, 1)["predictions"]
    dead = runner.score(_evidence(.01, .01, .1, .1), 0, 0, 0, 1)["predictions"]
    assert specific["pred_a_opposite_number_control_live"]
    assert specific["pred_b_number_specific_payload"]
    assert leaky["pred_c_noun_or_syntax_sensitive_payload"]
    assert not dead["pred_a_opposite_number_control_live"]
    assert not dead["pred_b_number_specific_payload"]


def test_score_rejects_failure_to_reproduce_parent_opposite_condition():
    predictions = runner.score(_evidence(.1, .1), 0, 0, 1e-3, 1)["predictions"]
    assert not predictions["pred_a_opposite_number_control_live"]


def test_score_does_not_hide_one_leaky_cell_in_a_pooled_average():
    evidence = _evidence(.01, .01)
    for row in evidence:
        row["cell_id"] = "clean"
    evidence.extend({
        "condition": condition, "cell_id": "leaky",
        "task_margin_delta": .8 if condition == "same_number_payload" else 2.0,
        "directed_margin_delta": .8 if condition == "same_number_payload" else 2.0,
        "answer_ce_gain": .4 if condition == "same_number_payload" else 1.0,
    } for condition in runner.CONDITIONS)
    predictions = runner.score(evidence, 0, 0, 0, 1)["predictions"]
    assert predictions["pred_a_opposite_number_control_live"]
    assert not predictions["pred_b_number_specific_payload"]
