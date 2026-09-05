#!/usr/bin/env python3

import run_task14_head11_3_ood_same_syntax_source_value_atlas as runner


def test_pairing_lengths_subject_positions_and_price():
    rows = runner.build_rows()
    assert len(rows) == 64
    for row in rows:
        position = 8 if row["target_family"] == "A1" else 1
        length = 9 if row["target_family"] == "A1" else 11
        assert row["subject_position"] == position
        assert len(row["base_ids"]) == len(row["donor_ids"]) == length
        assert [index for index, pair in enumerate(zip(row["base_ids"], row["donor_ids"]))
                if pair[0] != pair[1]] == [position]
    plan = runner.compile_plan()
    assert plan["price"] == {"model_forwards": 3, "example_evaluations": 1088,
                             "backwards": 0, "parameter_updates": 0}
    assert plan["scope"] == "OOD_TEXT_REUSE_NEW_INTERVENTION"


def test_shared_patch_compiler_excludes_padding():
    import torch
    rows = [{"base_ids": list(range(9))}, {"base_ids": list(range(11))}]
    tokens = torch.zeros(2, 11, dtype=torch.long)
    finals = torch.tensor([8, 10])
    base = {"p": torch.ones(2, 11), "u": torch.ones(2, 11, 1),
            "head": torch.ones(2, 1)}
    donor = {"p": torch.ones(2, 11), "u": 2 * torch.ones(2, 11, 1),
             "head": 3 * torch.ones(2, 1)}
    patch = runner.test_atlas._compile_patch_batch(tokens, finals, base, donor, rows, torch)
    positions = {0: [], 1: []}
    for row_index, condition, position in patch["specs"]:
        if condition == "single_source_value":
            positions[row_index].append(position)
    assert positions[0] == list(range(9))
    assert positions[1] == list(range(11))


def _evidence(relative_signal=True, joint_recovery=.9, fronted_subject=.9):
    output = []
    cells = (("fronted_singular_to_fronted_plural", "A1"),
             ("fronted_plural_to_fronted_singular", "A1"),
             ("two_attractor_relative_singular_to_two_attractor_relative_plural", "A2"),
             ("two_attractor_relative_plural_to_two_attractor_relative_singular", "A2"))
    for cell, family in cells:
        length = 9 if family == "A1" else 11
        subject = 8 if family == "A1" else 1
        for position in range(length):
            recovery = fronted_subject if position == subject and family == "A1" else 0.0
            if family == "A2" and position == 4 and relative_signal:
                recovery = .2
            for _ in range(16):
                output.append({"atlas_cell_id": cell, "target_family": family,
                               "condition": "single_source_value", "source_position": position,
                               "semantic_role": "subject" if position == subject else "other",
                               "margin_delta": 2 * recovery, "donor_ce_gain": recovery})
        for condition, recovery in (("joint_all_values", joint_recovery),
                                    ("complete_head", 1.0)):
            for _ in range(16):
                output.append({"atlas_cell_id": cell, "target_family": family,
                               "condition": condition, "source_position": None,
                               "margin_delta": 2 * recovery, "donor_ce_gain": recovery})
    return output


def _capability():
    return {cell: {"base": 1.0, "donor": 1.0} for cell in (
        "fronted_singular_to_fronted_plural", "fronted_plural_to_fronted_singular",
        "two_attractor_relative_singular_to_two_attractor_relative_plural",
        "two_attractor_relative_plural_to_two_attractor_relative_singular")}


def test_score_requires_all_three_registered_mechanism_predictions():
    bars = runner.compile_plan()["bars"]
    passed = runner.score(_evidence(), _capability(), 0, 0, 0, 0, 0, bars)["predictions"]
    assert all(passed.values())
    no_relay = runner.score(_evidence(relative_signal=False), _capability(),
                            0, 0, 0, 0, 0, bars)["predictions"]
    assert no_relay["pred_a_instrument_live"]
    assert not no_relay["pred_d_two_attractor_relative_later_relay"]


def test_fronted_causal_mask_and_subject_joint_exactness_are_instrument_gates():
    bars = runner.compile_plan()["bars"]
    assert not runner.score(_evidence(), _capability(), 0, 0, 0, 1e-3, 0,
                            bars)["predictions"]["pred_a_instrument_live"]
    assert not runner.score(_evidence(), _capability(), 0, 0, 0, 0, 1e-3,
                            bars)["predictions"]["pred_a_instrument_live"]
