#!/usr/bin/env python3

import run_task14_head11_3_same_syntax_source_value_atlas as runner


def test_rows_are_same_length_same_syntax_and_differ_only_at_subject():
    rows = runner.build_rows()
    assert len(rows) == 64
    assert {row["atlas_cell_id"] for row in rows} == {
        "pp_singular_to_pp_plural", "pp_plural_to_pp_singular",
        "relative_singular_to_relative_plural", "relative_plural_to_relative_singular",
    }
    assert all(len(row["base_ids"]) == len(row["donor_ids"]) for row in rows)
    assert all(row["base_ids"][:1] + row["base_ids"][2:]
               == row["donor_ids"][:1] + row["donor_ids"][2:] for row in rows)
    assert all(row["base_ids"][1] != row["donor_ids"][1] for row in rows)
    plan = runner.compile_plan()
    assert plan["price"] == {"model_forwards": 3, "example_evaluations": 928,
                             "backwards": 0, "parameter_updates": 0}
    assert plan["closed_claims"] == ["OOD", "selectivity", "completeness"]


def test_patch_batch_covers_every_position_and_exact_value_products():
    import torch
    rows = [{"base_ids": [1, 2, 3]}, {"base_ids": [4, 5]}]
    tokens = torch.tensor([[1, 2, 3], [4, 5, 0]])
    finals = torch.tensor([2, 1])
    base = {"p": torch.tensor([[2., 3., 5.], [7., 11., 0.]]),
            "u": torch.ones(2, 3, 1), "head": torch.tensor([[10.], [20.]])}
    donor = {"p": torch.zeros(2, 3),
             "u": torch.tensor([[[11.], [13.], [17.]], [[19.], [23.], [0.]]]),
             "head": torch.tensor([[30.], [40.]])}
    patch = runner._compile_patch_batch(tokens, finals, base, donor, rows, torch)
    singles = [(spec, float(term)) for spec, term in
               zip(patch["specs"], patch["replacement_terms"])
               if spec[1] == "single_source_value"]
    assert singles == [((0, "single_source_value", 0), 22.),
                       ((0, "single_source_value", 1), 39.),
                       ((0, "single_source_value", 2), 85.),
                       ((1, "single_source_value", 0), 133.),
                       ((1, "single_source_value", 1), 253.)]
    joint_indices = [index for index, spec in enumerate(patch["specs"])
                     if spec[1] == "joint_all_values"]
    assert patch["replacement_heads"][joint_indices, 0].tolist() == [146., 386.]


def _fake_evidence(non_subject_margin):
    evidence = []
    for cell in ("a", "b", "c", "d"):
        for position, margin in ((0, non_subject_margin), (1, 1.0)):
            for _ in range(16):
                evidence.append({"atlas_cell_id": cell, "condition": "single_source_value",
                                 "source_position": position, "semantic_role": "subject" if position == 1 else "determiner",
                                 "recipient_token_text": " x", "donor_token_text": " x",
                                 "margin_delta": margin, "donor_ce_gain": margin})
        for condition, margin in (("complete_head", 2.), ("joint_all_values", 1.)):
            for _ in range(16):
                evidence.append({"atlas_cell_id": cell, "condition": condition,
                                 "source_position": None, "margin_delta": margin,
                                 "donor_ce_gain": margin})
    return evidence


def test_score_distinguishes_non_subject_signal_and_subject_only():
    capability = {cell: {"base": 1., "donor": 1.} for cell in ("a", "b", "c", "d")}
    bars = runner.compile_plan()["bars"]
    signal = runner.score(_fake_evidence(.3), capability, 0, 0, 0, 0, 0, bars)["predictions"]
    subject = runner.score(_fake_evidence(.05), capability, 0, 0, 0, 0, 0, bars)["predictions"]
    assert signal["pred_a_instrument_live"] and signal["pred_b_non_subject_source_signal"]
    assert subject["pred_a_instrument_live"] and subject["pred_c_subject_only_among_single_sources"]
