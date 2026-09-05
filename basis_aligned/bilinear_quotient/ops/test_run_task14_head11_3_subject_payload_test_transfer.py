#!/usr/bin/env python3

import run_task14_head11_3_subject_payload_test_transfer as runner


def test_frozen_test_reuse_authority_and_three_forward_price():
    plan = runner.compile_plan()
    rows = runner.authority.build_rows()
    assert len(rows) == 64
    assert len({row["cell_id"] for row in rows}) == 4
    assert all(row["base_subject_number"] != row["donor_subject_number"] for row in rows)
    assert all(row["base_ids"][1] != row["donor_ids"][1] for row in rows)
    assert plan["price"] == {"model_forwards": 3, "example_evaluations": 384,
                             "backwards": 0, "parameter_updates": 0}
    assert plan["scope"] == "TEST_REUSE_NEW_INTERVENTION"
    assert plan["closed_splits"] == ["OOD"]
    assert runner.PRIOR_ART_SHA256 == "0960cb4b6a5893fcd0f0ec10b6b25937cd2e6d3ba0e7cc783c49401abcc1734a"


def test_patch_batch_keeps_recipient_score_for_payload_and_uses_donor_head_for_ceiling():
    import torch
    base = {"p": torch.tensor([[2., 3.], [5., 7.]]),
            "u": torch.tensor([[[11.], [13.]], [[17.], [19.]]]),
            "head": torch.tensor([[101.], [103.]])}
    donor = {"p": torch.tensor([[107., 109.], [127., 131.]]),
             "u": torch.tensor([[[23.], [29.]], [[31.], [37.]]]),
             "head": torch.tensor([[137.], [139.]])}
    batch = runner._patch_batch(torch.zeros(2, 3, dtype=torch.long), torch.tensor([2, 2]),
                                base, donor, torch)
    assert batch["replacement_terms"][:, 0].tolist() == [87., 259., 39., 133.]
    assert batch["replacement_heads"][:, 0].tolist() == [101., 103., 137., 139.]


def _evidence(payload_recovery=.4, payload_direction=1.0, payload_ce=.4,
              complete_margin=2.0, complete_ce=1.0):
    rows = []
    positive = int(16 * payload_direction)
    for cell in ("a", "b", "c", "d"):
        for index in range(16):
            rows.append({"cell_id": cell, "condition": "complete_head",
                         "margin_delta": complete_margin, "donor_ce_gain": complete_ce})
            value = payload_recovery * complete_margin if index < positive else -0.01
            rows.append({"cell_id": cell, "condition": "subject_payload",
                         "margin_delta": value, "donor_ce_gain": payload_ce})
    return rows


def _capability(value=1.0):
    return {cell: {"base": value, "donor": value} for cell in ("a", "b", "c", "d")}


def test_score_requires_every_cell_and_live_exact_instrument():
    bars = runner.compile_plan()["bars"]
    passed = runner.score(_evidence(), _capability(), 0, 0, bars)["predictions"]
    assert passed["pred_a_instrument_live"]
    assert passed["pred_b_subject_payload_transfers_each_cell"]

    weak = _evidence()
    for row in weak:
        if row["cell_id"] == "d" and row["condition"] == "subject_payload":
            row["margin_delta"] = .2
    scored = runner.score(weak, _capability(), 0, 0, bars)
    assert scored["predictions"]["pred_a_instrument_live"]
    assert not scored["predictions"]["pred_b_subject_payload_transfers_each_cell"]
    assert scored["predictions"]["pred_c_subject_payload_fails_some_cell"]

    assert not runner.score(_evidence(), _capability(.8), 0, 0, bars)["predictions"]["pred_a_instrument_live"]
    assert not runner.score(_evidence(), _capability(), 1e-3, 0, bars)["predictions"]["pred_a_instrument_live"]


def test_score_requires_positive_ce_recovery_in_every_cell():
    bars = runner.compile_plan()["bars"]
    evidence = _evidence()
    for row in evidence:
        if row["cell_id"] == "c" and row["condition"] == "subject_payload":
            row["donor_ce_gain"] = 0.0
    scored = runner.score(evidence, _capability(), 0, 0, bars)
    assert scored["predictions"]["pred_a_instrument_live"]
    assert not scored["predictions"]["pred_b_subject_payload_transfers_each_cell"]
