#!/usr/bin/env python3

import run_task14_head11_3_subject_payload_lemma_direction_factorial as runner


def test_paired_authority_is_same_lemma_opposite_number_and_balanced():
    rows = runner.build_rows()
    assert len(rows) == 64
    assert {sum(row["cell_id"] == cell for row in rows)
            for cell in {row["cell_id"] for row in rows}} == {16}
    assert all(row["same_lemma_subject_number"] == row["donor_subject_number"] for row in rows)
    assert all(row["same_lemma_answer_id"] == row["donor_answer_id"] for row in rows)
    assert all(len(row["same_lemma_ids"]) == len(row["base_ids"]) for row in rows)
    assert all(row["same_lemma_ids"][:1] + row["same_lemma_ids"][2:]
               == row["base_ids"][:1] + row["base_ids"][2:] for row in rows)
    plan = runner.compile_plan()
    assert plan["price"]["model_forwards"] == 3
    assert plan["price"]["example_evaluations"] == 576
    assert plan["closed_splits"] == ["OOD"]


def test_patch_batch_keeps_recipient_score_for_both_payloads():
    import torch
    base = {"p": torch.tensor([[2., 3.]]), "u": torch.tensor([[[5.], [7.]]]),
            "head": torch.tensor([[11.]])}
    same = {"p": torch.tensor([[13., 17.]]), "u": torch.tensor([[[19.], [23.]]]),
            "head": torch.tensor([[29.]])}
    cross = {"p": torch.tensor([[31., 37.]]), "u": torch.tensor([[[41.], [43.]]]),
             "head": torch.tensor([[47.]])}
    batch = runner._patch_batch(torch.zeros(1, 2, dtype=torch.long), torch.tensor([1]),
                                base, same, cross, torch)
    assert batch["replacement_terms"][:, 0].tolist() == [69., 129., 21.]
    assert batch["replacement_heads"][:, 0].tolist() == [11., 11., 47.]


def _capability():
    return {cell: {side: 1.0 for side in ("base", "same_lemma", "cross_noun")}
            for cell in (runner.WEAK_CELL, "b", "c", "d")}


def _evidence(weak_same, weak_cross, other_same=.35):
    rows = []
    for cell in _capability():
        same = weak_same if cell == runner.WEAK_CELL else other_same
        cross = weak_cross if cell == runner.WEAK_CELL else .30
        for _ in range(16):
            for condition, recovery in (("same_lemma_payload", same),
                                        ("cross_noun_payload", cross),
                                        ("complete_head", 1.0)):
                rows.append({"cell_id": cell, "condition": condition,
                             "margin_delta": 2 * recovery,
                             "donor_ce_gain": recovery})
    return rows


def test_score_separates_rescue_asymmetry_and_broad_outcomes():
    bars = runner.compile_plan()["bars"]
    rescue = runner.score(_evidence(.36, .18, .55), _capability(), 0, 0, 0, bars)["predictions"]
    asymmetric = runner.score(_evidence(.20, .18), _capability(), 0, 0, 0, bars)["predictions"]
    broad = runner.score(_evidence(.35, .18, .35), _capability(), 0, 0, 0, bars)["predictions"]
    assert rescue["pred_b_lemma_conditioning_rescues_weak_cell"]
    assert asymmetric["pred_c_plural_to_singular_asymmetry_persists"]
    assert broad["pred_d_same_lemma_direction_symmetric_rescue"]
    assert not broad["pred_b_lemma_conditioning_rescues_weak_cell"]


def test_parent_reproduction_and_exactness_are_instrument_gates():
    bars = runner.compile_plan()["bars"]
    assert not runner.score(_evidence(.2, .18), _capability(), 0, 0, 1e-3, bars)["predictions"]["pred_a_instrument_live"]
    assert not runner.score(_evidence(.2, .18), _capability(), 1e-3, 0, 0, bars)["predictions"]["pred_a_instrument_live"]
