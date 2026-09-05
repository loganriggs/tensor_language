#!/usr/bin/env python3

import run_task14_head11_3_ood_fronted_self_qk_factorial as runner


def test_plan_is_fronted_16_corner_three_forward_screen():
    plan = runner.compile_plan()
    assert plan["row_count"] == 32
    assert plan["factor_bit_order"] == ["q", "k", "q2", "k2"]
    assert plan["price"] == {"model_forwards": 3, "example_evaluations": 640,
                             "backwards": 0, "parameter_updates": 0}


def test_patch_has_every_corner_and_exact_constructed_heads():
    import torch
    tokens = torch.zeros(1, 9, dtype=torch.long); finals = torch.tensor([8])
    vectors = {name: torch.ones(1, 2) for name in ("q", "q2")}
    vectors.update({name: torch.ones(1, 9, 2) for name in ("k", "k2")})
    base = {"p": torch.ones(1, 9), "u": torch.ones(1, 9, 1),
            "head": torch.tensor([[9.]]), **vectors}
    donor = {"p": 2 * torch.ones(1, 9), "u": 3 * torch.ones(1, 9, 1),
             "head": torch.tensor([[54.]]),
             "q": 2 * vectors["q"], "k": 2 * vectors["k"],
             "q2": 2 * vectors["q2"], "k2": 2 * vectors["k2"]}
    patch = runner._compile_patch_batch(tokens, finals, base, donor,
        [{"atlas_cell_id": "cell", "group_id": "group"}], torch)
    assert [spec[1] for spec in patch["specs"]] == list(range(16))
    assert float(patch["replacement_heads"][0]) == 11.0
    assert float(patch["replacement_heads"][-1]) == 56.0


def test_mobius_and_shapley_are_exact():
    values = {mask: float(mask * mask - 3 * mask) for mask in range(16)}
    _, _, reconstruction, efficiency = runner._mobius_shapley(values)
    assert reconstruction < 1e-12
    assert efficiency < 1e-12


def _fake_evidence(pair1=.8, pair2=.8, interaction=.2):
    rows = []
    for cell in ("fronted_singular_to_fronted_plural", "fronted_plural_to_fronted_singular"):
        for index in range(16):
            for mask in range(16):
                value = 0.0
                if mask == 3: value = pair1
                if mask == 12: value = pair2
                if mask == 15: value = pair1 + pair2 + interaction
                rows.append({"row_id": f"{cell}-{index}", "group_id": f"g{index:02d}",
                             "atlas_cell_id": cell, "mask": mask,
                             "donor_margin": value, "margin_delta": value,
                             "donor_ce": 3-value, "donor_ce_gain": value})
    return rows


def test_score_applies_registered_pair_and_interaction_bars():
    capability = {cell: {"base": 1., "donor": 1.} for cell in
                  ("fronted_singular_to_fronted_plural", "fronted_plural_to_fronted_singular")}
    exact = {name: 0. for name in ("native_replay_max_absolute_logit_error",
        "source_term_sum_max_absolute_error", "pre_subject_value_max_absolute_error",
        "endpoint_metric_reproduction_max_absolute_error", "installed_term_max_absolute_error")}
    live = {"minimum_factor_or_score_norm": 1.,
            "minimum_recipient_donor_factor_or_score_difference": 1., "all_finite": True}
    predictions = runner.score(_fake_evidence(), capability, exact, live,
                               runner.compile_plan()["bars"])["predictions"]
    assert predictions == {"pred_a_instrument_live": True,
                           "pred_b_qk1_pair_sufficiency": False,
                           "pred_c_qk2_pair_sufficiency": False,
                           "pred_d_branch_composition_dependence": True}
