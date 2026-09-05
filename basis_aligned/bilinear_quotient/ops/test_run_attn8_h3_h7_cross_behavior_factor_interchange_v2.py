#!/usr/bin/env python3

import run_attn8_h3_h7_cross_behavior_factor_interchange_v2 as runner


def test_plan_has_only_fixed_v2_arms_and_price():
    plan = runner.compile_plan()
    assert plan["target_arms"] == list(runner.ARMS)
    assert plan["control_arms"] == list(runner.CONTROL_ARMS)
    assert plan["fixed_heads"] == [3, 7]
    assert plan["price"] == {"model_forwards": 6, "example_evaluations": 720,
                             "backwards": 0, "parameter_updates": 0}


def test_exact_cached_replacement_keeps_recipient_own_value():
    import torch
    recipient = {"complete": torch.zeros(2, 2), "score": torch.ones(2, 3),
                 "own": 2*torch.ones(2, 3, 2), "cached": 3*torch.ones(2, 3, 2),
                 "value": 5*torch.ones(2, 3, 2)}
    donor = {"score": 7*torch.ones(2, 3), "cached": 11*torch.ones(2, 3, 2)}
    cached = runner._replace(recipient, donor, "cached")
    score = runner._replace(recipient, donor, "score")
    joint = runner._replace(recipient, donor, "joint")
    assert torch.equal(cached, 8*torch.ones(2, 2))       # 1*(2+11) - 1*5
    assert torch.equal(score, 30*torch.ones(2, 2))       # 7*5 - 1*5
    assert torch.equal(joint, 86*torch.ones(2, 2))       # 7*(2+11) - 1*5


def _fixture(dead=False, collateral=False):
    evidence = []
    for split in ("FIT", "SELECT"):
        for format_id in ("list", "digit"):
            for direction in ("base_to_donor", "donor_to_base"):
                for index in range(4):
                    for arm in runner.ARMS:
                        effect = .8 if arm.startswith("cross") else 1.0
                        if arm == "cross_score": effect = .6
                        if arm == "cross_same_joint": effect = 0.
                        if dead and arm == "within_joint": effect = .1
                        evidence.append({"split": split, "format": format_id,
                            "direction": direction, "arm": arm, "margin_effect": effect,
                            "natural_margin_effect": 1., "donor_ce_gain": max(effect, 0.),
                            "intervention_norm": 1.})
    controls = []
    for split in ("FIT", "SELECT"):
        for control in ("repeated_list_copy", "digit_copy", "step_two"):
            for arm in runner.CONTROL_ARMS:
                for index in range(8):
                    controls.append({"split": split, "control_id": control, "arm": arm,
                        "native_preference_margin": 1.,
                        "preference_margin": -1. if collateral else 1.,
                        "preference_margin_change": -2. if collateral else 0.,
                        "answer_ce_change": .2 if collateral else 0.,
                        "intervention_norm": .2, "target_intervention_norm": 1.})
    capability = {split: {"target": {
        f"{format_id}__{direction}": {role: 1. for role in
            ("recipient", "within_donor", "cross_same", "cross_opposite")}
        for format_id in ("list", "digit") for direction in ("base_to_donor", "donor_to_base")}}
        for split in ("FIT", "SELECT")}
    exact = {"native_replay_relative_squared_error": 0.,
             "head_source_sum_relative_squared_error": 0.,
             "value_split_relative_squared_error": 0.,
             "installed_term_max_absolute_error": 0.}
    return evidence, controls, capability, exact


def test_dead_target_cannot_pass_instrument_or_transfer():
    args = _fixture(dead=True)
    predictions = runner.score(*args, runner.compile_plan()["bars"])["predictions"]
    assert not predictions["pred_a_instrument_live"]
    assert not predictions["pred_b_shared_payload_private_router"]
    assert not predictions["pred_c_shared_score_and_payload"]


def test_installed_mismatch_fails_exact_gate():
    evidence, controls, capability, exactness = _fixture()
    exactness["installed_term_max_absolute_error"] = 2e-5
    predictions = runner.score(evidence, controls, capability, exactness,
                               runner.compile_plan()["bars"])["predictions"]
    assert not predictions["pred_a_instrument_live"]


def test_collateral_damage_blocks_selective_transfer_and_is_reported_as_copy_bus():
    args = _fixture(collateral=True)
    predictions = runner.score(*args, runner.compile_plan()["bars"])["predictions"]
    assert predictions["pred_a_instrument_live"]
    assert not predictions["pred_b_shared_payload_private_router"]
    assert not predictions["pred_c_shared_score_and_payload"]
    assert predictions["pred_d_generic_numeral_or_copy_bus"]
