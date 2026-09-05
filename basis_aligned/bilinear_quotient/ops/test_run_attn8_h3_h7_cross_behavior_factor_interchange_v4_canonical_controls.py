#!/usr/bin/env python3

import run_attn8_h3_h7_cross_behavior_factor_interchange_v4_canonical_controls as runner


def _fixture(damage=None, incapable=None, dead=None):
    evidence = []
    for family in runner.authority.FAMILIES:
        for split in ("FIT", "SELECT"):
            for arm in runner.ARMS:
                for index in range(4):
                    harmed = damage == (family, split, arm)
                    evidence.append({"family_id": family, "split": split, "arm": arm,
                        "native_answer_correct": incapable != (family, split, arm),
                        "post_answer_correct": not harmed,
                        "answer_ce_change": .2 if harmed else 0.,
                        "margin_change": 1. if harmed else 0.,
                        "intervention_norm": 0.01 if dead == (family, split, arm) else 1.})
    exactness = {"native_replay_relative_squared_error": 0.,
                 "head_source_sum_relative_squared_error": 0.,
                 "value_split_relative_squared_error": 0.,
                 "installed_term_max_absolute_error": 0.}
    target = {"target_exact": True, "target_live": True, "cached_transfer": True,
              "intervention_scales": {split: {arm: 1. for arm in runner.ARMS}
                                      for split in ("FIT", "SELECT")},
              "margin_scales": {split: {arm: 1. for arm in runner.ARMS}
                                for split in ("FIT", "SELECT")}}
    return evidence, exactness, target, runner.compile_plan()["bars"]


def test_decisive_cached_control_damage_reports_broad_service():
    args = _fixture(damage=("sequence_digit_copy_control", "FIT", "cached"))
    predictions = runner.score(*args)["predictions"]
    assert predictions["pred_a_instrument_live"]
    assert predictions["pred_c_broad_numeral_or_copy_service"]
    assert not predictions["pred_b_shared_cached_payload_private_router"]


def test_score_only_damage_keeps_cached_candidate_narrower():
    args = _fixture(damage=("sequence_word_copy_control", "SELECT", "score"))
    predictions = runner.score(*args)["predictions"]
    assert predictions["pred_d_score_or_joint_collateral_only"]
    assert not predictions["pred_b_shared_cached_payload_private_router"]


def test_incapable_or_dead_decisive_control_invalidates_instrument():
    incapable = _fixture(incapable=("list_step_two_conflict", "FIT", "joint"))
    dead = _fixture(dead=("sequence_digit_copy_control", "SELECT", "cached"))
    assert not runner.score(*incapable)["predictions"]["pred_a_instrument_live"]
    assert not runner.score(*dead)["predictions"]["pred_a_instrument_live"]


def test_secondary_failure_is_reported_but_not_decisive():
    args = _fixture(damage=(runner.authority.SECONDARY, "FIT", "cached"))
    predictions = runner.score(*args)["predictions"]
    assert predictions["pred_b_shared_cached_payload_private_router"]
