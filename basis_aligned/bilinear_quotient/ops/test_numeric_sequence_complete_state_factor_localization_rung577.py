from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


MODULE_PATH = Path(__file__).with_name(
    "numeric_sequence_complete_state_factor_localization_rung577.py")
SPEC = importlib.util.spec_from_file_location("numeric_r577", MODULE_PATH)
assert SPEC and SPEC.loader
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def _toy_attention():
    batch, length, heads, width = 1, 4, 9, 2
    finals = torch.tensor([3])
    positions = torch.tensor([[0, 1, 2]])
    pattern = torch.zeros(batch, heads, length, length)
    value = torch.zeros(batch, length, heads, width)
    cached = torch.zeros_like(value)
    own = torch.zeros_like(value)
    for slot, head in enumerate(R.HEADS):
        for ordinal in range(3):
            pattern[0, head, 3, ordinal] = 1 + slot + ordinal
            value[0, ordinal, head] = torch.tensor([2 + ordinal, 4 + slot])
            own[0, ordinal, head] = value[0, ordinal, head] * .25
            cached[0, ordinal, head] = value[0, ordinal, head] * .75
    head_output = torch.einsum("bhqk,bkhd->bqhd", pattern, value)
    donor = {
        "a8_head_output": torch.arange(heads * width, dtype=torch.float32).reshape(1, heads, width),
        "a8_factors": {
            "score": torch.tensor([[[5., 6., 7.], [8., 9., 10.]]]),
            "value": torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2) + 10,
            "own": torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2) + 20,
            "cached": torch.zeros(1, 2, 3, 2),
        },
    }
    tensors = {"pattern": pattern, "value": value, "cached": cached,
               "own": own, "head_output": head_output}
    return head_output, tensors, finals, positions, donor


def test_factor_edits_equal_literal_source_term_replacement():
    head_output, tensors, finals, positions, donor = _toy_attention()
    for arm in R.FACTOR_ARMS:
        changed, delta = R.modify_a8_head_output(
            head_output, tensors, finals, positions, donor, arm)
        expected = head_output[0, 3].clone()
        for slot, head in enumerate(R.HEADS):
            for ordinal in R._factor_ordinals(arm):
                score = tensors["pattern"][0, head, 3, ordinal]
                native = score * tensors["value"][0, ordinal, head]
                donor_score = donor["a8_factors"]["score"][0, slot, ordinal]
                if arm.endswith("joint"):
                    replacement = donor_score * donor["a8_factors"]["value"][0, slot, ordinal]
                elif arm.endswith("score"):
                    replacement = donor_score * tensors["value"][0, ordinal, head]
                elif arm.endswith("cached_value"):
                    replacement = score * (tensors["own"][0, ordinal, head]
                                           + donor["a8_factors"]["cached"][0, slot, ordinal])
                else:
                    replacement = score * (donor["a8_factors"]["own"][0, slot, ordinal]
                                           + tensors["cached"][0, ordinal, head])
                expected[head] += replacement - native
        torch.testing.assert_close(changed[0, 3], expected)
        torch.testing.assert_close(delta[0], expected - head_output[0, 3])


def test_complete_head_edits_have_the_declared_scope():
    head_output, tensors, finals, positions, donor = _toy_attention()
    changed, _ = R.modify_a8_head_output(
        head_output, tensors, finals, positions, donor, "a8_h73_complete")
    for head in range(9):
        expected = donor["a8_head_output"][0, head] if head in R.HEADS else head_output[0, 3, head]
        torch.testing.assert_close(changed[0, 3, head], expected)
    changed, _ = R.modify_a8_head_output(
        head_output, tensors, finals, positions, donor, "a8_all_heads_complete")
    torch.testing.assert_close(changed[0, 3], donor["a8_head_output"][0])


def _passing_raw(arm: str) -> dict:
    raw = {arm: {family: {direction: [] for direction in R.DIRECTIONS}
                 for family in R.FAMILIES}}
    for family in R.TARGETS:
        for direction in R.DIRECTIONS:
            raw[arm][family][direction] = [
                {"effect": 1., "natural_effect": 1., "target_answer_best": True,
                 "full_vocabulary_logit_rms": 1., "intervention_vector_norm": 1.}
                for _ in range(4)]
    for direction in R.DIRECTIONS:
        raw[arm][R.RELATION][direction] = [
            {"effect": 1., "natural_effect": 1., "full_vocabulary_logit_rms": 1.,
             "intervention_vector_norm": 1.} for _ in range(4)]
    for family in R.CONTROLS:
        for direction in R.DIRECTIONS:
            common = {"registered_margin_change": .2, "full_vocabulary_logit_rms": .2,
                      "intervention_vector_norm": .2}
            if family == "sequence_step_two_conflict":
                common.update({"preference_sign_preserved": True})
            else:
                common.update({"registered_answer_best": True, "ce_increase": 0.})
            raw[arm][family][direction] = [dict(common) for _ in range(4)]
    return raw


def test_select_control_normalization_can_be_frozen_to_fit_scales():
    arm = R.SITE_ARMS[0]
    raw = _passing_raw(arm)
    fit = R.arm_report(raw, arm, 1)
    assert fit["passed"]
    frozen = {"answer_effect": .5, "logit_rms": .5, "intervention_norm": 4.}
    selected = R.arm_report(raw, arm, 1, reference_scales=frozen)
    assert selected["control_reference_scales"] == frozen
    assert not selected["controls_pass"]
    assert not selected["passed"]


def test_zero_target_scale_is_a_scientific_failure_not_an_exception():
    arm = R.SITE_ARMS[0]
    raw = _passing_raw(arm)
    for family in R.TARGETS:
        for direction in R.DIRECTIONS:
            for cell in raw[arm][family][direction]:
                cell["effect"] = 0.0
                cell["full_vocabulary_logit_rms"] = 0.0
                cell["intervention_vector_norm"] = 0.0
    report = R.arm_report(raw, arm, 1)
    assert not report["passed"]
    assert not report["controls_pass"]


def test_authority_price_and_subprocess_dryrun_are_outcome_closed():
    rows, positions = R.load_authority()
    assert R.price(rows, positions)["maximum_forwards_if_all_conditionals_open"] == 652
    environment = dict(os.environ, BQLIB_DRYRUN="1", CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run([sys.executable, str(MODULE_PATH)], env=environment,
                               check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    assert result["status"] == "dryrun_passed"
    assert result["model_loaded"] is False
    assert result["model_forwards"] == 0
    assert result["FINAL_TEST_or_OOD_opened"] is False


def test_factor_order_prefers_structurally_smaller_exact_terms():
    assert R.FACTOR_ARMS == (
        "semantic_final_score", "semantic_nonfinal_score", "semantic_all_score",
        "semantic_nonfinal_cached_value", "semantic_final_own_value",
        "semantic_nonfinal_own_value", "semantic_all_own_value",
        "semantic_final_joint", "semantic_nonfinal_joint", "semantic_all_joint",
    )
