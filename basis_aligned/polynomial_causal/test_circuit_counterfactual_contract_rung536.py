import copy

import pytest

import circuit_counterfactual_contract_rung536 as contract


def test_all_pilots_pass_contract():
    for pilot in contract.PILOTS:
        contract.validate_pilot(pilot)


def test_every_pilot_has_two_variable_changing_families_and_answer_changing_interchange():
    for pilot in contract.PILOTS:
        families = pilot["counterfactual_families"]
        assert sum(family["proposed_variable_changes"] for family in families) >= 2
        assert any(family["answer_changes"] for family in families)
        assert any(family["intervention_role"] == "invariance" for family in families)


def test_match_break_is_necessity_not_answer_changing_interchange():
    induction = contract.PILOTS[0]
    match_break = next(
        family for family in induction["counterfactual_families"]
        if family["family_id"] == "match_break_payload_preserved"
    )
    assert match_break["intervention_role"] == "necessity"
    assert match_break["proposed_variable_changes"]
    assert not match_break["answer_changes"]


def test_one_counterfactual_family_is_rejected():
    pilot = copy.deepcopy(contract.PILOTS[0])
    pilot["counterfactual_families"] = pilot["counterfactual_families"][:1]
    with pytest.raises(AssertionError):
        contract.validate_pilot(pilot)


def test_weakly_controlled_family_is_rejected():
    pilot = copy.deepcopy(contract.PILOTS[0])
    pilot["counterfactual_families"][0]["controls"] = ["one control"]
    with pytest.raises(AssertionError):
        contract.validate_pilot(pilot)


def test_mislabeled_necessity_family_is_rejected():
    pilot = copy.deepcopy(contract.PILOTS[0])
    match_break = next(
        family for family in pilot["counterfactual_families"]
        if family["family_id"] == "match_break_payload_preserved"
    )
    match_break["intervention_role"] = "interchange"
    with pytest.raises(AssertionError):
        contract.validate_pilot(pilot)
