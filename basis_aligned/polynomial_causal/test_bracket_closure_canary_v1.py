from __future__ import annotations

import dataclasses
import inspect

import pytest
import torch
import torch.nn as nn

from bracket_closure_canary_v1 import (
    ARM_NAMES,
    BracketCanaryAuthority,
    BracketRole,
    PairwiseDisjointness,
    ProgramBinding,
    RoleBinding,
    SOURCE_CLOSURE,
    build_circuit_plan,
    make_attention_replacement,
    support_sha256,
    validate_forward_closure,
    validate_logits,
)
import circuit_campaign_runtime as campaign
from bracket_closure_masks_v1 import (
    BracketDomain, DelimiterFamily, DelimiterRegistry, build_bracket_masks,
)
from bracket_closure_tensor_v1 import (
    BracketTensorArm, PRODUCTION_STORED_VALUES, build_bracket_tensor_program,
    cyclic_derangement,
)


class FakeRotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.tensor([1.0]))


class FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(29)
        self.n_embd, self.n_head = 18, 9
        for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v", "c_proj"):
            setattr(self, name, nn.Linear(18, 18, bias=False))
        self.lamb = nn.Parameter(torch.tensor(0.25))
        self.rotary = FakeRotary()


def _authority() -> BracketCanaryAuthority:
    roles = tuple(
        RoleBinding(role, format(index + 1, "064x"), format(index + 11, "064x"),
                    format(index + 21, "064x"), format(index + 31, "064x"), 8)
        for index, role in enumerate(BracketRole)
    )
    role_list = tuple(BracketRole)
    disjoint = tuple(
        PairwiseDisjointness(left, right, 0, 0, 0)
        for index, left in enumerate(role_list) for right in role_list[index + 1:]
    )
    programs = tuple(
        ProgramBinding(arm, format(index + 101, "064x"), PRODUCTION_STORED_VALUES,
                       0, 0, True)
        for index, arm in enumerate(ARM_NAMES[1:])
    )
    return BracketCanaryAuthority(
        "a" * 40,
        tuple((path, format(index + 201, "064x")) for index, path in enumerate(SOURCE_CLOSURE)),
        "b" * 64, "c" * 64, "d" * 64, "e" * 64,
        roles, disjoint, programs,
    )


def test_plan_replaces_only_attention13_and_all_programs_have_equal_price() -> None:
    plan = build_circuit_plan()
    assert tuple(arm.name for arm in plan.arms) == ARM_NAMES
    for arm in plan.arms:
        for item in arm.attention:
            assert (item.action is campaign.ComponentAction.REPLACE) == (
                arm.name != "native" and item.site == 13
            )
        assert all(item.action is campaign.ComponentAction.NATIVE for item in arm.mlp)
    authority = _authority()
    assert {binding.stored_values for binding in authority.programs} == {
        PRODUCTION_STORED_VALUES,
    }
    assert SOURCE_CLOSURE == (
        "basis_aligned/polynomial_causal/BRACKET_CLOSURE_CANARY_V1_PREREGISTRATION.md",
        "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
        "basis_aligned/polynomial_causal/bracket_closure_canary_v1.py",
        "basis_aligned/polynomial_causal/bracket_closure_masks_v1.py",
        "basis_aligned/polynomial_causal/bracket_closure_tensor_v1.py",
        "basis_aligned/polynomial_causal/circuit_campaign_runtime.py",
        "basis_aligned/polynomial_causal/tensor_preserving_attention.py",
    )


def test_authority_requires_exact_role_disjointness_and_program_registry() -> None:
    authority = _authority()
    with pytest.raises(ValueError, match="exactly disjoint"):
        BracketCanaryAuthority(
            authority.source_commit, authority.source_files,
            authority.model_config_sha256, authority.model_weights_sha256,
            authority.delimiter_registry_sha256, authority.derangement_sha256,
            authority.roles,
            (dataclasses.replace(authority.disjointness[0], row_collisions=1),
             *authority.disjointness[1:]),
            authority.programs,
        )
    with pytest.raises(ValueError, match="distinct programs"):
        BracketCanaryAuthority(
            authority.source_commit, authority.source_files,
            authority.model_config_sha256, authority.model_weights_sha256,
            authority.delimiter_registry_sha256, authority.derangement_sha256,
            authority.roles, authority.disjointness,
            (authority.programs[0], dataclasses.replace(
                authority.programs[1], sha256=authority.programs[0].sha256,
            ), authority.programs[2]),
        )


def test_callback_is_mask_blind_and_preserves_first_value_bus() -> None:
    callback_source = inspect.getsource(make_attention_replacement)
    assert "event.tokens" not in callback_source
    assert "masks_module" not in callback_source
    program = build_bracket_tensor_program(FakeAttention(), BracketTensorArm.DELETE_H8)
    callback = make_attention_replacement(program)
    state = torch.randn(2, 4, 18)
    bus = torch.randn(2, 4, 9, 2)
    event_a = campaign.AttentionReplacementEvent(13, state, torch.zeros(2, 5, dtype=torch.long), bus)
    event_b = campaign.AttentionReplacementEvent(13, state, torch.full((2, 5), 999), bus)
    write_a, bus_a = callback(event_a)
    write_b, bus_b = callback(event_b)
    torch.testing.assert_close(write_a, write_b, rtol=0, atol=0)
    assert bus_a is bus_b is bus
    with pytest.raises(ValueError, match="wrong physical site"):
        callback(dataclasses.replace(event_a, site=12))


def test_support_hash_binds_rows_parser_metadata_and_domains() -> None:
    registry = DelimiterRegistry(
        (DelimiterFamily("round", (1,), (2,)), DelimiterFamily("square", (3,), (4,))),
        quote_control_ids=(5,), punctuation_control_ids=(6,),
    )
    rows = torch.full((2, 257), 99, dtype=torch.long).contiguous()
    rows[0, 63], rows[0, 65] = 1, 2
    rows[1, 63], rows[1, 65] = 3, 4
    masks = build_bracket_masks(
        rows, registry, (BracketDomain.PROSE, BracketDomain.CODE), first_prediction=64,
    )
    original = support_sha256(rows, masks)
    changed = rows.clone()
    changed[0, 100] = 98
    changed_masks = build_bracket_masks(
        changed, registry, (BracketDomain.PROSE, BracketDomain.CODE), first_prediction=64,
    )
    assert original != support_sha256(changed, changed_masks)


def test_exact_component_ledger_and_logit_currency_fail_closed() -> None:
    sites = tuple(campaign.SiteCallLedger(
        site,
        native_attention_calls=0 if site == 13 else 1,
        replacement_attention_calls=1 if site == 13 else 0,
        native_mlp_calls=1,
        replacement_mlp_calls=0,
    ) for site in range(18))
    closure = campaign.ForwardClosure(
        "bracket_closure_canary_v1", BracketTensorArm.DELETE_H8.value,
        campaign.ArmKind.CANDIDATE, 1, 1, 1, 2, sites, True, True,
    )
    validate_forward_closure(closure, BracketTensorArm.DELETE_H8.value, document_count=2)
    with pytest.raises(ValueError, match="component-call"):
        validate_forward_closure(
            dataclasses.replace(closure, sites=(
                *sites[:13], dataclasses.replace(sites[13], native_attention_calls=1),
                *sites[14:],
            )),
            BracketTensorArm.DELETE_H8.value, document_count=2,
        )
    with pytest.raises(ValueError, match="exact float32"):
        validate_logits(torch.zeros(2, 256, 10), documents=2)


def test_derangement_is_frozen_authority_input_not_runtime_choice() -> None:
    attention = FakeAttention()
    with pytest.raises(ValueError, match="requires"):
        build_bracket_tensor_program(attention, BracketTensorArm.DERANGED_H8)
    program = build_bracket_tensor_program(
        attention, BracketTensorArm.DERANGED_H8, permutation=cyclic_derangement(2),
    )
    assert program.cost_receipt().native_calls_per_forward == 0
