from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import circuit_campaign_runtime as runtime
import circuit_campaign_statistics as statistics
from circuit_newline_fixed_crew_v1 import (
    CANARY_HEAD,
    FIVE_HEAD_CONTROL,
    FIVE_HEAD_CREW,
    NewlineArm,
    NewlineAttentionExecutor,
    NewlineMaskSpec,
    NewlineScope,
    build_newline_masks,
    build_newline_plan,
    expected_call_ledger,
    head_weights,
    newline_coordinate_specs,
    newline_price,
    replacement_sites,
)


class TinyNativeAttention(nn.Module):
    def __init__(self, width: int = 18, heads: int = 9) -> None:
        super().__init__()
        torch.manual_seed(71)
        for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v", "c_proj"):
            setattr(self, name, nn.Linear(width, width, bias=False))
        self.lamb = nn.Parameter(torch.tensor(0.25))
        self.n_head = heads
        self.n_embd = width
        self.rotary = SimpleNamespace(inv_freq=torch.ones(width // heads // 2))


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({"attn": TinyNativeAttention()}) for _ in range(18)
        ])
        # ModuleDict does not provide attribute access; mirror the production surface.
        self.transformer = SimpleNamespace(h=tuple(
            SimpleNamespace(attn=block["attn"]) for block in self.blocks
        ))


def test_frozen_heads_and_shifted_controls_are_exact() -> None:
    assert CANARY_HEAD == (12, 6)
    assert FIVE_HEAD_CREW == ((7, 2), (8, 2), (10, 2), (11, 0), (12, 6))
    assert FIVE_HEAD_CONTROL == ((7, 3), (8, 3), (10, 3), (11, 1), (12, 7))
    assert replacement_sites(NewlineScope.CANARY) == (12,)
    assert replacement_sites(NewlineScope.FIVE_HEAD) == (7, 8, 10, 11, 12)


@pytest.mark.parametrize("scope", tuple(NewlineScope))
def test_head_projectors_are_constant_and_arm_prices_match(scope) -> None:
    for site in replacement_sites(scope):
        exact = head_weights(scope, NewlineArm.EXACT, site)
        remove = head_weights(scope, NewlineArm.REMOVE, site)
        control = head_weights(scope, NewlineArm.HEAD_LABEL_CONTROL, site)
        assert exact.tolist() == [1.0] * 9
        assert int(remove.eq(0).sum()) == int(control.eq(0).sum()) == 1
        assert not torch.equal(remove, control)
        assert exact.numel() == remove.numel() == control.numel() == 9


def test_canary_and_five_head_plans_are_complete_and_have_exact_ledgers() -> None:
    for scope in NewlineScope:
        plan = build_newline_plan(scope)
        assert tuple(arm.name for arm in plan.arms) == tuple(arm.value for arm in NewlineArm)
        selected = set(replacement_sites(scope))
        for arm in NewlineArm:
            ledger = expected_call_ledger(scope, arm)
            assert len(ledger) == 18
            for item in ledger:
                expected = item.site in selected and arm is not NewlineArm.NATIVE
                assert item.replacement_attention_calls == int(expected)
                assert item.native_attention_calls == 1 - int(expected)
                assert (item.native_mlp_calls, item.replacement_mlp_calls) == (1, 0)
            plan_arm = plan.arm(arm.value)
            assert all(component.action is runtime.ComponentAction.NATIVE for component in plan_arm.mlp)


def test_literal_prices_include_projector_and_are_known_answers() -> None:
    canary = newline_price(NewlineScope.CANARY)
    crew = newline_price(NewlineScope.FIVE_HEAD)
    assert canary.stored_values_per_site == 7_962_698
    assert canary.total_stored_values == 7_962_698
    assert crew.total_stored_values == 39_813_490
    assert crew.total_operations == 5 * canary.total_operations
    assert canary.token_table_values == canary.native_calls_at_replaced_sites == 0
    assert canary.total_input_support


def test_statistics_plan_uses_generic_typed_coordinates_without_sign_fabrication() -> None:
    canary = newline_coordinate_specs(NewlineScope.CANARY)
    assert len(canary) == 5
    assert {item.role for item in canary} == {"CANARY_SELECT"}
    assert {item.kind for item in canary} == {
        statistics.CoordinateKind.TARGET_DAMAGE,
        statistics.CoordinateKind.SPECIFICITY,
        statistics.CoordinateKind.COLLATERAL,
        statistics.CoordinateKind.EXTRACTION_RECOVERY,
    }
    control = next(item for item in canary if item.name.endswith("head_label_control"))
    assert control.native_arm == NewlineArm.HEAD_LABEL_CONTROL.value
    assert control.candidate_arm == NewlineArm.REMOVE.value
    full = newline_coordinate_specs(NewlineScope.FIVE_HEAD)
    assert len(full) == 10 and {item.role for item in full} == {"FINAL", "OOD"}


def test_executor_is_owned_ordered_complete_and_native_storage_disjoint() -> None:
    model = TinyModel()
    executor = NewlineAttentionExecutor.from_model(
        model, scope=NewlineScope.CANARY, arm=NewlineArm.REMOVE,
    )
    callbacks = executor.callbacks()
    callback = next(iter(callbacks.values()))
    native = model.transformer.h[12].attn
    for module in native.modules():
        if isinstance(module, nn.Linear):
            module.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("native projection called")
            )
    event = runtime.AttentionReplacementEvent(
        site=12,
        state=torch.randn(2, 5, 18),
        tokens=torch.zeros(2, 5, dtype=torch.long),
        first_value=torch.randn(2, 5, 9, 2),
    )
    write, bus = callback(event)
    assert write.shape == (2, 5, 18)
    assert bus.data_ptr() == event.first_value.data_ptr()
    assert executor.closure.sites == ((12, 1),)
    assert executor.closure.ordered and executor.closure.complete and executor.closure.closed
    with pytest.raises(RuntimeError, match="closed"):
        callback(event)


def test_executor_rejects_skipped_or_reordered_sites() -> None:
    executor = NewlineAttentionExecutor.from_model(
        TinyModel(), scope=NewlineScope.FIVE_HEAD, arm=NewlineArm.EXACT,
    )
    callback = next(iter(executor.callbacks().values()))
    with pytest.raises(RuntimeError, match="skipped"):
        callback(runtime.AttentionReplacementEvent(
            site=8,
            state=torch.randn(1, 3, 18),
            tokens=torch.zeros(1, 3, dtype=torch.long),
            first_value=torch.randn(1, 3, 9, 2),
        ))
    assert executor.closure.closed and not executor.closure.complete


def test_executor_cannot_route_on_newline_tokens_or_score_masks() -> None:
    model = TinyModel()
    left = NewlineAttentionExecutor.from_model(
        model, scope=NewlineScope.CANARY, arm=NewlineArm.REMOVE,
    )
    right = NewlineAttentionExecutor.from_model(
        model, scope=NewlineScope.CANARY, arm=NewlineArm.REMOVE,
    )
    state = torch.randn(1, 4, 18)
    bus = torch.randn(1, 4, 9, 2)
    first = runtime.AttentionReplacementEvent(
        12, state, torch.full((1, 4), 10, dtype=torch.long), bus,
    )
    second = runtime.AttentionReplacementEvent(
        12, state.clone(), torch.full((1, 4), 999, dtype=torch.long), bus.clone(),
    )
    left_write, _ = next(iter(left.callbacks().values()))(first)
    right_write, _ = next(iter(right.callbacks().values()))(second)
    assert torch.equal(left_write, right_write)


def _mask_spec() -> NewlineMaskSpec:
    return NewlineMaskSpec(
        newline_token_ids=(10,),
        punctuation_token_ids=(11,),
        capitalized_token_ids=(12,),
        quote_bracket_token_ids=(13,),
        first_prediction=2,
        jitter_offsets=(2, -2, 3, -3, 4, -4),
        random_seed=19,
    )


def test_masks_are_score_only_disjoint_and_document_count_matched() -> None:
    rows = torch.tensor([
        [0, 1, 2, 3, 10, 4, 11, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 10, 6, 12, 7, 13, 8],
    ], dtype=torch.long)
    masks = build_newline_masks(rows, _mask_spec())
    assert masks.newline_target.sum(1).tolist() == [1, 1]
    assert masks.position_jitter.sum(1).tolist() == [1, 1]
    assert masks.matched_random.sum(1).tolist() == [1, 1]
    named = torch.stack([
        masks.newline_target, masks.position_jitter, masks.matched_random,
        masks.punctuation, masks.capitalized, masks.quote_bracket,
    ]).to(torch.int8).sum(0)
    assert int(named.max()) == 1
    assert not masks.newline_target[:, :2].any()
    assert torch.equal(masks.global_off_target, ~masks.newline_target & torch.tensor(
        [[False, False] + [True] * 9, [False, False] + [True] * 9],
        dtype=torch.bool,
    ))


def test_masks_fail_closed_on_token_overlap_and_insufficient_controls() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        NewlineMaskSpec((10,), (10,), (12,), (13,), first_prediction=0)
    rows = torch.tensor([[0, 10, 10, 10]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="position-jitter"):
        build_newline_masks(rows, NewlineMaskSpec(
            (10,), (11,), (12,), (13,), first_prediction=0,
            jitter_offsets=(1, -1),
        ))


def test_invalid_head_and_price_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="nonnative"):
        head_weights(NewlineScope.CANARY, NewlineArm.NATIVE, 12)
    with pytest.raises(ValueError, match="outside"):
        head_weights(NewlineScope.CANARY, NewlineArm.REMOVE, 11)
    with pytest.raises(ValueError, match="head-compatible"):
        newline_price(NewlineScope.CANARY, width=17)
