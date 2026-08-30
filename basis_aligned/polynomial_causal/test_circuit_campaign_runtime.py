from __future__ import annotations

import copy
import pickle
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import bilin18_observed_model_facade as facade
import circuit_campaign_runtime as runtime


class TinyAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.proj.weight)
        self.calls = 0

    def forward(self, state, first_value=None):
        self.calls += 1
        value = state.unsqueeze(2)
        return 0.01 * self.proj(state), value if first_value is None else first_value


class TinyMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.proj.weight)
        self.calls = 0

    def forward(self, state):
        self.calls += 1
        return 0.02 * self.proj(state)


class TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.lambdas = nn.Parameter(torch.tensor([0.9, 0.1]))
        self.attn = TinyAttention(width)
        self.mlp = TinyMLP(width)


class TinyModel(nn.Module):
    def __init__(self, *, sites: int = 4, width: int = 8, vocab: int = 13) -> None:
        super().__init__()
        torch.manual_seed(123)
        self.wte = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList([TinyBlock(width) for _ in range(sites)])
        self.transformer = SimpleNamespace(wte=self.wte, h=self.blocks)
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.config = SimpleNamespace(vocab_size=vocab)


def _native_forward(model, tokens):
    return facade.forward_with_dispatch(
        model,
        tokens,
        lambda event: event.block.attn(event.state, event.first_value),
        lambda event: event.block.mlp(event.state),
        require_production=False,
    )


def _plan(*, candidate_attention=None, candidate_mlp=None):
    native = runtime.ArmPlan.build(
        "native", runtime.ArmKind.NATIVE, site_count=4,
    )
    candidate = runtime.ArmPlan.build(
        "candidate",
        runtime.ArmKind.CANDIDATE,
        site_count=4,
        attention_replacements=candidate_attention,
        mlp_replacements=candidate_mlp,
    )
    return runtime.CircuitPlan("test-circuit", 4, (native, candidate))


def test_native_arm_is_exact_identity_with_complete_call_ledger():
    model = TinyModel()
    reference = copy.deepcopy(model)
    tokens = torch.randint(0, 13, (2, 5))
    expected = _native_forward(reference, tokens)
    plan = _plan(candidate_mlp={1: "zero"})
    owner = runtime.CircuitForwardOwner(plan=plan, arm="native")
    observed = owner.run(model, tokens, require_production=False)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    closure = owner.closure
    assert closure.attempted_outer_forwards == 1
    assert closure.completed_outer_forwards == closure.outer_returns == 1
    assert closure.document_count == 2
    assert all(
        (site.native_attention_calls, site.replacement_attention_calls,
         site.native_mlp_calls, site.replacement_mlp_calls) == (1, 0, 1, 0)
        for site in closure.sites
    )


def test_candidate_replacements_bypass_native_components_and_preserve_interfaces():
    model = TinyModel()
    plan = _plan(candidate_attention={0: "attention-zero"}, candidate_mlp={1: "mlp-zero"})
    observed = {}

    def replace_attention(event):
        observed["attention"] = event
        assert not hasattr(event, "block")
        return torch.zeros_like(event.state), event.state.unsqueeze(2)

    def replace_mlp(event):
        observed["mlp"] = event
        assert not hasattr(event, "block")
        assert len(event.prior_writes) == 1
        return torch.zeros_like(event.state)

    model.blocks[0].attn.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("replaced native attention was called")
    )
    model.blocks[1].mlp.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("replaced native MLP was called")
    )
    tokens = torch.randint(0, 13, (2, 5))
    hooks_before = tuple(
        (len(block.attn._forward_hooks), len(block.mlp._forward_hooks))
        for block in model.blocks
    )
    owner = runtime.CircuitForwardOwner(
        plan=plan,
        arm="candidate",
        attention_replacements={"attention-zero": replace_attention},
        mlp_replacements={"mlp-zero": replace_mlp},
    )
    logits = owner.run(model, tokens, require_production=False)
    hooks_after = tuple(
        (len(block.attn._forward_hooks), len(block.mlp._forward_hooks))
        for block in model.blocks
    )
    assert logits.shape == (2, 5, 13)
    assert hooks_before == hooks_after == ((0, 0),) * 4
    closure = owner.closure
    assert closure.candidate_native_call_prohibition_passed
    assert closure.sites[0].native_attention_calls == 0
    assert closure.sites[0].replacement_attention_calls == 1
    assert closure.sites[1].native_mlp_calls == 0
    assert closure.sites[1].replacement_mlp_calls == 1
    assert model.blocks[0].attn.calls == 0 and model.blocks[1].mlp.calls == 0
    assert observed["attention"].tokens is not tokens
    assert observed["mlp"].attention_write.data_ptr() != observed["mlp"].state.data_ptr()


def test_owner_is_exactly_once_noncopyable_and_nonserializable():
    model = TinyModel()
    plan = _plan(candidate_mlp={1: "zero"})
    owner = runtime.CircuitForwardOwner(
        plan=plan,
        arm="candidate",
        mlp_replacements={"zero": lambda event: torch.zeros_like(event.state)},
    )
    with pytest.raises(TypeError, match="copied"):
        copy.copy(owner)
    with pytest.raises(TypeError, match="deep-copied"):
        copy.deepcopy(owner)
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(owner)
    with pytest.raises(AttributeError, match="sealed"):
        owner._state = "spent"
    with pytest.raises(TypeError):
        owner._native_mlp[0] = 99
    tokens = torch.randint(0, 13, (1, 3))
    owner.run(model, tokens, require_production=False)
    with pytest.raises(RuntimeError, match="already spent"):
        owner.run(model, tokens, require_production=False)


def test_callback_failure_poisons_owner_and_preserves_partial_attempt_ledger():
    model = TinyModel()
    plan = _plan(candidate_mlp={2: "fail"})

    def fail(_event):
        raise RuntimeError("injected replacement failure")

    owner = runtime.CircuitForwardOwner(
        plan=plan, arm="candidate", mlp_replacements={"fail": fail},
    )
    tokens = torch.randint(0, 13, (1, 3))
    with pytest.raises(RuntimeError, match="injected"):
        owner.run(model, tokens, require_production=False)
    ledger = owner.failure_ledger
    assert ledger[0].native_attention_calls == ledger[0].native_mlp_calls == 1
    assert ledger[2].replacement_mlp_calls == 1
    assert ledger[3].native_attention_calls == ledger[3].native_mlp_calls == 0
    with pytest.raises(RuntimeError, match="spent or failed"):
        owner.run(model, tokens, require_production=False)
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = owner.closure


def test_callback_registry_must_exactly_match_selected_arm():
    plan = _plan(candidate_attention={0: "a"}, candidate_mlp={1: "m"})
    attention = lambda event: (torch.zeros_like(event.state), event.state.unsqueeze(2))
    mlp = lambda event: torch.zeros_like(event.state)
    with pytest.raises(ValueError, match="exactly match"):
        runtime.CircuitForwardOwner(
            plan=plan, arm="candidate", attention_replacements={}, mlp_replacements={"m": mlp},
        )
    with pytest.raises(ValueError, match="exactly match"):
        runtime.CircuitForwardOwner(
            plan=plan, arm="candidate",
            attention_replacements={"a": attention, "extra": attention},
            mlp_replacements={"m": mlp},
        )


def test_plans_are_typed_complete_and_immutable():
    with pytest.raises(ValueError, match="at least one"):
        runtime.ArmPlan.build("candidate", runtime.ArmKind.CANDIDATE, site_count=4)
    with pytest.raises(ValueError, match="native arm"):
        runtime.ArmPlan.build(
            "native", runtime.ArmKind.NATIVE, site_count=4, mlp_replacements={0: "bad"},
        )
    good = _plan(candidate_mlp={1: "zero"})
    with pytest.raises(ValueError, match="cover every site"):
        runtime.CircuitPlan(
            "bad",
            4,
            (good.arms[0], runtime.ArmPlan(
                "candidate", runtime.ArmKind.CANDIDATE,
                good.arms[1].attention[:-1], good.arms[1].mlp,
            )),
        )
    with pytest.raises((AttributeError, TypeError)):
        good.name = "mutated"  # type: ignore[misc]


def test_wrong_model_depth_fails_before_any_component_call_and_poison_owner():
    model = TinyModel(sites=3)
    owner = runtime.CircuitForwardOwner(
        plan=_plan(candidate_mlp={1: "zero"}),
        arm="candidate",
        mlp_replacements={"zero": lambda event: torch.zeros_like(event.state)},
    )
    with pytest.raises(ValueError, match="depth"):
        owner.run(model, torch.randint(0, 13, (1, 3)), require_production=False)
    assert all(site.native_attention_calls == 0 for site in owner.failure_ledger)
