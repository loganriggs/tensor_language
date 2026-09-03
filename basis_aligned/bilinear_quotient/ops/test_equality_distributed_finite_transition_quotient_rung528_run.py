from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).with_name("equality_distributed_finite_transition_quotient_rung528_run.py")
SPEC = importlib.util.spec_from_file_location("r528_run", PATH)
R = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(R)


class FakeAttention(torch.nn.Module):
    def forward(self, state, first_value):
        if first_value is None:
            first_value = torch.zeros_like(state)
        return torch.zeros_like(state), first_value


class FakeMLP(torch.nn.Module):
    def forward(self, state):
        return torch.zeros_like(state)


class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("lambdas", torch.tensor([1.0, 0.0]))
        self.attn = FakeAttention()
        self.mlp = FakeMLP()


class FakeTransformer(torch.nn.Module):
    def __init__(self, vocab, dimension):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab, dimension)
        self.h = torch.nn.ModuleList([FakeBlock() for _ in range(18)])


class FakeModel(torch.nn.Module):
    def __init__(self, vocab=13, dimension=8):
        super().__init__()
        self.transformer = FakeTransformer(vocab, dimension)
        self.lm_head = torch.nn.Linear(dimension, vocab, bias=False)


def test_frozen_dependencies_and_population_hold():
    observed, population = R.validate_dependencies()
    rows, _task, _circuit, _scales, discovery, validation, _metadata = population
    assert len(observed) == len(R.FROZEN_SHA256)
    assert tuple(rows.shape) == (1000, 257)
    assert (len(discovery), len(validation)) == (32, 30)


def test_scaled_boundary_rounds_once_and_recovers_unit_scale():
    absent = torch.tensor([1.0, -2.0, 3.0], dtype=torch.bfloat16)
    action = torch.tensor([4.0, 5.0, -6.0], dtype=torch.bfloat16)
    delta = action.float() - absent.float()
    assert torch.equal(R.scaled_boundary(absent, delta, 1.0), action)
    assert torch.equal(R.scaled_boundary(absent, delta, 0.0), absent)


def test_scaled_boundary_is_fail_closed():
    with pytest.raises(ValueError, match="shapes"):
        R.scaled_boundary(torch.zeros(2), torch.zeros(3), 1.0)
    with pytest.raises(ValueError, match="nonfinite"):
        R.scaled_boundary(torch.zeros(2), torch.ones(2), float("nan"))


def test_toy_raw_boundary_override_occurs_after_mlp12(monkeypatch):
    dimension = 8
    vocab = 13
    model = FakeModel(vocab=vocab, dimension=dimension)
    tokens = torch.randint(0, vocab, (2, 5))
    monkeypatch.setattr(R, "D", dimension)
    monkeypatch.setattr(R.facade, "LOGIT_VOCAB", vocab)
    monkeypatch.setattr(R.facade, "validate_production_model", lambda _model: None)
    monkeypatch.setattr(R.facade, "validate_tokens", lambda _tokens, production_shape: None)

    def fake_factor(state, first_value, attention, site, event_tokens):
        write, next_value = attention(state, first_value)
        return write, {}, torch.ones(1), 0.0

    monkeypatch.setattr(R.r505.action_parent.factor_parent, "_factor_site", fake_factor)
    direct_logits, direct_state, _diag, direct_audit = R.boundary_forward(
        model, tokens, direct=True)
    native_logits, native_state, _diag, native_audit = R.boundary_forward(
        model, tokens, action="N", capture_writes=("a14", "m17"))
    assert torch.equal(direct_logits, native_logits)
    assert torch.equal(direct_state["native_boundary"], native_state["native_boundary"])
    assert direct_audit["native_attention"] == 18
    assert native_audit["replayed_attention"] == 3

    replacement = native_state["native_boundary"] + 0.25
    changed_logits, changed_state, diagnostics, audit = R.boundary_forward(
        model, tokens, action="N", boundary_override=replacement,
        patch_writes={"a14": native_state["a14"], "m17": native_state["m17"]})
    assert torch.equal(changed_state["native_boundary"], native_state["native_boundary"])
    assert torch.equal(changed_state["effective_boundary"], replacement)
    assert audit["boundary_overrides"] == 1
    assert audit["write_patches"] == 2
    assert diagnostics["boundary_override_rms"] == pytest.approx(0.25)
    assert not torch.equal(changed_logits, native_logits)


def test_dry_run_freezes_price_and_keeps_outcomes_closed():
    report = R.dry_run()
    assert report["status"] == "dry_run_passed"
    assert report["model_loaded"] is False
    assert report["outcomes_opened"] is False
    assert report["unconditional_discovery_forwards"] == 1984
    assert report["maximum_conditional_forwards"] == 11485
    assert report["planted_suite_passes"] is True


def test_effect_views_preserve_source_half_continuation_order():
    documents = 4
    tags = 3
    task_counts = torch.ones(documents, len(R.CELLS), dtype=torch.float64)
    circuit_counts = torch.ones(2, 2, tags, dtype=torch.float64)
    task_sums = torch.zeros(4, 4, documents, len(R.CELLS), dtype=torch.float64)
    circuit_sums = torch.zeros(4, 4, 2, 2, tags, dtype=torch.float64)
    for source in range(4):
        for continuation in range(4):
            value = 10.0 * source + continuation + 1.0
            task_sums[source, continuation] = value
            circuit_sums[source, continuation, :, 0] = value
    views = R._effect_views(task_sums, circuit_sums, task_counts, circuit_counts)
    assert views["task_halves"].shape == (4, 2, 4, 4)
    assert views["circuit_halves"].shape == (4, 2, 4, tags)
    assert views["circuit_halves"][2, 1, 3, 0] == pytest.approx(24.0)
    assert views["task_pooled_full"][1, 2, 0] == pytest.approx(13.0)


def test_discovery_controls_keep_all_three_planted_positive_relations():
    generator = torch.Generator().manual_seed(528)
    circuit = torch.randn(2, 4, 32, generator=generator, dtype=torch.float64) * .01
    task = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64) * .01
    unit = {
        "circuit_halves": torch.stack((circuit, circuit / 1.2, circuit / .8, circuit / 1.5)),
        "task_halves": torch.stack((task, task / 1.2, task / .8, task / 1.5)),
    }
    wrong = {
        "circuit_halves": torch.stack((-circuit, -.5 * circuit)),
        "task_halves": torch.stack((-task, -.5 * task)),
    }
    candidates, checks = R.discover_candidates(unit, wrong)
    assert [row["source"] for row in candidates] == ["P", "Z7", "Z8"]
    assert [row["beta"] for row in candidates] == pytest.approx([1.2, .8, 1.5])
    assert all(checks[source]["control_margin"] >= .10 for source in R.CANDIDATE_SOURCES)


def test_physical_scoring_requires_both_directions():
    generator = torch.Generator().manual_seed(529)
    native_circuit = torch.randn(2, 4, 32, generator=generator, dtype=torch.float64) * .01
    native_task = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64) * .01
    source_circuit = native_circuit / 1.2
    source_task = native_task / 1.2
    discovery = {
        "circuit_halves": torch.stack((native_circuit, source_circuit)),
        "task_halves": torch.stack((native_task, source_task)),
    }
    physical = {
        "circuit_pooled": torch.stack((torch.stack((native_circuit[1], source_circuit[1])),)),
        "task_pooled": torch.stack((torch.stack((native_task[1], source_task[1])),)),
    }
    candidate = [{"source": "P", "beta": 1.2}]
    passing, checks = R.score_physical(discovery, physical, candidate)
    assert passing == candidate
    assert checks["P"]["holds"]
    physical["circuit_pooled"][0, 1].zero_()
    passing, _checks = R.score_physical(discovery, physical, candidate)
    assert passing == []
