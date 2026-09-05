#!/usr/bin/env python3
"""Focused CPU tests for the numerical-only MLP8 polarization repair."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial as v1
import run_task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v2 as run


def _exact(value=0.0):
    return {name: value for name in (
        "native_replay_max_absolute_logit_error",
        "state_sum_max_absolute_error",
        "normalized_state_max_absolute_error",
        "source_term_sum_max_absolute_error",
        "product_closure_max_absolute_error",
        "output_closure_max_absolute_error",
        "propagated_recipient_MLP8_max_absolute_error",
        "propagated_source_MLP8_max_absolute_error",
        "gauge_invariance_max_absolute_error",
        "parent_head_endpoint_max_absolute_error",
        "same_batch_native_noop_endpoint_max_absolute_error",
        "installed_head_max_absolute_error",
    )}


def test_plan_changes_only_engineering_and_is_v1_result_bound():
    plan = run.compile_plan()
    assert plan["conditions"] == list(v1.CONDITIONS)
    assert plan["bars"] == v1.BARS
    assert set(plan["predictions"]) == set(v1.score(
        _evidence(), _exact())["predictions"])
    assert plan["subject_position"] == 8
    assert plan["invalid_v1_result_sha256"] == run.V1_RESULT_SHA256
    assert plan["price"] == {
        "model_forwards": 4, "example_evaluations": 480,
        "causal_interventions": 192, "backwards": 0, "parameter_updates": 0,
    }


def test_no_model_dry_run_is_exact_plan():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == run.compile_plan()


def test_preflight_fails_closed_on_correction_or_v1_hash(monkeypatch):
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", "0" * 64)
    with pytest.raises(run.MLP8PolarizedResponseV2Error, match="correction receipt changed"):
        run.validate_preflight()
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", run._sha256(run.PRIOR_ART))
    monkeypatch.setattr(run, "V1_RESULT_SHA256", "0" * 64)
    with pytest.raises(run.MLP8PolarizedResponseV2Error, match="invalid v1 result changed"):
        run.validate_preflight()


def test_product_remainder_closes_native_float32_and_gauge_is_float64():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Left = torch.nn.Linear(5, 31, bias=False)
            self.Right = torch.nn.Linear(5, 31, bias=False)
            self.Down = torch.nn.Linear(31, 5, bias=False)
            self.Down_bias = torch.nn.Parameter(torch.randn(5))

    torch.manual_seed(81)
    mlp = MLP()
    recipient = 100 * torch.randn(3, 9, 5)
    source = 100 * torch.randn(3, 9, 5)
    products, diagnostics = run._polarized_products(
        mlp, recipient, source, torch, F)
    response = (products["cross"] - products["recipient"]) \
        + (products["quadratic"] - products["recipient"])
    # The registered closure uses the corrected response-term expression.
    assert diagnostics["product_closure_max_absolute_error"] <= 5e-5
    assert diagnostics["gauge_invariance_max_absolute_error"] < 1e-10
    assert torch.allclose(products["full"] - products["recipient"], response,
                          atol=2e-2, rtol=2e-6)


def test_output_remainder_closes_and_down_bias_occurs_once():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Down = torch.nn.Linear(17, 5, bias=False)
            self.Down_bias = torch.nn.Parameter(torch.randn(5))

    class Block:
        def __init__(self, value):
            self.lambdas = [torch.tensor(value), torch.tensor(0.0)]

    class Model:
        def __init__(self):
            self.transformer = type("T", (), {})()
            self.transformer.h = [Block(1.0) for _ in range(18)]
            self.transformer.h[9] = Block(.9)
            self.transformer.h[10] = Block(.8)
            self.transformer.h[11] = Block(.7)

    torch.manual_seed(82)
    mlp, model = MLP(), Model()
    base = torch.randn(2, 9, 17)
    cross_response = torch.randn(2, 9, 17)
    quad_response = torch.randn(2, 9, 17)
    products = {
        "recipient": base,
        "cross": base + cross_response,
        "quadratic": base + quad_response,
        "full": (base + cross_response) + quad_response,
    }
    recipient_slot = torch.randn(2, 9, 5)
    slots, outputs, diagnostics = run._propagated_slots(
        model, mlp, products, recipient_slot, F)
    assert diagnostics["output_closure_max_absolute_error"] <= 5e-5
    assert torch.allclose(outputs["recipient"],
                          F.linear(base.double(), mlp.Down.weight.double())
                          + mlp.Down_bias.double())
    for component in run.COMPONENTS:
        expected = outputs[component].float()
        for scale in (.9, .8, .7):
            expected = torch.tensor(scale) * expected
        assert torch.allclose(slots[component][:, 8], expected[:, 8])
        assert torch.equal(slots[component][:, :8], recipient_slot[:, :8])


def test_sequential_propagation_is_not_collapsed_product():
    torch = pytest.importorskip("torch")

    class Block:
        def __init__(self, value):
            self.lambdas = [torch.tensor(value, dtype=torch.float32), torch.tensor(0.0)]

    class Model:
        def __init__(self):
            self.transformer = type("T", (), {})()
            self.transformer.h = [Block(1.0) for _ in range(18)]
            for layer, value in zip((9, 10, 11), (.99123, .98765, .97654)):
                self.transformer.h[layer] = Block(value)

    x = torch.tensor([1.2345678e4], dtype=torch.float32)
    expected = x
    for layer in (9, 10, 11):
        expected = Model().transformer.h[layer].lambdas[0] * expected
    observed = run._sequentially_propagate(Model(), x)
    assert torch.equal(observed, expected)


def _evidence():
    values = {}
    for background, base in (("standalone", 0.0), ("conditional", .2)):
        values[f"{background}_recipient"] = base
        values[f"{background}_cross"] = base + .8
        values[f"{background}_quadratic"] = base + .1
        values[f"{background}_full"] = base + 1.0
    values.update({"lexical_recipient": 0.0, "lexical_cross": .01,
                   "lexical_quadratic": .01, "lexical_full": .02})
    return [{
        "row_id": row["row_id"],
        "cell_id": f"{row['direction_id']}__{row['template_id']}",
        "condition": condition,
        "target_margin_improvement": effect,
        "target_CE_improvement": effect,
    } for row in run.build_rows() for condition, effect in values.items()]


def test_scientific_score_is_exact_v1_delegation():
    assert run.score(_evidence(), _exact()) == v1.score(_evidence(), _exact())
