from __future__ import annotations

import numpy as np
import pytest
import torch

import circuit_intervention_primitives as run


def _call_hook(runtime: run.ProductInterventionRuntime, layer: int, product: torch.Tensor):
    result = runtime.hook(layer)(None, (product,))
    return product if result is None else result[0]


def test_reset_changes_only_registered_final_token_and_records_live_value():
    product = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    base = {
        ("a", "base"): torch.tensor([-1.0, -2.0, -3.0]),
        ("b", "base"): torch.tensor([5.0, 6.0, 7.0]),
    }
    plan = run.ProductPlan("reset", (run.ProductAction(
        4, "reset", base_key="base", capture_key="live",
        events_before=("enter",), events_after=("applied",),
    ),))
    runtime = run.ProductInterventionRuntime(
        plan, row_ids=("a", "b"), positions=(1, 3), base_vectors=base,
    )
    changed = _call_hook(runtime, 4, product)
    expected = product.clone()
    expected[0, 1] = base[("a", "base")]
    expected[1, 3] = base[("b", "base")]
    assert torch.equal(changed, expected)
    assert torch.equal(changed[0, 0], product[0, 0])
    assert torch.equal(changed[0, 2:], product[0, 2:])
    assert torch.equal(runtime.captures[("a", "live")], product[0, 1])
    assert runtime.provenance() == (("enter", "applied"), ("enter", "applied"))
    assert runtime.endpoint_error == {"a": 0.0, "b": 0.0}


def test_rescue_is_exact_noop_but_executes_the_declared_hook():
    product = torch.randn(2, 3, 5, generator=torch.Generator().manual_seed(17))
    base = {(row, "base"): torch.randn(5) for row in ("a", "b")}
    runtime = run.ProductInterventionRuntime(
        run.ProductPlan("rescue", (run.ProductAction(
            9, "rescue", base_key="base", events_after=("rescued",),
        ),)),
        row_ids=("a", "b"), positions=(2, 1), base_vectors=base,
    )
    torch.testing.assert_close(_call_hook(runtime, 9, product), product, atol=2e-7, rtol=0)
    assert runtime.provenance() == (("rescued",), ("rescued",))


def test_depth_order_and_exact_five_event_provenance_are_fail_closed():
    plan = run.ProductPlan("joint", (
        run.ProductAction(15, "reset", base_key="z15",
                          events_before=("mlp15_product_enter",),
                          events_after=("mlp15_reset_applied",)),
        run.ProductAction(17, "reset", base_key="z17", capture_key="z17_live",
                          events_before=("mlp17_product_enter_after_mlp15",),
                          capture_event="mlp17_live_product_captured",
                          events_after=("mlp17_reset_applied",)),
    ))
    runtime = run.ProductInterventionRuntime(
        plan, row_ids=("r",), positions=(0,),
        base_vectors={("r", "z15"): torch.zeros(2), ("r", "z17"): torch.ones(2)},
    )
    with pytest.raises(run.CircuitInterventionError, match="out of depth order"):
        _call_hook(runtime, 17, torch.randn(1, 1, 2))
    _call_hook(runtime, 15, torch.randn(1, 1, 2))
    z17 = torch.randn(1, 1, 2)
    _call_hook(runtime, 17, z17)
    assert runtime.provenance()[0] == (
        "mlp15_product_enter", "mlp15_reset_applied",
        "mlp17_product_enter_after_mlp15", "mlp17_live_product_captured",
        "mlp17_reset_applied",
    )
    assert torch.equal(runtime.captures[("r", "z17_live")], z17[0, 0])
    with pytest.raises(run.CircuitInterventionError, match="repeatedly"):
        _call_hook(runtime, 17, z17)


def test_plan_rejects_duplicate_or_non_depth_ordered_sites():
    with pytest.raises(run.CircuitInterventionError, match="increasing"):
        run.ProductPlan("bad", (
            run.ProductAction(4, "observe"), run.ProductAction(3, "observe"),
        ))
    with pytest.raises(run.CircuitInterventionError, match="globally unique"):
        run.ProductPlan("bad-events", (
            run.ProductAction(3, "observe", events_before=("x",)),
            run.ProductAction(4, "observe", events_before=("x",)),
        ))


def test_exact_bilinear_terms_close_and_detect_wrong_base():
    generator = torch.Generator().manual_seed(1517)
    left = torch.randn(11, 5, generator=generator)
    right = torch.randn(11, 5, generator=generator)
    x_base = torch.randn(5, generator=generator)
    x_changed = torch.randn(5, generator=generator)
    z_base = (left @ x_base) * (right @ x_base)
    z_changed = (left @ x_changed) * (right @ x_changed)
    assert run.bilinear_closure_max_abs(
        left, right, x_base, x_changed, z_base, z_changed,
    ) < 3e-6
    assert run.bilinear_closure_max_abs(
        left, right, x_base + 0.2, x_changed, z_base, z_changed,
    ) > 1e-2


def test_full_vocab_and_signed_metrics_preserve_causal_sign():
    summary = run.full_vocab_difference(
        np.array([1.0, 4.0, -1.0]), np.array([0.0, 2.0, -1.0]),
    )
    assert summary["max_abs"] == 2.0
    assert summary["rms"] == pytest.approx(np.sqrt(5 / 3))
    restorative = run.signed_response_metrics([1.0, 2.0], [-0.5, -1.0])
    compensatory = run.signed_response_metrics([1.0, 2.0], [0.5, 1.0])
    assert restorative["beta"] == pytest.approx(0.5)
    assert compensatory["beta"] == pytest.approx(-0.5)
    assert restorative["q"] == compensatory["q"] == pytest.approx(0.5)
