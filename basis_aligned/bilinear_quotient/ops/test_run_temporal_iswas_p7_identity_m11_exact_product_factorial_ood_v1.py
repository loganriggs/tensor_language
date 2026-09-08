import json
import subprocess
import sys

import torch

import run_temporal_iswas_p7_identity_m11_exact_product_factorial_ood_v1 as target


def test_unbound_runner_is_model_free_and_exactly_priced():
    completed = subprocess.run([sys.executable, target.__file__], check=True,
        capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["status"] == "awaiting_binding"
    assert payload["authority_ok"] is True
    assert payload["gpu_accessed"] is payload["model_loaded"] is False
    assert payload["queue_touched"] is False
    assert payload["price"]["model_forwards"] == 11


def test_three_product_factors_close_exactly_in_float32():
    generator = torch.Generator().manual_seed(7)
    tensors = [torch.randn(3, 4, 11, generator=generator) for _ in range(4)]
    live_left, live_right, writer_left, writer_right = tensors
    factors = target.exact_product_factors(*tensors)
    predicted = target.compose_hidden(live_left, live_right, factors, target.FACTORS)
    exact = writer_left.float() * writer_right.float()
    assert torch.allclose(predicted, exact, rtol=2e-6, atol=2e-6)
    assert target.closure_relative_error(*tensors) < 2e-7


def test_factor_subsets_are_distinct_and_invalid_sets_fail():
    one = torch.ones(1, 2, 3)
    two = 2 * one
    four = 4 * one
    seven = 7 * one
    factors = target.exact_product_factors(one, two, four, seven)
    values = [target.compose_hidden(one, two, factors, subset)
              for subset in ((), ("left",), ("right",), ("interaction",), target.FACTORS)]
    assert len({tuple(value.flatten().tolist()) for value in values}) == 5
    for subset in (("left", "left"), ("unknown",)):
        try:
            target.compose_hidden(one, two, factors, subset)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid factor subset accepted")
