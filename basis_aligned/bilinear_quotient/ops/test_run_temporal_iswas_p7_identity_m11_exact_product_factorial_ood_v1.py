import json
import subprocess
import sys

import torch

import run_temporal_iswas_p7_identity_m11_exact_product_factorial_ood_v1 as target
import run_temporal_iswas_p7_residual_two_stream_ood_composition_v1 as two_stream


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


def test_all_three_arm_uses_live_local_factors_and_cleans_hooks(monkeypatch):
    class Handle:
        def __init__(self, hooks, hook):
            self.hooks, self.hook = hooks, hook
        def remove(self):
            self.hooks.remove(self.hook)

    class Module:
        def __init__(self):
            self.pre_hooks, self.hooks = [], []
        def register_forward_pre_hook(self, hook):
            self.pre_hooks.append(hook); return Handle(self.pre_hooks, hook)
        def register_forward_hook(self, hook):
            self.hooks.append(hook); return Handle(self.hooks, hook)
        def pre(self, arguments):
            for hook in tuple(self.pre_hooks):
                changed = hook(self, arguments)
                if changed is not None: arguments = changed
            return arguments
        def fire(self, value):
            for hook in tuple(self.hooks):
                changed = hook(self, (), value)
                if changed is not None: value = changed
            return value

    block10 = Module()
    mlp = Module(); mlp.Left = Module(); mlp.Right = Module(); mlp.Down = Module()
    modules = {name: Module() for name in two_stream.factorial.atlas.MODULES}
    modules["M11"] = mlp
    blocks = [None] * 18
    blocks[10] = block10
    blocks[11] = type("Block", (), {"mlp": mlp})()
    model = type("Model", (), {"transformer": type("Transformer", (), {"h": blocks})()})()
    backend = type("Backend", (), {"model": model})()
    monkeypatch.setattr(two_stream.factorial.atlas, "module_targets", lambda _model: modules)

    class Parent:
        @staticmethod
        def _forward(_backend, _tokens):
            x = block10.pre((torch.zeros(2, 3, 18), None, torch.zeros(2, 3, 18)))[0]
            for name in two_stream.factorial.atlas.MODULES:
                if name == "M11":
                    left = mlp.Left.fire(x + 1)
                    right = mlp.Right.fire(2 * x + 1)
                    hidden = mlp.Down.pre((left * right,))[0]
                    x = mlp.fire(hidden)
                elif name in two_stream.P7:
                    x = modules[name].fire(x)
            return x, {}

    monkeypatch.setattr(two_stream.factorial.atlas.mediation, "parent", Parent)
    writer_factors = {"left": torch.full((2, 3, 18), 4.0),
                      "right": torch.full((2, 3, 18), 7.0)}
    writer_modules = {name: torch.full((2, 3, 18), float(index + 2))
                      for index, name in enumerate(two_stream.factorial.atlas.MODULES)}
    writer_modules["M11"] = torch.full((2, 3, 18), 28.0)
    logits, calls, m11_output, closure = target.run_factor_arm(
        two_stream, backend, torch.zeros(2, 3), torch.ones(2, 3, 18),
        writer_factors, writer_modules, [[0, 1], [0, 1, 2]], "all_three")
    assert calls["entry"] == calls["left"] == calls["right"] == calls["down"] == calls["m11"] == 1
    assert set(calls["modules"]) == set(two_stream.P7) - {"M11"}
    assert set(calls["modules"].values()) == {1}
    assert closure < 2e-7
    assert torch.allclose(m11_output[0, :2], torch.full((2, 18), 28.0))
    assert torch.allclose(m11_output[1], torch.full((3, 18), 28.0))
    assert logits.shape == (2, 3, 18)
    assert not block10.pre_hooks and not mlp.hooks and not mlp.Down.pre_hooks
    assert not mlp.Left.hooks and not mlp.Right.hooks

    _complete_logits, complete_calls, complete_output, complete_closure = target.run_factor_arm(
        two_stream, backend, torch.zeros(2, 3), torch.ones(2, 3, 18),
        writer_factors, writer_modules, [[0, 1], [0, 1, 2]], "complete_M11")
    _none_logits, none_calls, none_output, none_closure = target.run_factor_arm(
        two_stream, backend, torch.zeros(2, 3), torch.ones(2, 3, 18),
        writer_factors, writer_modules, [[0, 1], [0, 1, 2]], "none")
    assert complete_calls["m11"] == none_calls["m11"] == 1
    assert complete_closure < 2e-7 and none_closure < 2e-7
    assert torch.equal(complete_output[0, :2], writer_modules["M11"][0, :2])
    assert torch.equal(complete_output[1], writer_modules["M11"][1])
    assert torch.allclose(m11_output[0, :2], torch.full((2, 18), 28.0))
    assert not torch.equal(none_output[0, :2], complete_output[0, :2])
    assert not block10.pre_hooks and not mlp.hooks and not mlp.Down.pre_hooks
    assert not mlp.Left.hooks and not mlp.Right.hooks
