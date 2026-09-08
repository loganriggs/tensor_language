import json
import subprocess
import sys

import torch

import run_temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1 as target
import run_temporal_iswas_p7_residual_two_stream_ood_composition_v1 as two


def test_unbound_runner_exits_model_free():
    completed = subprocess.run([sys.executable, target.__file__], check=True,
        capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["status"] == "awaiting_binding"
    assert payload["authority_ok"] is True
    assert payload["gpu_accessed"] is payload["model_loaded"] is False
    assert payload["queue_touched"] is False
    assert payload["price"]["model_forwards"] == 22


def test_head_slice_replacement_is_exact_and_scoped():
    native = torch.zeros(2, 3, 18)
    writer = torch.arange(108, dtype=torch.float32).reshape(2, 3, 18)
    changed = target.replace_head_slices(native, writer, (3, 8), [[0, 2], [1]], n_heads=9)
    for row, positions in enumerate(([0, 2], [1])):
        for position in range(3):
            for head in range(9):
                sl = slice(2 * head, 2 * head + 2)
                expected = writer if position in positions and head in (3, 8) else native
                assert torch.equal(changed[row, position, sl], expected[row, position, sl])


def test_invalid_head_geometry_fails_closed():
    x = torch.zeros(1, 1, 18)
    for heads in ((9,), (3, 3)):
        try:
            target.replace_head_slices(x, x, heads, [[0]], n_heads=9)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid head set accepted")


def test_head_arm_orders_entry_head_and_six_mlp_clamps_and_cleans(monkeypatch):
    class Handle:
        def __init__(self, hooks, hook):
            self.hooks, self.hook = hooks, hook
        def remove(self):
            self.hooks.remove(self.hook)

    class Module:
        def __init__(self):
            self.pre_hooks, self.hooks = [], []
        def register_forward_pre_hook(self, hook):
            self.pre_hooks.append(hook)
            return Handle(self.pre_hooks, hook)
        def register_forward_hook(self, hook):
            self.hooks.append(hook)
            return Handle(self.hooks, hook)
        def pre(self, arguments):
            for hook in tuple(self.pre_hooks):
                changed = hook(self, arguments)
                if changed is not None:
                    arguments = changed
            return arguments
        def fire(self, value):
            for hook in tuple(self.hooks):
                changed = hook(self, (), value)
                if changed is not None:
                    value = changed
            return value

    block10, cproj = Module(), Module()
    modules = {name: Module() for name in two.factorial.atlas.MODULES}
    modules["A11"] = cproj
    blocks = [None] * 18
    blocks[10] = block10
    blocks[11] = type("Block", (), {"attn": type("Attn", (), {"c_proj": cproj})()})()
    model = type("Model", (), {"transformer": type("Transformer", (), {"h": blocks})()})()
    backend = type("Backend", (), {"model": model})()
    monkeypatch.setattr(two.factorial.atlas, "module_targets", lambda _model: modules)

    class Parent:
        @staticmethod
        def _forward(_backend, _tokens):
            x = block10.pre((torch.zeros(2, 3, 18), None, torch.zeros(2, 3, 18)))[0]
            x = cproj.pre((x,))[0]
            x = cproj.fire(x)
            for name in two.factorial.atlas.MODULES:
                if name.startswith("M") and name in two.P7:
                    x = modules[name].fire(x)
            return x, {}

    monkeypatch.setattr(two.factorial.atlas.mediation, "parent", Parent)
    writer_heads = torch.arange(108, dtype=torch.float32).reshape(2, 3, 18)
    writer_modules = {name: torch.full((2, 3, 18), float(index + 1))
                      for index, name in enumerate(two.factorial.atlas.MODULES)}
    logits, calls, a11 = target.run_head_arm(two, backend, torch.zeros(2, 3),
        torch.ones(2, 3, 18), writer_heads, writer_modules,
        [[0, 1], [0, 1, 2]], target.HEADS)
    assert calls["entry"] == calls["head"] == calls["output"] == 1
    assert set(calls["modules"]) == set(two.P7) - {"A11"}
    assert set(calls["modules"].values()) == {1}
    assert torch.equal(a11[0, :2], writer_heads[0, :2])
    assert torch.equal(a11[1], writer_heads[1])
    last_mlp = [name for name in two.factorial.atlas.MODULES if name in two.P7 and name.startswith("M")][-1]
    assert torch.equal(logits[0, :2], writer_modules[last_mlp][0, :2])
    assert not block10.pre_hooks and not cproj.pre_hooks and not cproj.hooks
    assert all(not module.hooks for name, module in modules.items() if name != "A11")
