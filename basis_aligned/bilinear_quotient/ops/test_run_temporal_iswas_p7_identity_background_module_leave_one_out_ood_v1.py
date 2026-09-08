import json
import os
from pathlib import Path
import subprocess
import sys

import torch

import run_temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1 as target


def test_dryrun_is_authority_bound_model_free_and_exact_price():
    completed = subprocess.run([sys.executable, str(Path(target.__file__))], check=True,
        capture_output=True, text=True,
        env=dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1"))
    payload = json.loads(completed.stdout)
    assert payload["authority_ok"] is True
    assert payload["population"] == "ood"
    assert payload["P7"] == list(target.P7)
    assert len(payload["arms"]) == 8
    assert payload["price"]["model_forwards"] == 10
    assert payload["gpu_accessed"] is False
    assert payload["model_loaded"] is False
    assert payload["queue_touched"] is False


def test_prior_and_parent_authorities_are_exact():
    prior = json.loads(target.PRIOR.read_text())
    assert prior["price"] == target.PRICE
    assert {name: target.sha(path) for name, path in target.FILES.items()} == target.EXPECTED
    assert len(target.PREDICTION_KEYS) == 5


def test_subset_arm_leaves_omitted_module_live_and_cleans_hooks(monkeypatch):
    events = []

    class Handle:
        def __init__(self, hooks, hook):
            self.hooks, self.hook = hooks, hook
        def remove(self):
            self.hooks.remove(self.hook)

    class Module:
        def __init__(self, name):
            self.name, self.pre_hooks, self.hooks = name, [], []
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
                events.append(self.name)
                changed = hook(self, (), value)
                if changed is not None:
                    value = changed
            return value

    entry = Module("entry")
    modules = {name: Module(name) for name in target.parent.factorial.atlas.MODULES}
    block = type("Block", (), {})()
    block.register_forward_pre_hook = entry.register_forward_pre_hook
    model = type("Model", (), {})()
    model.transformer = type("Transformer", (), {})()
    model.transformer.h = [None] * 18
    model.transformer.h[10] = block
    backend = type("Backend", (), {"model": model})()
    monkeypatch.setattr(target.parent.factorial.atlas, "module_targets", lambda _model: modules)

    omitted = "M12"
    subset = tuple(name for name in target.P7 if name != omitted)

    class Parent:
        @staticmethod
        def _forward(_backend, _tokens):
            value = entry.pre((torch.zeros(2, 3, 1), None, torch.zeros(2, 3, 1)))[0]
            for name, module in modules.items():
                if name == omitted:
                    events.append(f"live:{name}")
                value = module.fire(value)
            return value, {}

    monkeypatch.setattr(target.parent.factorial.atlas.mediation, "parent", Parent)
    writes = {name: torch.full((2, 3, 1), float(index + 2))
              for index, name in enumerate(target.parent.factorial.atlas.MODULES)}
    output, entry_calls, calls = target.run_subset_arm(
        backend, torch.zeros(2, 3), torch.ones(2, 3, 1), writes,
        [[0, 1], [0, 1, 2]], subset)
    assert entry_calls == 1 and set(calls) == set(subset) and set(calls.values()) == {1}
    assert f"live:{omitted}" in events and omitted not in calls
    physical = [name for name in target.parent.factorial.atlas.MODULES if name in subset]
    assert torch.equal(output[0, :2], writes[physical[-1]][0, :2])
    assert not entry.pre_hooks and all(not module.hooks for module in modules.values())


def test_full_qualification_uses_joint_bars():
    assert target.full_qualified({"signed_recovery": 1.0, "cosine": .99,
        "relative_residual": .1, "direction_agreement": 1.0})
    assert not target.full_qualified({"signed_recovery": .5, "cosine": .99,
        "relative_residual": .1, "direction_agreement": 1.0})
