import json
import os
from pathlib import Path
import subprocess
import sys

import torch

import run_temporal_iswas_p7_residual_two_stream_ood_composition_v1 as target


def test_dryrun_is_authority_bound_model_free_and_ood_only():
    completed = subprocess.run([sys.executable, str(Path(target.__file__))], check=True,
        capture_output=True, text=True,
        env=dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1"))
    payload = json.loads(completed.stdout)
    assert payload["authority_ok"] is True
    assert payload["population"] == "ood"
    assert payload["P7"] == list(target.P7)
    assert payload["gpu_accessed"] is False
    assert payload["model_loaded"] is False
    assert payload["queue_touched"] is False
    assert payload["price"]["model_forwards"] == 6


def test_prior_inventory_and_authorities_are_exact():
    prior = json.loads(target.PRIOR.read_text())
    assert tuple(prior["fixed_response_prefix"]["modules"]) == target.P7
    assert prior["price"] == target.PRICE
    assert len(target.PREDICTION_KEYS) == 5
    assert {name: target.sha(path) for name, path in target.FILES.items()} == target.EXPECTED


def test_p7_arm_patches_entry_then_only_frozen_modules(monkeypatch):
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
    modules = {name: Module(name) for name in target.factorial.atlas.MODULES}
    block = type("Block", (), {})()
    block.register_forward_pre_hook = entry.register_forward_pre_hook
    model = type("Model", (), {})()
    model.transformer = type("Transformer", (), {})()
    model.transformer.h = [None] * 18
    model.transformer.h[10] = block
    backend = type("Backend", (), {"model": model})()
    monkeypatch.setattr(target.factorial.atlas, "module_targets", lambda _model: modules)

    class Parent:
        @staticmethod
        def _forward(_backend, _tokens):
            value = torch.zeros(2, 3, 1)
            value = entry.pre((value, None, value))[0]
            events.append("entry")
            for name, module in modules.items():
                value = module.fire(value)
                if name not in target.P7:
                    events.append(f"live:{name}")
            return value, {}

    monkeypatch.setattr(target.factorial.atlas.mediation, "parent", Parent)
    replacement = torch.ones(2, 3, 1)
    writes = {name: torch.full((2, 3, 1), float(index + 2))
              for index, name in enumerate(target.factorial.atlas.MODULES)}
    output, entry_calls, module_calls = target.run_p7_arm(
        backend, torch.zeros(2, 3), replacement, writes, [[0, 1], [0, 1, 2]])
    assert entry_calls == 1 and set(module_calls.values()) == {1}
    physical = [name for name in target.factorial.atlas.MODULES if name in target.P7]
    assert [event for event in events if event in target.P7] == physical
    assert torch.equal(output[0, :2], writes[physical[-1]][0, :2])
    assert torch.equal(output[1], writes[physical[-1]][1])
    assert not entry.pre_hooks and all(not module.hooks for module in modules.values())


def test_qualification_keeps_p7_and_joint_bars_distinct():
    p7 = {"signed_recovery": .5, "cosine": .9,
          "relative_residual": .6, "direction_agreement": .9}
    joint = {"signed_recovery": 1.0, "cosine": .99,
             "relative_residual": .1, "direction_agreement": 1.0}
    assert target.qualified(p7, "p7")
    assert not target.qualified(p7, "joint")
    assert target.qualified(joint, "joint")
