import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_temporal_iswas_selected_writer_residual_skip_module_response_factorial_v1 as target


def test_dryrun_is_authority_bound_and_model_free():
    completed = subprocess.run([sys.executable, str(Path(target.__file__))], check=True,
        capture_output=True, text=True,
        env=dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1"))
    payload = json.loads(completed.stdout)
    assert payload["authority_ok"] is True
    assert payload["arms"] == list(target.ARMS)
    assert payload["modules"] == list(target.atlas.MODULES)
    assert payload["gpu_accessed"] is False
    assert payload["model_loaded"] is False
    assert payload["queue_touched"] is False
    assert payload["price"]["model_forwards"] == 6


def test_factorial_inventory_predictions_and_price_match_prior():
    prior = json.loads(target.PRIOR.read_text())
    assert target.ARMS == ("R1M0", "R0M1", "R1M1")
    assert len(target.PREDICTION_KEYS) == 5
    assert prior["price"] == target.PRICE
    assert prior["locked_writer"]["heads"] == ["L07H07", "L09H04"]


def test_joint_and_carrier_qualification_are_distinct():
    exact = {"signed_recovery": 1.0, "cosine": 1.0,
             "relative_residual": 0.0, "direction_agreement": 1.0}
    majority = {"signed_recovery": .6, "cosine": .95,
                "relative_residual": .5, "direction_agreement": .95}
    assert target.joint_qualified(exact)
    assert not target.joint_qualified(majority)
    assert target.carrier_qualified(exact)
    assert target.carrier_qualified(majority)


def test_additivity_error_uses_joint_effect_as_denominator():
    residual = np.asarray([1.0, 0.0])
    modules = np.asarray([0.0, 2.0])
    joint = residual + modules
    assert target.relative_additivity_error(joint, residual, modules) == 0.0
    assert np.isclose(target.relative_additivity_error(
        joint + np.asarray([0.0, 1.0]), residual, modules), 1.0 / np.sqrt(10.0))


def test_expected_authorities_match_bytes():
    observed = {"prior": target.sha(target.PRIOR),
                "route_result": target.sha(target.ROUTE_RESULT),
                "atlas_result": target.sha(target.ATLAS_RESULT),
                "atlas_runner": target.sha(target.ATLAS_RUNNER),
                "greedy_result": target.sha(target.GREEDY_RESULT)}
    assert observed == target.EXPECTED


def test_factorial_arm_patches_state_then_all_modules_once(monkeypatch):
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
        def fire_pre(self, arguments):
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

    state = Module("state")
    modules = {name: Module(name) for name in target.atlas.MODULES}
    block = type("Block", (), {})()
    block.register_forward_pre_hook = state.register_forward_pre_hook
    model = type("Model", (), {})()
    model.transformer = type("Transformer", (), {})()
    model.transformer.h = [None] * 18
    model.transformer.h[10] = block
    backend = type("Backend", (), {"model": model})()
    monkeypatch.setattr(target.atlas, "module_targets", lambda _model: modules)

    class Parent:
        @staticmethod
        def _forward(_backend, _tokens):
            value = torch.zeros(2, 3, 1)
            arguments = state.fire_pre((value, None, value))
            value = arguments[0]
            events.append("state")
            for module in modules.values():
                value = module.fire(value)
            return value, {}

    monkeypatch.setattr(target.atlas.mediation, "parent", Parent)
    residual = torch.ones(2, 3, 1)
    replacements = {name: torch.full((2, 3, 1), float(index + 2))
                    for index, name in enumerate(target.atlas.MODULES)}
    logits, state_calls, module_calls = target.run_factorial_arm(
        backend, torch.zeros(2, 3), residual, replacements, [[0, 1], [0, 1, 2]])
    assert state_calls == 1
    assert set(module_calls.values()) == {1}
    assert events == ["state", *target.atlas.MODULES]
    assert torch.equal(logits[0, :2], replacements[target.atlas.MODULES[-1]][0, :2])
    assert torch.equal(logits[1], replacements[target.atlas.MODULES[-1]][1])
    assert not state.pre_hooks
    assert all(not module.hooks for module in modules.values())
