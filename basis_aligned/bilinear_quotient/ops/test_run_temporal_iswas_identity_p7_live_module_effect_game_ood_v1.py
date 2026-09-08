import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_temporal_iswas_identity_p7_live_module_effect_game_ood_v1 as target


def test_dryrun_is_authority_bound_model_free_and_exact_lattice_price():
    completed = subprocess.run([sys.executable, str(Path(target.__file__))], check=True,
        capture_output=True, text=True,
        env=dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1"))
    payload = json.loads(completed.stdout)
    assert payload["authority_ok"] is True
    assert payload["P7"] == list(target.P7)
    assert payload["n_masks"] == 128
    assert payload["source_states"] == ["absent", "present"]
    assert payload["price"] == target.PRICE
    assert payload["gpu_accessed"] is False and payload["model_loaded"] is False


def test_vector_mobius_shapley_and_pair_index_recover_planted_game():
    weights = np.arange(1, 8, dtype=np.float64)[:, None]
    values = np.zeros((target.N_MASKS, 3), dtype=np.float64)
    for mask in range(target.N_MASKS):
        values[mask] = sum((weights[bit] for bit in range(7) if mask & (1 << bit)),
                           start=np.zeros(1))[0]
        if mask & 3 == 3:
            values[mask] += np.array([4.0, -2.0, 1.0])
        if mask & 7 == 7:
            values[mask] += np.array([3.0, 6.0, -3.0])
    coefficients = target.mobius(values)
    assert np.allclose(target.reconstruct(coefficients), values)
    phi = target.shapley(coefficients)
    assert np.allclose(phi.sum(0), values[-1] - values[0])
    interactions = target.pair_interactions(coefficients)
    # I_01 = d_01 + d_012 / 2; higher-order dividends are intentionally aggregated.
    assert np.allclose(interactions[(target.P7[0], target.P7[1])],
                       np.array([5.5, 1.0, -0.5]))
    assert np.allclose(phi[0], weights[0] + np.array([3.0, 1.0, -0.5]))


def test_game_arm_clamps_complement_leaves_subset_live_and_cleans_hooks(monkeypatch):
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
            if not self.hooks:
                events.append(f"live:{self.name}")
            for hook in tuple(self.hooks):
                events.append(f"clamped:{self.name}")
                changed = hook(self, (), value)
                if changed is not None:
                    value = changed
            return value

    entry = Module("entry")
    modules = {name: Module(name) for name in target.necessity.parent.factorial.atlas.MODULES}
    block = type("Block", (), {})()
    block.register_forward_pre_hook = entry.register_forward_pre_hook
    model = type("Model", (), {})()
    model.transformer = type("Transformer", (), {})()
    model.transformer.h = [None] * 18
    model.transformer.h[10] = block
    backend = type("Backend", (), {"model": model})()
    monkeypatch.setattr(target.necessity.parent.factorial.atlas,
                        "module_targets", lambda _model: modules)

    class Parent:
        @staticmethod
        def _forward(_backend, _tokens):
            value = entry.pre((torch.zeros(2, 3, 1), None, torch.zeros(2, 3, 1)))[0]
            for module in modules.values():
                value = module.fire(value)
            return value, {}

    monkeypatch.setattr(target.necessity.parent.factorial.atlas.mediation, "parent", Parent)
    native = {name: torch.full((2, 3, 1), float(index + 2))
              for index, name in enumerate(target.necessity.parent.factorial.atlas.MODULES)}
    enabled = ("A11", "M12")
    _output, calls = target.run_game_arm(
        backend, torch.zeros(2, 3), torch.ones(2, 3, 1), native,
        [[0, 1], [0, 1, 2]], enabled)
    assert calls["entry"] == 1
    assert set(calls["modules"]) == set(target.P7) - set(enabled)
    assert set(calls["modules"].values()) == {1}
    assert all(f"live:{name}" in events for name in enabled)
    assert all(f"clamped:{name}" in events for name in set(target.P7) - set(enabled))
    assert not entry.pre_hooks and all(not module.hooks for module in modules.values())


def test_masks_cover_each_fixed_subset_once_and_reject_invalid_masks():
    assert len({target.members(mask) for mask in range(target.N_MASKS)}) == 128
    assert target.members(0) == () and target.members(127) == target.P7
    for bad in (-1, 128, 1.5):
        try:
            target.members(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid mask {bad}")
