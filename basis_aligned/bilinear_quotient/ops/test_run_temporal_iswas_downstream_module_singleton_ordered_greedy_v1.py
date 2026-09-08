import json
import os
from pathlib import Path
import subprocess
import sys

import torch

import run_temporal_iswas_downstream_module_singleton_ordered_greedy_v1 as target


def test_dryrun_is_authority_bound_and_model_free():
    completed = subprocess.run([sys.executable, str(Path(target.__file__))], check=True,
        capture_output=True, text=True,
        env=dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1"))
    payload = json.loads(completed.stdout)
    assert payload["authority_ok"] is True
    assert payload["order"] == list(target.ORDER)
    assert payload["gpu_accessed"] is False
    assert payload["model_loaded"] is False
    assert payload["queue_touched"] is False
    assert payload["price"]["model_forwards"] == 19


def test_order_is_exact_singleton_fit_ranking():
    result = json.loads(target.ATLAS_RESULT.read_text())
    fit = {row["module"]: row["signed_recovery"]
           for row in result["reports"] if row["phase"] == "FIT"}
    observed = tuple(sorted(fit, key=lambda name: (-fit[name], name)))
    assert observed == target.ORDER


def test_fit_and_holdout_qualification_are_distinct():
    row = {"signed_recovery": .55, "cosine": .92,
           "relative_residual": .6, "direction_agreement": .95}
    assert target.qualified(row, "FIT")
    assert target.qualified(row, "HOLDOUT")
    row["signed_recovery"] = .45
    assert not target.qualified(row, "FIT")
    assert target.qualified(row, "HOLDOUT")


def test_prediction_inventory_and_price_match_prior():
    prior = json.loads(target.PRIOR.read_text())
    assert len(target.PREDICTION_KEYS) == 5
    assert prior["price"] == target.PRICE
    assert prior["locked_module_order"] == list(target.ORDER)


def test_expected_authorities_match_bytes():
    observed = {"prior": target.sha(target.PRIOR),
                "atlas_result": target.sha(target.ATLAS_RESULT),
                "atlas_runner": target.sha(target.ATLAS_RUNNER),
                "greedy_result": target.sha(target.GREEDY_RESULT)}
    assert observed == target.EXPECTED


def test_patch_prefix_executes_every_selected_hook_once(monkeypatch):
    class Handle:
        def __init__(self, module, hook):
            self.module, self.hook = module, hook
        def remove(self):
            self.module.hooks.remove(self.hook)

    class Module:
        def __init__(self):
            self.hooks = []
        def register_forward_hook(self, hook):
            self.hooks.append(hook)
            return Handle(self, hook)
        def fire(self, value):
            for hook in tuple(self.hooks):
                changed = hook(self, (), value)
                if changed is not None:
                    value = changed
            return value

    modules = {name: Module() for name in target.ORDER[:2]}
    monkeypatch.setattr(target.atlas, "module_targets", lambda _model: modules)

    class Parent:
        @staticmethod
        def _forward(_backend, _tokens):
            value = torch.zeros(2, 3, 1)
            for module in modules.values():
                value = module.fire(value)
            return value, {}

    monkeypatch.setattr(target.atlas.mediation, "parent", Parent)
    backend = type("Backend", (), {"model": object()})()
    replacements = {
        target.ORDER[0]: torch.ones(2, 3, 1),
        target.ORDER[1]: torch.full((2, 3, 1), 2.0),
    }
    logits, calls = target.patch_prefix(
        backend, torch.zeros(2, 3), target.ORDER[:2], replacements, [[0, 1], [0, 1, 2]])
    assert calls == {target.ORDER[0]: 1, target.ORDER[1]: 1}
    assert torch.equal(logits[0, :2], replacements[target.ORDER[1]][0, :2])
    assert torch.equal(logits[1], replacements[target.ORDER[1]][1])
    assert all(not module.hooks for module in modules.values())
