from types import SimpleNamespace

import pytest
import torch

import residual_reader_module_atlas as atlas


class Hookable:
    def __init__(self): self.hooks = []
    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))
    def run(self, value):
        for hook in tuple(self.hooks):
            changed = hook(self, (), value)
            if changed is not None: value = changed
        return value


def fake_model(n=18):
    layers = [SimpleNamespace(attn=SimpleNamespace(c_proj=Hookable()), mlp=Hookable()) for _ in range(n)]
    return SimpleNamespace(transformer=SimpleNamespace(h=layers))


def test_capture_and_reciprocal_prefix_patch_are_exact():
    model = fake_model(); value = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    def execute():
        outputs = {}
        for name, target in atlas.module_targets(model, first_layer=12, last_layer=12).items():
            outputs[name] = target.run(value + (1 if name == "A12" else 2))
        return outputs
    result, saved, calls = atlas.capture_complete_modules(model, execute, first_layer=12, last_layer=12)
    assert set(saved) == {"A12", "M12"} and set(calls.values()) == {1}
    replacement = torch.full_like(value, -1)
    patched, patch_calls = atlas.execute_with_module_patch(
        model, execute, label="A12", replacement=replacement, semantic_positions=(0, 1),
        first_layer=12, last_layer=12)
    assert patch_calls == {"A12": 1}
    assert torch.equal(patched["A12"][0, :1], replacement[0, :1])
    assert torch.equal(patched["A12"][1, :2], replacement[1, :2])
    assert torch.equal(patched["M12"], result["M12"])


def test_selection_uses_weaker_reciprocal_effect_and_stable_tie_break():
    reports = {
        "M13": {"transfer": {"signed_projection": .2, "cosine": .9},
                 "reset": {"signed_projection": .3, "cosine": .9}},
        "A12": {"transfer": {"signed_projection": .2, "cosine": .9},
                 "reset": {"signed_projection": .2, "cosine": .9}},
        "M12": {"transfer": {"signed_projection": .4, "cosine": .9},
                 "reset": {"signed_projection": .1, "cosine": .9}},
    }
    assert atlas.select_reciprocal_candidates(
        reports, fit_projection=.15, fit_cosine=.85, maximum=2) == ["A12", "M13"]


def test_invalid_interval_and_position_fail_closed():
    model = fake_model()
    with pytest.raises(atlas.ResidualReaderAtlasError):
        atlas.module_targets(model, first_layer=17, last_layer=18)
    with pytest.raises(atlas.ResidualReaderAtlasError):
        atlas.replace_prefix(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), (2,))
