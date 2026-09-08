import pytest
import torch

import final_residual_intervention as target


class Handle:
    def __init__(self, hooks, hook):
        self.hooks, self.hook = hooks, hook

    def remove(self):
        self.hooks.remove(self.hook)


class Block:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return Handle(self.hooks, hook)

    def run(self, state):
        output = (state, "first-value-state")
        for hook in tuple(self.hooks):
            changed = hook(self, (), output)
            if changed is not None:
                output = changed
        return output


def model_with_final(block):
    model = type("Model", (), {})()
    model.transformer = type("Transformer", (), {})()
    model.transformer.h = [block]
    return model


def test_execute_captures_preintervention_state_and_replaces_selected_positions():
    block = Block()
    model = model_with_final(block)
    state = torch.zeros(2, 3, 2)
    replacement = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    (changed, first), captured, calls = target.execute(
        model, lambda: block.run(state), replacement=replacement,
        position_rows=[[0, 1], [2]])
    assert calls == 1
    assert torch.equal(captured, state)
    assert torch.equal(changed[0, :2], replacement[0, :2])
    assert torch.equal(changed[1, 2], replacement[1, 2])
    assert torch.equal(changed[0, 2], state[0, 2])
    assert first == "first-value-state"
    assert not block.hooks


def test_execute_capture_only_and_shape_guards():
    block = Block()
    model = model_with_final(block)
    state = torch.ones(2, 3, 2)
    output, captured, calls = target.execute(model, lambda: block.run(state))
    assert calls == 1 and torch.equal(output[0], state) and torch.equal(captured, state)
    assert not block.hooks
    with pytest.raises(target.FinalResidualInterventionError):
        target.replace_positions(state, torch.zeros(2, 2, 2), [[0], [0]])
    with pytest.raises(target.FinalResidualInterventionError):
        target.execute(model, lambda: block.run(state), replacement=state)
