import pytest
import torch

import module_reader_loss_rescue as intervention


class ToyChain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = torch.nn.Linear(2, 2, bias=False)
        self.M = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.A.weight.copy_(torch.tensor([[1.0, 0.5], [0.0, 2.0]]))
            self.M.weight.copy_(torch.tensor([[2.0, 0.0], [0.5, 1.0]]))

    def forward(self, value):
        return self.M(torch.tanh(self.A(value)))


def captures(model, value):
    return intervention.capture_reader_outputs(
        lambda: model(value), {"A11": model.A, "M11": model.M}
    )


def test_reader_loss_and_same_module_output_rescue_are_directed_and_exact():
    model = ToyChain()
    absent = torch.zeros(2, 3, 2)
    present = torch.tensor([
        [[1.0, -0.5], [0.2, 0.4], [0.7, -0.1]],
        [[-0.3, 0.8], [0.5, 0.6], [-0.2, 0.9]],
    ])
    absent_y, absent_readers, _absent_outputs, _ = captures(model, absent)
    present_y, _present_readers, present_outputs, _ = captures(model, present)
    positions = ((0, 1, 2), (0, 1, 2))

    lost_a, calls = intervention.run_reader_loss_rescue(
        lambda: model(present), {"A11": model.A, "M11": model.M},
        absent_readers, present_outputs, positions, losses=("A11",),
    )
    assert torch.equal(lost_a, absent_y)
    assert calls == {"A11": {"reader_loss": 1, "output_rescue": 0}}

    rescued_a, _ = intervention.run_reader_loss_rescue(
        lambda: model(present), {"A11": model.A, "M11": model.M},
        absent_readers, present_outputs, positions,
        losses=("A11",), rescues=("A11",),
    )
    assert torch.equal(rescued_a, present_y)

    lost_m, _ = intervention.run_reader_loss_rescue(
        lambda: model(present), {"A11": model.A, "M11": model.M},
        absent_readers, present_outputs, positions, losses=("M11",),
    )
    assert torch.equal(lost_m, absent_y)
    rescued_m, _ = intervention.run_reader_loss_rescue(
        lambda: model(present), {"A11": model.A, "M11": model.M},
        absent_readers, present_outputs, positions,
        losses=("M11",), rescues=("M11",),
    )
    assert torch.equal(rescued_m, present_y)


def test_joint_causal_order_and_registered_position_scope():
    model = ToyChain()
    absent = torch.zeros(1, 3, 2)
    present = torch.tensor([[[1.0, 0.5], [0.2, -0.1], [0.7, 0.9]]])
    _absent_y, absent_readers, _absent_outputs, _ = captures(model, absent)
    present_y, _present_readers, present_outputs, _ = captures(model, present)
    joint, calls = intervention.run_reader_loss_rescue(
        lambda: model(present), {"A11": model.A, "M11": model.M},
        absent_readers, present_outputs, ((1,),),
        losses=("A11", "M11"), rescues=("A11", "M11"),
    )
    assert torch.equal(joint, present_y)
    assert calls == {
        "A11": {"reader_loss": 1, "output_rescue": 1},
        "M11": {"reader_loss": 1, "output_rescue": 1},
    }
    # Hooks are gone: an ordinary forward is unchanged.
    assert torch.equal(model(present), present_y)


def test_invalid_geometry_and_unmatched_rescue_fail_closed():
    tensor = torch.zeros(1, 2, 3)
    with pytest.raises(intervention.ReaderLossRescueError):
        intervention.replace_positions(tensor, torch.zeros(1, 2, 4), ((0,),))
    with pytest.raises(intervention.ReaderLossRescueError):
        intervention.replace_positions(tensor, tensor, ((2,),))
    model = ToyChain()
    with pytest.raises(intervention.ReaderLossRescueError):
        intervention.run_reader_loss_rescue(
            lambda: model(torch.zeros(1, 2, 2)), {"A11": model.A},
            {"A11": torch.zeros(1, 2, 2)}, {"A11": torch.zeros(1, 2, 2)},
            ((0,),), losses=(), rescues=("A11",),
        )
