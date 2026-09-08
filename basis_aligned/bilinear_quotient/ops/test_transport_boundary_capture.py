import pytest
import torch

import transport_boundary_capture as subject


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lambdas = torch.nn.Parameter(torch.tensor([.75, .25]))
        self.attn = torch.nn.Module(); self.attn.c_proj = torch.nn.Linear(3, 3, bias=False)
        self.mlp = torch.nn.Identity()
        with torch.no_grad(): self.attn.c_proj.weight.copy_(torch.eye(3))

    def forward(self, x, v1, x0):
        mixed = self.lambdas[0]*x + self.lambdas[1]*x0
        raw = mixed + self.attn.c_proj(mixed)
        return self.mlp(raw), v1


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.transformer = torch.nn.Module(); self.transformer.h = torch.nn.ModuleList([Block(), Block()])


def test_capture_and_raw_reconstruction():
    model = Model(); x = torch.tensor([[1., 2., 3.]]); x0 = torch.tensor([[3., 2., 1.]])
    def execute():
        y, v = model.transformer.h[0](x, None, x0)
        return model.transformer.h[1](y, v, x0)
    result, state, calls = subject.execute_with_boundaries(
        model, execute, block_layers=(0, 1), raw_attention_layer=1, mlp_input_layer=1)
    raw = subject.pre_mlp_raw_state(model, state, layer=1)
    assert torch.equal(raw, state["M1_input"])
    assert all(value == 1 for value in calls.values())
    assert torch.equal(result[0], raw)


def test_rejects_inconsistent_inventory():
    with pytest.raises(subject.TransportBoundaryError):
        subject.execute_with_boundaries(Model(), lambda: None,
            block_layers=(0,), raw_attention_layer=1, mlp_input_layer=0)
