from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import mlp0_c512_mlp1_interchange as assay


class FakeAttention(nn.Module):
    def __init__(self, d_model, scale):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.linear.weight)
        self.linear.weight.data.mul_(scale)

    def forward(self, x, v1):
        value = self.linear(x)
        return value, value if v1 is None else v1


class FakeBlock(nn.Module):
    def __init__(self, d_model, scale):
        super().__init__()
        self.lambdas = nn.Parameter(torch.tensor([0.9, 0.1]))
        self.attn = FakeAttention(d_model, scale)
        self.mlp = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.mlp.weight)
        self.mlp.weight.data.mul_(scale / 2)

    def forward(self, x, v1, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        attn, v1 = self.attn(F.rms_norm(x, (x.shape[-1],)), v1)
        x = x + attn
        return x + self.mlp(F.rms_norm(x, (x.shape[-1],))), v1


class FakeModel(nn.Module):
    def __init__(self, d_model=4, vocab=7, layers=4):
        super().__init__()
        self.transformer = SimpleNamespace()
        self.transformer.wte = nn.Embedding(vocab, d_model)
        self.transformer.h = nn.ModuleList([FakeBlock(d_model, 0.03 * (i + 1)) for i in range(layers)])
        self.lm_head = nn.Linear(d_model, vocab, bias=False)


@torch.no_grad()
def ordinary_forward(model, idx, omit_mlp2=False):
    d_model = model.transformer.wte.weight.shape[1]
    x = F.rms_norm(model.transformer.wte(idx), (d_model,))
    x0, v1 = x, None
    for layer, block in enumerate(model.transformer.h):
        if omit_mlp2 and layer == 2:
            x = block.lambdas[0] * x + block.lambdas[1] * x0
            attn, v1 = block.attn(F.rms_norm(x, (d_model,)), v1)
            x = x + attn
        else:
            x, v1 = block(x, v1, x0)
    return (30 * torch.tanh(model.lm_head(F.rms_norm(x, (d_model,))) / 30)).float()


def test_capture_and_live_suffix_exactly_replay_parent():
    torch.manual_seed(2)
    model = FakeModel()
    idx = torch.randint(0, 7, (3, 5))
    cap = assay.capture_through_mlp1(model, model.transformer.h, idx)
    replay = assay.suffix_forward(model, model.transformer.h, cap["post"], cap["v1"], cap["x0"])
    torch.testing.assert_close(replay, ordinary_forward(model, idx), rtol=0, atol=0)


def test_raw_and_capped_suffix_readout_are_consistent():
    torch.manual_seed(22)
    model = FakeModel()
    idx = torch.randint(0, 7, (2, 4))
    cap = assay.capture_through_mlp1(model, model.transformer.h, idx)
    raw, capped = assay.suffix_forward(
        model, model.transformer.h, cap["post"], cap["v1"], cap["x0"], return_raw=True
    )
    torch.testing.assert_close(capped, 30 * torch.tanh(raw / 30), rtol=0, atol=0)


def test_mlp2_omit_suffix_matches_manual_parent():
    torch.manual_seed(3)
    model = FakeModel()
    idx = torch.randint(0, 7, (2, 6))
    cap = assay.capture_through_mlp1(model, model.transformer.h, idx)
    replay = assay.suffix_forward(
        model, model.transformer.h, cap["post"], cap["v1"], cap["x0"], background="mlp2_omit"
    )
    torch.testing.assert_close(replay, ordinary_forward(model, idx, omit_mlp2=True), rtol=0, atol=0)


def test_factorial_states_and_additive_interaction():
    exact = {"s": torch.tensor([1.0]), "m": torch.tensor([2.0])}
    candidate = {"s": torch.tensor([4.0]), "m": torch.tensor([8.0])}
    states = assay.physical_post_states(exact, candidate)
    assert {key: float(value) for key, value in states.items()} == {
        "OO": 3.0, "CC": 12.0, "CO": 6.0, "OC": 9.0
    }
    logits = {key: value.view(1, 1).expand(1, 3) for key, value in states.items()}
    assert torch.equal(assay.additive_interaction_prediction(logits), torch.zeros(1, 3))


def test_document_derangement_preserves_cells_multiset_and_changes_document():
    document = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3])
    cell = torch.tensor([0] * 8 + [1] * 4)
    permutation = assay.document_derangement(document, cell)
    assert torch.equal(cell, cell[permutation])
    assert not bool((document == document[permutation]).any())
    assert sorted(permutation.tolist()) == list(range(len(document)))


def test_document_derangement_fails_when_one_document_dominates_cell():
    with pytest.raises(RuntimeError, match="cannot be deranged"):
        assay.document_derangement(torch.tensor([0, 0, 0, 1]), torch.zeros(4))


def test_norm_matched_control_and_unknown_background():
    delta = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    native = torch.tensor([[[2.0, 0.0], [1.0, 0.0]]])
    control = assay.norm_matched_native_write(delta, native)
    torch.testing.assert_close(control.norm(dim=-1), delta.norm(dim=-1))
    with pytest.raises(ValueError, match="unknown suffix"):
        assay.suffix_forward(None, [], torch.ones(1, 1, 2), torch.ones(1), torch.ones(1, 1, 2), background="x")
