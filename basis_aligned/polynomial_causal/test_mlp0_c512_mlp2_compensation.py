from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import mlp0_c512_mlp2_compensation as assay
import mlp0_c512_mlp1_interchange as mlp1_assay


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
    def __init__(self, d_model=4, vocab=7, layers=5):
        super().__init__()
        self.transformer = SimpleNamespace()
        self.transformer.wte = nn.Embedding(vocab, d_model)
        self.transformer.h = nn.ModuleList(
            [FakeBlock(d_model, 0.03 * (i + 1)) for i in range(layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab, bias=False)


@torch.no_grad()
def ordinary_forward_from_mlp1(model, post, v1, x0, *, omit_mlp2=False):
    x = post
    for layer, block in enumerate(model.transformer.h[2:], start=2):
        if omit_mlp2 and layer == 2:
            x = block.lambdas[0] * x + block.lambdas[1] * x0
            attn, v1 = block.attn(F.rms_norm(x, (x.shape[-1],)), v1)
            x = x + attn
        else:
            x, v1 = block(x, v1, x0)
    raw = model.lm_head(F.rms_norm(x, (x.shape[-1],))).float()
    return raw, 30 * torch.tanh(raw / 30)


def candidate_capture(exact):
    return {
        key: value.clone() for key, value in exact.items()
    } | {
        "s": exact["s"] + 0.2,
        "m": exact["m"] - 0.1,
        "post": exact["post"] + 0.1,
    }


def test_exact_and_candidate_mlp2_capture_replay_ordinary_live_parents():
    torch.manual_seed(2)
    model = FakeModel()
    idx = torch.randint(0, 7, (3, 5))
    exact = mlp1_assay.capture_through_mlp1(model, model.transformer.h, idx)
    candidate = candidate_capture(exact)
    paths = assay.post_mlp1_paths(exact, candidate)
    interfaces = assay.capture_mlp2_paths(model.transformer.h, paths)
    for path, parent in (("O", exact), ("C", candidate)):
        raw, capped = assay.suffix_from_mlp2(
            model, model.transformer.h, interfaces[path]["post"],
            interfaces[path]["v1"], interfaces[path]["x0"], return_raw=True,
        )
        parent_raw, parent_capped = ordinary_forward_from_mlp1(
            model, parent["post"], parent["v1"], parent["x0"]
        )
        torch.testing.assert_close(raw, parent_raw, rtol=0, atol=0)
        torch.testing.assert_close(capped, parent_capped, rtol=0, atol=0)


def test_full_two_by_two_matrix():
    torch.manual_seed(3)
    model = FakeModel()
    idx = torch.randint(0, 7, (2, 6))
    exact = mlp1_assay.capture_through_mlp1(model, model.transformer.h, idx)
    candidate = candidate_capture(exact)
    paths = assay.post_mlp1_paths(exact, candidate)
    interfaces = assay.capture_mlp2_paths(model.transformer.h, paths)
    matrix = assay.physical_mlp2_matrix(interfaces)
    assert set(matrix) == set(assay.PHYSICAL_ARMS)
    for p in assay.UPSTREAM_PATHS:
        for q in assay.UPSTREAM_PATHS:
            torch.testing.assert_close(matrix[p + q], interfaces[p]["s"] + interfaces[q]["m"])


def test_exact_and_candidate_omitted_write_paths_replay_ordinary_parents():
    torch.manual_seed(4)
    model = FakeModel()
    idx = torch.randint(0, 7, (2, 4))
    exact = mlp1_assay.capture_through_mlp1(model, model.transformer.h, idx)
    paths = assay.post_mlp1_paths(exact, candidate_capture(exact))
    interfaces = assay.capture_mlp2_paths(model.transformer.h, paths)
    candidate = candidate_capture(exact)
    for path, parent in (("O", exact), ("C", candidate)):
        raw, capped = assay.suffix_from_mlp2(
            model, model.transformer.h, interfaces[path]["s"],
            interfaces[path]["v1"], interfaces[path]["x0"], return_raw=True,
        )
        parent_raw, parent_capped = ordinary_forward_from_mlp1(
            model, parent["post"], parent["v1"], parent["x0"], omit_mlp2=True,
        )
        torch.testing.assert_close(raw, parent_raw, rtol=0, atol=0)
        torch.testing.assert_close(capped, parent_capped, rtol=0, atol=0)


def test_additive_prediction_and_write_norm_control():
    base = torch.tensor([[[1.0, 2.0, 4.0]]])
    logits = {
        "OO": base,
        "CO": base + torch.tensor([[[2.0, 0.0, 0.0]]]),
        "OC": base + torch.tensor([[[0.0, 3.0, 0.0]]]),
    }
    pred = assay.additive_factorial_prediction(logits)
    assert torch.allclose(pred.mean(-1), torch.zeros_like(pred.mean(-1)))
    torch.testing.assert_close(
        pred, assay.centered_logits(logits["CO"] + logits["OC"] - logits["OO"])
    )
    delta = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    native = torch.tensor([[[2.0, 0.0], [1.0, 0.0]]])
    control = assay.norm_matched_native_write(delta, native)
    torch.testing.assert_close(control.norm(dim=-1), delta.norm(dim=-1))
    with pytest.raises(ValueError, match="zero native write"):
        assay.norm_matched_native_write(
            torch.tensor([[[1.0, 0.0]]]), torch.zeros(1, 1, 2)
        )
    with pytest.raises(ValueError, match="incomplete"):
        assay.additive_factorial_prediction({})


def test_rejects_incomplete_or_misaligned_inputs():
    with pytest.raises(ValueError, match="incomplete"):
        assay.post_mlp1_paths({}, {})
    with pytest.raises(ValueError, match="exactly O/C"):
        assay.physical_mlp2_matrix({})


@pytest.mark.parametrize("key", ["x0", "v1"])
def test_carried_state_identity_mismatch_fails_closed(key):
    torch.manual_seed(5)
    model = FakeModel()
    idx = torch.randint(0, 7, (2, 4))
    exact = mlp1_assay.capture_through_mlp1(model, model.transformer.h, idx)
    candidate = candidate_capture(exact)
    candidate[key] = candidate[key] + 1e-3
    with pytest.raises(ValueError, match=f"{key} exact/candidate identity"):
        assay.post_mlp1_paths(exact, candidate, state_identity_tolerance=1e-6)


def test_crossed_suffix_never_calls_mlp1_or_mlp2():
    torch.manual_seed(6)
    model = FakeModel()
    idx = torch.randint(0, 7, (2, 4))
    exact = mlp1_assay.capture_through_mlp1(model, model.transformer.h, idx)
    candidate = candidate_capture(exact)
    paths = assay.post_mlp1_paths(exact, candidate)
    interfaces = assay.capture_mlp2_paths(model.transformer.h, paths)
    matrix = assay.physical_mlp2_matrix(interfaces)
    calls = {"mlp1": 0, "mlp2": 0}

    def count(name):
        def hook(module, args, output):
            calls[name] += 1
        return hook

    hooks = [
        model.transformer.h[1].mlp.register_forward_hook(count("mlp1")),
        model.transformer.h[2].mlp.register_forward_hook(count("mlp2")),
    ]
    try:
        assay.suffix_from_mlp2(
            model, model.transformer.h, matrix["CO"], interfaces["C"]["v1"],
            interfaces["C"]["x0"],
        )
    finally:
        for hook in hooks:
            hook.remove()
    assert calls == {"mlp1": 0, "mlp2": 0}
