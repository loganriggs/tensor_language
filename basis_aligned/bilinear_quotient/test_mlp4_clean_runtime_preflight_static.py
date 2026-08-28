import ast
import hashlib
import json

from pathlib import Path

import torch
import torch.nn.functional as F

import jacclust.tt_model as TT
from . import bilin18_reference_forward as reference


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/"mlp4_clean_runtime_preflight_protocol.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_is_synthetic_only_and_source_pinned():
    protocol = json.loads(PROTOCOL.read_text())
    for filename, expected in protocol["pinned_sources"].items():
        assert sha(HERE/filename) == expected
    source = (HERE/"mlp4_clean_runtime_preflight.py").read_text()
    ast.parse(source)
    assert "torch.arange" in source and "reference_forward" in source
    assert source.index("before = telemetry()") < source.index("runtime.initialize()")
    assert all(stage in source for stage in
               ("after_load", "after_candidate", "after_reference"))
    for forbidden in ("fineweb", "ROWS", "validation_rows", "fit_rows",
                      "census", "datasets", "torch.load"):
        assert forbidden not in source
    assert protocol["synthetic_input"]["token_ids"] == "arange(32)"
    assert protocol["resources"]["hard_abort_peak_gib"] <= 5
    assert protocol["resources"]["hard_abort_temperature_c"] <= 82
    assert not protocol["permissions"]["validation_rows_may_be_opened"]
    assert not protocol["permissions"]["any_natural_rows_may_be_opened"]


def test_independent_reference_is_data_free_and_does_not_call_model_blocks():
    source = (HERE/"bilin18_reference_forward.py").read_text()
    ast.parse(source)
    for forbidden in ("torch.load", "huggingface", "dataset", "fineweb",
                      "block(", "attention(", "mlp("):
        assert forbidden not in source
    assert "mlp.Down(mlp.Left(z)*mlp.Right(z))+mlp.Down_bias" in source
    assert "score1*score2" in source and "masked_fill" in source


def test_independent_equations_equal_tiny_randomized_module_forward():
    torch.manual_seed(17)
    config = TT.GPTConfig(vocab_size=32, n_layer=3, n_head=2, n_embd=8,
                          expansion_factor=2, bilinear=True,
                          bilinear_attn=True, squared_attn=True)
    model = TT.GPT(config).double().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, .15)
    tokens = torch.tensor([[0, 3, 7, 1, 9], [2, 8, 4, 6, 5]])
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (config.n_embd,))
        x0 = x; shared = None
        for block in model.transformer.h:
            x, shared = block(x, shared, x0)
        expected = 30*torch.tanh(
            model.lm_head(F.rms_norm(x, (config.n_embd,)))/30)
        actual = reference.reference_forward(model, tokens)
    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-11)
