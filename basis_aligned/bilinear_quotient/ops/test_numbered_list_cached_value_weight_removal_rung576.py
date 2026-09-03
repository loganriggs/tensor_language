from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


PATH = Path(__file__).with_name("numbered_list_cached_value_weight_removal_rung576.py")
SPEC = importlib.util.spec_from_file_location("r576", PATH)
r576 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(r576)


def test_projected_term_uses_base_score_position_and_distinct_donor_value_position():
    pattern = torch.zeros(1, r576.r573.N_HEAD, 3, 3)
    cached = torch.zeros(1, 3, r576.r573.N_HEAD, r576.r573.HEAD_D)
    expected = torch.zeros(1, r576.r573.D)
    for slot, head in enumerate(r576.HEADS):
        score = float(slot + 2)
        pattern[0, head, 2, 0] = score
        cached[0, 1, head] = float(slot + 5)
        expected[0, head * r576.r573.HEAD_D:(head + 1) * r576.r573.HEAD_D] = score * float(slot + 5)
    observed = r576.projected_terms(
        {"pattern": pattern}, cached, torch.tensor([2]), torch.tensor([0]),
        torch.eye(r576.r573.D), value_sources=torch.tensor([1]))
    assert torch.equal(observed, expected)


def test_compiled_cached_is_literal_embedding_block0_value_formula():
    torch.manual_seed(576)
    embedding = nn.Embedding(7, r576.r573.D)
    value = nn.Linear(r576.r573.D, r576.r573.D, bias=False)
    with torch.no_grad():
        value.weight.copy_(torch.eye(r576.r573.D))
    block0 = SimpleNamespace(lambdas=torch.tensor([0.7, 0.3]), attn=SimpleNamespace(c_v=value))
    blocks = [block0] + [SimpleNamespace() for _ in range(r576.LAYER - 1)]
    blocks.append(SimpleNamespace(attn=SimpleNamespace(lamb=torch.tensor(4.0))))
    model = SimpleNamespace(transformer=SimpleNamespace(wte=embedding, h=blocks))
    tokens = torch.tensor([[1, 4]])
    x0 = F.rms_norm(embedding(tokens), (r576.r573.D,))
    expected = 4 * F.rms_norm(x0, (r576.r573.D,))
    expected = expected.view(1, 2, r576.r573.N_HEAD, r576.r573.HEAD_D)
    assert torch.allclose(r576.compiled_cached(model, tokens), expected, atol=1e-6, rtol=1e-6)


def test_dryrun_price_formula_is_exact_for_frozen_row_counts():
    rows, _ = r576.load_authority()
    fit_pair, fit_remove = r576.count_chunks(rows, "FIT")
    select_pair, select_remove = r576.count_chunks(rows, "SELECT")
    assert (fit_pair, fit_remove, select_pair, select_remove) == (9, 34, 8, 19)
    assert 6 * (fit_pair + select_pair) + 2 * (fit_remove + select_remove) + 2 == 210
