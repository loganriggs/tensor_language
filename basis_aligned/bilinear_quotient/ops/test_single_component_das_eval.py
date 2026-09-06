#!/usr/bin/env python3
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import single_component_das_eval as s


def test_validate_same_attention_component():
    assert s.validate_units(["attn:11:head:03", "attn:11:head:07"]) == (11, "head")


def test_validate_rejects_cross_layer():
    with pytest.raises(s.SingleComponentDASError):
        s.validate_units(["attn:08:head:01", "attn:11:head:03"])


def test_validate_rejects_mixed_native_components():
    with pytest.raises(s.SingleComponentDASError):
        s.validate_units(["attn:08:head:01", "mlp:08"])


def test_validate_rejects_multiple_whole_mlps():
    with pytest.raises(s.SingleComponentDASError):
        s.validate_units(["mlp:08", "mlp:08"])


def test_empirical_span_recovers_matrix_rank_and_row_space():
    delta = torch.tensor([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 3.0]])
    span, singular, rank = s.empirical_span(delta)
    assert rank == 2
    assert span.shape == (2, 3)
    assert torch.allclose(span @ span.T, torch.eye(2), atol=1e-6)
    projected = (delta @ span.T) @ span
    assert torch.allclose(projected, delta, atol=1e-5)
    assert singular[0] > singular[1] > singular[2]


def test_empirical_span_rejects_zero_delta():
    with pytest.raises(s.SingleComponentDASError):
        s.empirical_span(torch.zeros(3, 4))
