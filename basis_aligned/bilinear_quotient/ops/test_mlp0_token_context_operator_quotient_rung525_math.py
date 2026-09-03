"""CPU tests for rung 525's exact operator-sketch and grouping math."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


OPS = Path(__file__).parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MATH = _load("mlp0_token_context_operator_quotient_rung525_math")


def test_factorized_sketch_matches_explicit_bilinear_output():
    generator = torch.Generator().manual_seed(1)
    tokens, probes, hidden, output = 7, 5, 11, 6
    token_input = torch.randn(tokens, 9, generator=generator, dtype=torch.float64)
    context_input = torch.randn(probes, 9, generator=generator, dtype=torch.float64)
    left = torch.randn(hidden, 9, generator=generator, dtype=torch.float64)
    right = torch.randn(hidden, 9, generator=generator, dtype=torch.float64)
    down = torch.randn(output, hidden, generator=generator, dtype=torch.float64)
    q = torch.randn(probes, output, generator=generator, dtype=torch.float64)
    tl, tr = token_input @ left.mT, token_input @ right.mT
    cl, cr = context_input @ left.mT, context_input @ right.mT
    sketch = MATH.operator_sketch(tl, tr, cl, cr, q @ down, gain=1.7)
    explicit = torch.empty_like(sketch)
    for token in range(tokens):
        for probe in range(probes):
            write = 1.7 * down @ (tl[token] * cr[probe] + cl[probe] * tr[token])
            explicit[token, probe] = q[probe] @ write
    assert torch.allclose(sketch, explicit, atol=1e-10, rtol=1e-10)


def test_far_search_recovers_shared_features_despite_raw_distance():
    # Receivers 0,1 and donors 2,3 have matching operator features but
    # orthogonal raw vectors.
    values = torch.tensor([[0., 0.], [2., 2.], [0., 0.], [2., 2.], [9., -9.]])
    raw = torch.eye(5)
    receivers = torch.tensor([0, 1], dtype=torch.int64)
    donors = torch.tensor([2, 3, 4], dtype=torch.int64)
    chosen, distances, cosines = MATH.nearest_far_donors(
        values, raw, receivers, donors, raw_cosine_ceiling=0.5, chunk_size=1
    )
    assert chosen.tolist() == [2, 3]
    assert torch.equal(distances, torch.zeros(2))
    assert bool((cosines <= 0.5).all())


def test_standardization_uses_only_donors():
    sketch = torch.tensor([[100., 100.], [0., 2.], [2., 4.]])
    donors = torch.tensor([1, 2], dtype=torch.int64)
    result = MATH.standardize_from_donors(sketch, donors)
    assert torch.allclose(result.mean, torch.tensor([1., 3.]))
    assert torch.allclose(result.scale, torch.tensor([1., 1.]))


def test_score_licenses_only_transfer_and_repeated_groups():
    n = 1_200
    candidate_a = torch.linspace(0.01, 0.1, n)
    candidate_b = candidate_a * 1.01
    raw = torch.full((n,), 1.0)
    random = torch.full((n, 16), 2.0)
    deranged = torch.full((n,), 1.5)
    donors = torch.arange(n, dtype=torch.int64) // 2
    score = MATH.score_real(
        candidate_a_distance=candidate_a,
        candidate_b_distance=candidate_b,
        raw_b_distance=raw,
        random_b_distances=random,
        deranged_b_distance=deranged,
        candidate_donors=donors,
        half_a_b_distances=(candidate_b, candidate_b * 1.01),
    )
    assert score["prediction_b_operator_transfer"]
    assert score["prediction_c_repeated_groups"]
    assert score["physical_successor_licensed"]


def test_score_rejects_shape_change():
    with pytest.raises(ValueError, match="sixteen"):
        MATH.score_real(
            candidate_a_distance=torch.ones(3), candidate_b_distance=torch.ones(3),
            raw_b_distance=torch.ones(3), random_b_distances=torch.ones(3, 15),
            deranged_b_distance=torch.ones(3), candidate_donors=torch.arange(3),
            half_a_b_distances=(torch.ones(3), torch.ones(3)),
        )
