"""Tests for rung 526's exact circuit-response contraction and gates."""

from __future__ import annotations

import torch

import mlp0_circuit_response_operator_quotient_rung526_math as qm


def test_aggregated_contraction_matches_explicit_float64():
    generator = torch.Generator().manual_seed(526)
    token_left = torch.randn(9, 7, generator=generator, dtype=torch.float64)
    token_right = torch.randn(9, 7, generator=generator, dtype=torch.float64)
    context_left = torch.randn(11, 7, generator=generator, dtype=torch.float64)
    context_right = torch.randn(11, 7, generator=generator, dtype=torch.float64)
    downstream = torch.randn(5, 11, 7, generator=generator, dtype=torch.float64)
    fast = qm.circuit_signature(
        token_left, token_right, context_left, context_right, downstream, gain=0.37
    )
    explicit = qm.explicit_circuit_signature(
        token_left, token_right, context_left, context_right, downstream, gain=0.37
    )
    error = (fast - explicit).square().sum() / explicit.square().sum()
    assert float(error) <= 1e-28


def test_derangement_preserves_each_token_marginal_and_changes_coupling():
    values = torch.arange(20 * 32, dtype=torch.float32).reshape(20, 32)
    changed = qm.derange_coordinates(values)
    assert not torch.equal(values, changed)
    assert torch.equal(values.sort(1).values, changed.sort(1).values)


def test_discovery_pass_and_strong_null_are_distinct():
    n = 1200
    distance = torch.full((n,), 0.1)
    raw = torch.full((n,), 1.0)
    random = torch.full((n, 16), 2.0)
    scrambled = torch.full((n,), 1.5)
    donors = torch.arange(n, dtype=torch.int64).remainder(200) * 5 + 1
    old = donors + 1
    result = qm.score_discovery(
        distance_d0=distance, distance_d1=distance,
        raw_d1=raw, random_d1=random, scrambled_d1=scrambled,
        candidate_donors=donors,
        circuit_half_d1=(distance, distance * 1.1), rung525_donors=old,
    )
    assert result["prediction_b_document_transfer"]
    assert result["prediction_d_reusable_changed_groups"]
    assert not result["strong_null"]


def test_discovery_raw_baseline_fires_strong_null():
    n = 1200
    distance = torch.full((n,), 1.0)
    random = torch.full((n, 16), 2.0)
    donors = torch.arange(n, dtype=torch.int64).remainder(200) * 5 + 1
    result = qm.score_discovery(
        distance_d0=distance, distance_d1=distance,
        raw_d1=distance, random_d1=random, scrambled_d1=random[:, 0],
        candidate_donors=donors,
        circuit_half_d1=(distance, distance), rung525_donors=donors,
    )
    assert result["strong_null"]
    assert not result["prediction_b_document_transfer"]


def test_validation_gate():
    n = 100
    candidate = torch.full((n,), 0.2)
    result = qm.score_validation_half(
        candidate=candidate,
        raw=torch.ones(n),
        random=torch.full((n, 16), 2.0),
        scrambled=torch.ones(n),
    )
    assert result["passes"]
