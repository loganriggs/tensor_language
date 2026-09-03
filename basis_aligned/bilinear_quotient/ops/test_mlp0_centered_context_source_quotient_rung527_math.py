from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("mlp0_centered_context_source_quotient_rung527_math.py")
SPEC = importlib.util.spec_from_file_location("r527_math", PATH)
R = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(R)


def test_term_vocabulary_is_five_linear_five_self_ten_cross():
    assert len(R.TERM_SPECS) == 20
    assert sum(row["operation"] == "linear" for row in R.TERM_SPECS) == 5
    assert sum(row["operation"] == "self" for row in R.TERM_SPECS) == 5
    assert sum(row["operation"] == "cross" for row in R.TERM_SPECS) == 10
    assert len(set(R.TERM_NAMES)) == 20


def test_exact_centered_polynomial_identity_and_expectation_assignment():
    result = R.planted_algebra()
    assert max(result.values()) <= 1e-10


def test_detector_recovers_only_the_two_planted_proportional_pairs():
    effects, expected = R.planted_pair_problem()
    pairs, summary = R.discover_pairs(effects)
    observed = {(row["left"], row["right"]) for row in pairs}
    assert observed == expected
    assert summary["candidate_count"] == 2
    assert summary["small_relation"]


def test_confirmation_uses_frozen_discovery_scale():
    effects, expected = R.planted_pair_problem()
    candidates, _ = R.discover_pairs(effects)
    confirmation = effects.clone()
    confirmation[:, 1] += 0.001 * torch.roll(confirmation[:, 1], 1, -1)
    passing, checks = R.confirmation_pairs(
        confirmation, candidates, confirmation.mean(1))
    assert {(row["left"], row["right"]) for row in passing} == expected
    assert all(row["holds"] for row in checks.values())


def test_independent_circuit_coordinate_permutations_destroy_planted_pairs():
    effects, _expected = R.planted_pair_problem()
    counts = R.permutation_control_counts(effects, range(527_300, 527_316))
    assert counts == [0] * 16


def test_full_planted_suite_holds():
    assert R.planted_suite()["holds"]
