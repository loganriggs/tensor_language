from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("mlp0_centered_context_source_quotient_rung527_run.py")
SPEC = importlib.util.spec_from_file_location("r527_run", PATH)
R = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(R)


def test_frozen_dependencies_and_population_are_present():
    observed = R.validate_dependencies()
    rows, _masks, discovery, validation, fit, _metadata = R.population()
    assert len(observed) == len(R.FROZEN_SHA256)
    assert tuple(rows.shape) == (1000, 257)
    assert (len(discovery), len(validation), len(fit)) == (32, 30, 96)


def test_effect_views_use_member_minus_matched_control():
    counts = torch.ones(2, 2, 3, dtype=torch.float64)
    sums = torch.zeros(R.qm.N_TERMS, 2, 2, 3, dtype=torch.float64)
    sums[:, :, 0] = 3.0
    sums[:, :, 1] = 1.0
    collection = {"sums": sums, "counts": counts}
    views = R.effect_views(collection)
    assert torch.equal(views["halves"], torch.full((20, 2, 3), 2.0))
    assert torch.equal(views["pooled"], torch.full((20, 3), 2.0))


def test_instrument_gate_requires_small_numerical_remainder_and_live_edits():
    diagnostics = R.empty_diagnostics()
    diagnostics.update({
        "calls_exact": True,
        "source_partition_maximum_relative_squared": 0.0,
        "context_closure_relative_squared": 0.0,
        "remainder_energy_fraction": [0.009, 0.008],
        "minimum_term_edit_rms": 1e-5,
        "supports_positive": True,
    })
    collection = {"diagnostics": diagnostics}
    assert R.instrument_holds(collection, require_support=True)
    diagnostics["remainder_energy_fraction"][1] = 0.011
    assert not R.instrument_holds(collection, require_support=True)


def test_physical_substitution_scoring_requires_both_directions_and_all_windows():
    effects, _expected = R.qm.planted_pair_problem()
    candidates, _summary = R.qm.discover_pairs(effects)
    candidates = candidates[:1]
    target_left, target_right = candidates[0]["left"], candidates[0]["right"]
    directions = [
        {"target": target_left},
        {"target": target_right},
    ]
    exact = {"halves": effects, "pooled": effects.mean(1)}
    observed_halves = torch.stack((effects[target_left], effects[target_right]))
    observed_pooled = observed_halves.mean(1)
    counts = torch.ones(2, 2, effects.shape[-1], dtype=torch.float64)
    sums = torch.zeros(2, 2, 2, effects.shape[-1], dtype=torch.float64)
    sums[:, :, 0] = observed_halves
    collection = {"directions": directions, "sums": sums, "counts": counts}
    passing, checks = R.score_substitutions(collection, exact, candidates)
    assert len(passing) == 1
    assert next(iter(checks.values()))["holds"]
    collection["sums"][1].zero_()
    passing, _checks = R.score_substitutions(collection, exact, candidates)
    assert not passing


def test_two_node_physical_pair_forms_a_scale_consistent_group():
    candidate = {
        "left": 0, "right": 1, "beta_left_from_right": 2.0,
        "left_name": R.qm.TERM_NAMES[0], "right_name": R.qm.TERM_NAMES[1],
    }
    groups = R.quotient_groups([candidate])
    assert len(groups) == 1
    assert groups[0]["maximum_scale_cycle_relative_error"] == 0.0
    assert R.nontrivial_pair(candidate)
