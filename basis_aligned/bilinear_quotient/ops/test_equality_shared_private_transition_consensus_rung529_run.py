from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).with_name("equality_shared_private_transition_consensus_rung529_run.py")
SPEC = importlib.util.spec_from_file_location("r529_run", PATH)
R = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(R)


def test_frozen_dependencies_and_population_hold():
    observed, population = R.validate_dependencies()
    rows, _task, _circuit, _scales, discovery, validation, _metadata = population
    assert len(observed) == len(R.FROZEN_SHA256)
    assert tuple(rows.shape) == (1000, 257)
    assert (len(discovery), len(validation)) == (32, 30)


def test_phase_pairs_are_complete_and_conditional_pairs_are_frozen():
    singles, wrong = R._pairs_for_phase(R.qm.ACTIONS, None)
    assert len(singles) == 12
    assert len(wrong) == 6
    assert ("Z7", "W7") not in wrong
    candidates = (
        {"target": "N", "single_donor": "Z7", "wrong_control": "W8"},
        {"target": "Z8", "single_donor": "P", "wrong_control": "W7"},
    )
    singles, wrong = R._pairs_for_phase(("N", "Z8"), candidates)
    assert singles == (("N", "Z7"), ("Z8", "P"))
    assert wrong == (("N", "W8"), ("Z8", "W7"))


def test_dry_run_closes_outcomes_and_freezes_corrected_price():
    report = R.dry_run()
    assert report["status"] == "dry_run_passed"
    assert report["model_loaded"] is False
    assert report["outcomes_opened"] is False
    assert report["unconditional_discovery_forwards"] == 7688
    assert report["maximum_conditional_forwards"] == 23396
    assert report["single_donor_substitutions"] == 12
    assert report["wrong_sign_consensuses"] == 6
    assert report["planted_suite_passes"] is True


def test_repeat_scoring_requires_improvement_over_frozen_single():
    generator = torch.Generator().manual_seed(529)
    target = torch.randn(1, 2, 4, 8, generator=generator, dtype=torch.float64) * .01
    task = torch.randn(1, 2, 4, 4, generator=generator, dtype=torch.float64) * .01
    consensus = target + torch.randn(target.shape, generator=generator, dtype=torch.float64) * .0003
    consensus_task = task + torch.randn(task.shape, generator=generator, dtype=torch.float64) * .0003
    # This donor is better than consensus, so the registered .03 advantage fails.
    single = target + torch.randn(target.shape, generator=generator, dtype=torch.float64) * .00005
    wrong = -target
    documents = 4
    tags = 8
    data = {
        "targets": ("N",),
        "task_counts": torch.ones(documents, len(R.CELLS), dtype=torch.float64),
        "circuit_counts": torch.ones(2, 2, tags, dtype=torch.float64),
    }
    def sums_from_views(prefix, circuit_values, task_values):
        circuit_sums = torch.zeros(1, 4, 2, 2, tags, dtype=torch.float64)
        task_sums = torch.zeros(1, 4, documents, len(R.CELLS), dtype=torch.float64)
        # Encode desired half effects as member-control mean differences.
        for half in range(2):
            circuit_sums[0, :, half, 0] = circuit_values[0, half]
            task_sums[0, :, 2 * half:2 * half + 2, list(R.TASK_CONTEXT_INDICES)] = task_values[0, half].unsqueeze(1)
        data[f"{prefix}_circuit_sums"] = circuit_sums
        data[f"{prefix}_task_sums"] = task_sums
    for prefix, cvalue, tvalue in (
        ("target", target, task), ("consensus", consensus, consensus_task),
        ("single", single, task), ("wrong", wrong, -task),
    ):
        sums_from_views(prefix, cvalue, tvalue)
    passers, checks = R.score_repeat(data, circuit_cosine=.75, circuit_error=.55)
    assert passers == []
    assert not checks["N"]["windows"]["half0"]["holds"]
