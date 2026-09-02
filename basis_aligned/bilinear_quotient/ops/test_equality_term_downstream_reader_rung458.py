from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("equality_term_downstream_reader_rung458.py")
SPEC = importlib.util.spec_from_file_location("equality_reader_rung458", PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def _report(cosine: float, live: float = .01) -> dict[str, object]:
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[0, 2] = matrix[2, 0] = cosine
    return {
        "cosine": matrix.tolist(),
        "response_rms_over_native_write_rms": [live] * 4,
    }


def test_frozen_search_has_72_pair_reader_candidates_and_deterministic_choice() -> None:
    reports = {
        cell: {
            component: _report(.10)
            for component in module.COMMON_COMPONENTS
        }
        for cell in module.RESPONSE_CELLS
    }
    reports["all_positive"]["m12"] = _report(.91)
    reports["matched_negative"]["m12"] = _report(.20)
    reports["off_target"]["m12"] = _report(.25)
    candidates, selected = module.select_candidate(reports)
    assert len(candidates) == 72
    assert selected is not None
    assert selected["component"] == "m12"
    assert selected["pair_indices"] == [0, 2]
    assert selected["task_margin"] == .91 - .25


def test_response_accumulator_recovers_planted_cosine() -> None:
    stats = module.empty_response_stats(component_count=1)
    shape = (module.BATCH, 3, 2)
    native = {"m12": torch.ones(shape)}
    common = torch.tensor([1.0, 2.0]).view(1, 1, 2).expand(shape)
    orthogonal = torch.tensor([2.0, -1.0]).view(1, 1, 2).expand(shape)
    singleton = [
        {"m12": native["m12"] + common},
        {"m12": native["m12"] + orthogonal},
        {"m12": native["m12"] + 2 * common},
        {"m12": native["m12"] - orthogonal},
    ]
    masks = {
        cell: torch.ones(module.BATCH, 3, dtype=torch.bool)
        for cell in module.RESPONSE_CELLS
    }
    module.accumulate_responses(
        stats, native, singleton, masks, 0, components=("m12",), blocks=None,
    )
    report = module.response_reports(stats, components=("m12",))
    cosine = report["all_positive"]["m12"]["cosine"]
    assert abs(cosine[0][2] - 1.0) < 1e-12
    assert abs(cosine[0][1]) < 1e-12
    assert report["all_positive"]["m12"]["tokens"] == 12


def test_patch_bootstrap_uses_signed_ce_recovery_and_simultaneous_lower_bound() -> None:
    counts = torch.ones(module.DOCUMENTS_PER_HALF, len(module.CE_CELLS), dtype=torch.float64)
    losses = {"native": torch.ones_like(counts)}
    for index in (0, 2):
        name = module.term_name(index)
        losses[f"remove:{name}"] = torch.full_like(counts, 2.0)
        losses[f"patch:{name}"] = torch.full_like(counts, 1.5)
        off = module.CE_CELLS.index("off_target")
        losses[f"patch:{name}"][:, off] = losses[f"remove:{name}"][:, off]
    original_draws = module.BOOTSTRAP_DRAWS
    module.BOOTSTRAP_DRAWS = 1_000
    try:
        report = module.patch_bootstrap(losses, counts, (0, 2))
    finally:
        module.BOOTSTRAP_DRAWS = original_draws
    assert report["every_bootstrap_removal_stake_positive"]
    for row in report["terms"]:
        assert abs(row["removal_stake_nat"] - 1.0) < 1e-12
        assert abs(row["patch_recovery"] - .5) < 1e-12
        assert row["simultaneous_lower"] > .49
        assert row["off_target_change_from_removal_nat"] == 0.0


def test_interchange_separates_small_within_from_large_between_swaps() -> None:
    counts = torch.ones(module.DOCUMENTS_PER_HALF, len(module.CE_CELLS), dtype=torch.float64)
    losses = {"native": torch.ones_like(counts)}
    for index in range(4):
        losses[f"remove:{module.term_name(index)}"] = torch.full_like(counts, 2.0)
    swaps = (
        ("within", 0, 2), ("within", 2, 0),
        ("between", 0, 3), ("between", 2, 1),
    )
    for group, target, source in swaps:
        value = 2.05 if group == "within" else 2.50
        losses[f"{group}:{module.term_name(target)}<-{module.term_name(source)}"] = (
            torch.full_like(counts, value)
        )
    report = module.interchange_report(losses, counts, (0, 2), swaps)
    assert report["separation"] > 9.9
    assert report["p_value"] <= .05
    assert report["within_mean_over_removal_stake"] < .051


def test_validation_gate_requires_task_specific_transfer_and_context_signs() -> None:
    selected = {"component": "m12", "pair_indices": [0, 2]}
    fit = {
        cell: {"m12": _report(value)}
        for cell, value in {
            "all_positive": .90, "matched_negative": .30, "off_target": .20,
            "near_positive": .80, "far_positive": .60,
            "one_predecessor_positive": .75, "multiple_predecessor_positive": .60,
        }.items()
    }
    validation = {
        cell: {"m12": _report(value)}
        for cell, value in {
            "all_positive": .75, "matched_negative": .30, "off_target": .25,
            "near_positive": .70, "far_positive": .55,
            "one_predecessor_positive": .65, "multiple_predecessor_positive": .50,
        }.items()
    }
    report = module.validation_response_gate(fit, validation, selected)
    assert report["passed"]
    validation["near_positive"]["m12"] = _report(.40)
    assert not module.validation_response_gate(fit, validation, selected)["passed"]
