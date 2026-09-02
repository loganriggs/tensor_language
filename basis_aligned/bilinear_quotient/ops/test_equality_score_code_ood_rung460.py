from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("equality_score_code_ood_rung460.py")
SPEC = importlib.util.spec_from_file_location("equality_score_code_ood_rung460", PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_code_role_and_natural_selection_are_hash_frozen() -> None:
    payload, masks, scales, metadata = module.validate_inputs()
    assert payload["role"] == "ood_code"
    assert tuple(payload["rows"].shape) == (192, 257)
    assert metadata["frozen_selected_pair"] == "L5H5->L8H4"
    assert metadata["frozen_control_pair"] == "L7H3->L8H4"
    assert metadata["frozen_factor"] == "score"
    assert metadata["frozen_component"] == "m9"
    assert scales["L5H5->L8H4"]["score_ratio"] == 0.5371214943951729
    assert int(masks["all_positive"].sum()) == 10848


def test_response_report_uses_task_specific_margin() -> None:
    stats = module._empty_response_stats()
    stats["ref2"].fill_(1.0)
    stats["hyb2"].fill_(1.0)
    stats["write2"].fill_(100.0)
    stats["tokens"].fill_(10)
    stats["cross"][0, module.CELLS.index("all_positive")] = .85
    stats["cross"][0, module.CELLS.index("matched_negative")] = .30
    stats["cross"][0, module.CELLS.index("off_target")] = .20
    report = module.response_report(stats, 0)
    assert abs(report["all_positive"]["cosine"] - .85) < 1e-12
    assert abs(report["task_margin"] - .55) < 1e-12
    assert abs(report["all_positive"]["reference_relative_error"] - (0.3 ** .5)) < 1e-12


def test_bootstrap_recovery_preserves_signed_code_effect() -> None:
    losses = torch.ones(
        2, len(module.ARMS), module.DOCUMENTS, len(module.CELLS), dtype=torch.float64,
    )
    counts = torch.ones(module.DOCUMENTS, len(module.CELLS), dtype=torch.float64)
    ci = module.CELLS.index("all_positive")
    losses[0, module.ARMS.index("base"), :, ci] = 2.0
    losses[0, module.ARMS.index("reference"), :, ci] = 1.0
    losses[0, module.ARMS.index("score"), :, ci] = 1.4
    original = module.BOOTSTRAP_DRAWS
    module.BOOTSTRAP_DRAWS = 1_000
    try:
        report = module.bootstrap_recovery(losses, counts)
    finally:
        module.BOOTSTRAP_DRAWS = original
    assert report["every_bootstrap_reference_stake_positive"]
    assert abs(report["recovery"] - .6) < 1e-12
    assert report["simultaneous_95_lower"] > .59


def test_only_frozen_score_arms_are_exposed() -> None:
    assert module.PAIRS == ((0, 3), (1, 3))
    assert module.ARMS == ("base", "reference", "score")
    assert module.FACTOR == "score"
    assert module.COMPONENT == "m9"
