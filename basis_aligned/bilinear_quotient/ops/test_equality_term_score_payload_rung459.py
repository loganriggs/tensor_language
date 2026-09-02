from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("equality_term_score_payload_rung459.py")
SPEC = importlib.util.spec_from_file_location("equality_score_payload_rung459", PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_scale_normalization_and_direct_factor_cosines() -> None:
    stats = module._empty_scale_stats()
    support = torch.tensor([[[True, False], [True, True]]])
    positive = torch.ones(1, 2, dtype=torch.bool)
    early = {
        "p": torch.tensor([[[2.0, 9.0], [4.0, 6.0]]]),
        "u": torch.full((1, 2, module.D), 3.0),
    }
    late = {
        "p": early["p"] / 2,
        "u": early["u"] * 4,
    }
    module._accumulate_scales(
        stats, module.PAIR_NAMES[0], early, late, support, positive,
    )
    # Populate the other registered pairs so the strict finalizer can run.
    for name in module.PAIR_NAMES[1:]:
        module._accumulate_scales(stats, name, early, late, support, positive)
    report = module._finish_scales(stats)[module.PAIR_NAMES[0]]
    assert abs(report["score_ratio"] - .5) < 1e-12
    assert abs(report["payload_ratio"] - 4.0) < 1e-12
    assert abs(report["direct_score_cosine"] - 1.0) < 1e-12
    assert abs(report["direct_payload_cosine"] - 1.0) < 1e-12


def _planted_fit_objects():
    losses = torch.ones(
        len(module.PAIRS), len(module.ARMS), module.FIT.stop, len(module.CE_CELLS),
        dtype=torch.float64,
    )
    counts = torch.ones(module.FIT.stop, len(module.CE_CELLS), dtype=torch.float64)
    response = module._empty_response_stats()
    response["ref2"].fill_(1.0)
    response["hyb2"].fill_(1.0)
    response["write2"].fill_(100.0)
    response["tokens"].fill_(10)
    ji = module.COMPONENTS.index("m12")
    response["cross"][0, 0, module.RESPONSE_CELLS.index("all_positive"), ji] = .90
    response["cross"][0, 0, module.RESPONSE_CELLS.index("matched_negative"), ji] = .20
    response["cross"][0, 0, module.RESPONSE_CELLS.index("off_target"), ji] = .20
    losses[:, module.ARMS.index("base"), :, :] = 2.0
    losses[:, module.ARMS.index("reference"), :, :] = 1.0
    losses[:, module.ARMS.index("score"), :, :] = 1.4
    losses[:, module.ARMS.index("payload"), :, :] = 1.8
    off = module.CE_CELLS.index("off_target")
    losses[:, :, :, off] = 1.0
    return response, losses, counts


def test_candidate_search_is_exactly_144_and_uses_task_margin() -> None:
    response, losses, counts = _planted_fit_objects()
    candidates, selected = module.make_candidates(response, losses, counts)
    assert len(candidates) == 144
    assert selected is not None
    assert selected["pair"] == "L5H5->L8H3"
    assert selected["factor"] == "score"
    assert selected["component"] == "m12"
    assert abs(selected["task_margin"] - .70) < 1e-12
    # A task-generic response with the same positive cosine must fail.
    ji = module.COMPONENTS.index("m12")
    response["cross"][0, 0, module.RESPONSE_CELLS.index("matched_negative"), ji] = .85
    response["cross"][0, 0, module.RESPONSE_CELLS.index("off_target"), ji] = .85
    _, selected = module.make_candidates(response, losses, counts)
    assert selected is None


def test_response_error_is_reference_relative() -> None:
    stats = module._empty_response_stats(pair_count=1, component_count=1)
    stats["ref2"].fill_(4.0)
    stats["hyb2"].fill_(1.0)
    stats["cross"].fill_(2.0)
    stats["write2"].fill_(100.0)
    stats["tokens"].fill_(5)
    row = module._response_row(stats, 0, 0, 0, 0)
    assert abs(row["cosine"] - 1.0) < 1e-12
    assert abs(row["reference_relative_error"] - .5) < 1e-12
    assert abs(row["reference_rms_over_reader_write_rms"] - .2) < 1e-12
    assert abs(row["hybrid_rms_over_reader_write_rms"] - .1) < 1e-12


def test_bootstrap_recovery_uses_positive_reference_stake() -> None:
    losses = torch.ones(1, len(module.ARMS), 96, len(module.CE_CELLS), dtype=torch.float64)
    counts = torch.ones(96, len(module.CE_CELLS), dtype=torch.float64)
    ci = module.CE_CELLS.index("all_positive")
    losses[0, module.ARMS.index("base"), :, ci] = 2.0
    losses[0, module.ARMS.index("reference"), :, ci] = 1.0
    losses[0, module.ARMS.index("score"), :, ci] = 1.5
    original = module.BOOTSTRAP_DRAWS
    module.BOOTSTRAP_DRAWS = 1_000
    try:
        report = module.bootstrap_recovery(losses, counts, 0, "score")
    finally:
        module.BOOTSTRAP_DRAWS = original
    assert report["every_bootstrap_reference_stake_positive"]
    assert abs(report["recovery"] - .5) < 1e-12
    assert report["simultaneous_95_lower"] > .49


def test_pair_direction_is_always_early_to_layer8() -> None:
    for early, late in module.PAIRS:
        assert module.TERMS[early][1] in (5, 7)
        assert module.TERMS[late][1] == 8
        assert module.TERMS[early][1] < module.TERMS[late][1]
