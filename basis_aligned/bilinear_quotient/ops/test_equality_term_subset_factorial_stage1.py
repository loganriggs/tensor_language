from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("equality_term_subset_factorial_stage1.py")
SPEC = importlib.util.spec_from_file_location("equality_stage1", PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_arm_identity_and_term_bits() -> None:
    assert len(module.ARMS) == 33
    assert len(set(module.ARMS)) == 33
    assert module.arm_parts("remove:0001") == ("remove", 1)
    assert module.arm_parts("extract:1100") == ("extract", 12)
    assert module._selected(1, 5, 5)
    assert not module._selected(1, 7, 3)
    assert module._selected(12, 8, 3)
    assert module._selected(12, 8, 4)


def test_mobius_recovers_planted_pair_interaction() -> None:
    values = torch.zeros(16, dtype=torch.float64)
    singleton = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    for mask in range(16):
        values[mask] = sum(singleton[bit] for bit in range(4) if mask & (1 << bit))
        if mask & 0b1100 == 0b1100:
            values[mask] -= .07
    dividends = module._mobius(values)
    assert torch.allclose(dividends[[1, 2, 4, 8]], singleton)
    assert torch.isclose(dividends[0b1100], torch.tensor(-.07, dtype=torch.float64))
    assert torch.allclose(
        dividends[[3, 5, 6, 7, 9, 10, 11, 13, 14, 15]],
        torch.zeros(10, dtype=torch.float64), atol=1e-12,
    )
    assert torch.isclose(dividends.sum(), values[-1])


def test_shapley_allocates_every_dividend_once() -> None:
    dividends = torch.arange(16, dtype=torch.float64) / 100
    allocation = module.shapley(dividends)
    assert torch.isclose(allocation.sum(), dividends[1:].sum())


def test_classification_uses_interval_and_floor() -> None:
    assert module.classify(-.01, -.012, -.007) == "redundant"
    assert module.classify(.01, .007, .014) == "complementary"
    assert module.classify(.01, .004, .016) == "additive_or_unresolved"
    assert module.classify(.003, .002, .004) == "additive_or_unresolved"


def test_frozen_masks_and_code_role_exclusion() -> None:
    payload, masks, metadata = module.validate_inputs()
    assert payload["role"] == "final_natural"
    assert list(masks) == list(module.CELLS)
    assert metadata["support"]["all_positive"]["tokens"] == 3084
    assert metadata["support"]["near_positive"]["tokens"] == 719
    assert torch.equal(
        masks["near_positive"] | masks["far_positive"], masks["all_positive"],
    )
    assert "ood_code" not in str(module.ROWS)


def test_full_analysis_recovers_stable_redundant_pair() -> None:
    original_draws = module.BOOTSTRAP_DRAWS
    module.BOOTSTRAP_DRAWS = 1_000
    try:
        arms, documents, cells = len(module.ARMS), module.DOCUMENTS, len(module.CELLS)
        counts = torch.ones(documents, cells, dtype=torch.float64)
        loss = torch.full((arms, documents, cells), 2.0, dtype=torch.float64)
        cell_scale = {
            "matched_positive": 1.0, "matched_negative": .10, "all_positive": 1.0,
            "near_positive": 1.20, "far_positive": .80,
            "one_predecessor_positive": 1.10, "multiple_predecessor_positive": .90,
            "off_target": .02, "all": .05,
        }
        singleton = (.10, .12, .20, .22)
        for cell_index, cell in enumerate(module.CELLS):
            scale = cell_scale[cell]
            for mask in module.SUBSETS:
                recovered = scale * sum(
                    singleton[bit] for bit in range(4) if mask & (1 << bit)
                )
                if mask & 0b1100 == 0b1100:
                    recovered -= scale * .05
                extraction_arm = module.ARMS.index(f"extract:{mask:04b}")
                removal_arm = module.ARMS.index(f"remove:{mask:04b}")
                loss[extraction_arm, :, cell_index] = 3.0 - recovered
                loss[removal_arm, :, cell_index] = 2.0 + .5 * recovered
        stats = {
            "loss_sums": loss, "kl_sums": torch.zeros_like(loss),
            "correct_sums": torch.zeros_like(loss), "counts": counts,
            "replay_relative_squared": 0.0,
        }
        result = module.analyze(stats)
        assert result["primary_pair"]["classification"] == "redundant"
        assert result["primary_pair"]["stable"]
        assert result["shapley_all_positive"]["half_spearman"] > .999
        assert result['pred_d_context_specialization_stable']
        assert result['pred_e_natural_grouping_eligible_for_code_confirmation']
        assert not result["strong_null"]
    finally:
        module.BOOTSTRAP_DRAWS = original_draws
